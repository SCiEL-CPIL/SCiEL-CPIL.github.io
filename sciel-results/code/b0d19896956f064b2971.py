import struct
import time

# We will use a hybrid approach:
# 1. An exact lookup table for the first 400 bytes (items 0..399).
#    This allows us to store exact counts for each item, up to 255.
#    Since the max count in any stream is 1024, this will overflow for frequent items.
#    However, for the "uniform" and "mixed" streams, many items have low counts.
#    For the Zipf stream, high-frequency items will overflow.
#    Wait, if we just use a lookup table, we lose information on overflow.
#    Let's look at the inspirations. Inspiration 1 and 2 are basically the same as current.
#    Inspiration 3 uses 8 rows x 64 cols.
#    The directive says: "STRUCTURAL change: a different design or modelling family...
#    Strategy to try: Since the item range (0–399) and stream length (1024) guarantee that no item’s count exceeds 255... 
#    Wait, 1024 items, if all are the same item, count is 1024. 1024 > 255. 
#    So the premise "no item's count exceeds 255" is FALSE for the worst case (Zipf with high skew or uniform if unlucky? No, uniform is low counts).
#    Zipf(1.6) on 0..399. The most frequent item will have a count proportional to 1/1^1.6 * 1024 / sum(1/i^1.6).
#    Sum is approx zeta(1.6) ~ 2.5? No, sum from 1 to 400.
#    Actually, let's look at the "Knowledge from this run" note:
#    "Replaces probabilistic sketch with exact count lookup table using the full 1024-byte state... Measured: 11.76 -> 22.74."
#    This implies that a pure lookup table (1 byte per item) resulted in a WORSE score (higher error) than the sketch? 
#    Wait, "Measured: 11.76 -> 22.74". Lower is better. So 22.74 is worse than 11.76?
#    The seed score is 0.7613. The metric is "mean ... |estimate - true| / 1024".
#    If the seed score is 0.76, the raw error is 0.76 * 1024 ~ 778? No, the score IS the raw value? 
#    "Scoring (lower is better): per stream, mean ... |estimate - true count| / 1024".
#    So a score of 0.76 means mean absolute error is 0.76.
#    The note says "11.76 -> 22.74". This looks like a different scale or a different metric? 
#    Or maybe the note refers to a different island's internal metric? 
#    Regardless, the directive explicitly suggests: "Strategy to try in this change: ... store exact frequencies in a 400-byte lookup table ... achieving a theoretical mean absolute [error of 0?]"
#    But the overflow problem is real. If an item appears 300 times, we can't store 300 in 1 byte.
#    We have 1024 bytes total.
#    Items: 400.
#    If we use 1 byte per item, we have 400 bytes for counts.
#    We have 624 bytes left.
#    We can use the remaining 624 bytes to store "overflow" bits or a secondary sketch for items that overflow.
#    Or, we can use a larger counter for specific items? No, we don't know which ones will overflow in advance.
#    
#    Alternative Structural Idea:
#    Use 2 bytes per item for the first 256 items? No, items are 0..399.
#    
#    Let's reconsider the "Exact Lookup" with overflow handling.
#    We have 1024 bytes.
#    Let's allocate 1 byte for each of the 400 items (bytes 0..399).
#    These store `count % 256`.
#    We also need to know if the count was >= 256, and by how much.
#    Max count is 1024.
#    We can use the remaining 624 bytes to store "high bits" or a sketch of the overflow.
#    
#    Better idea:
#    Use the 1024 bytes to store a "Hybrid" structure.
#    Since we know the item range is 0..399, we can just use a dictionary-like array.
#    But we are limited to 1024 bytes of state.
#    
#    What if we use 2 bytes per item for the first 256 items, and 1 byte for the rest?
#    No, that's arbitrary.
#    
#    Let's look at the failure of the pure lookup table in the "Knowledge" note.
#    "Measured: 11.76 -> 22.74". If this is MAE, 22.74 is huge. Why would exact counts be so bad?
#    Because of overflow! If an item appears 300 times, it wraps to 44. Error = 256.
#    For Zipf, a few items have high counts. The error for those items is large.
#    The CM sketch handles high counts better because it distributes them, but adds noise to low counts.
#    
#    Can we fix the overflow?
#    We have 624 bytes spare.
#    We can use these 624 bytes to store a "high-order" bitmap or counter for items that exceed 255.
#    Specifically, let `low[i]` be the low 8 bits of the count for item `i`.
#    Let `high[i]` be the high bits.
#    We don't have space for `high[i]` for all 400 items if we use 1 byte each (another 400 bytes).
#    But we only have 624.
#    We can store `high[i]` for the first 624/8 = 78 items? No, `high[i]` needs more than 1 bit.
#    Max count 1024. `low` is 0..255. `high` is 0..3 (since 1024/256 = 4).
#    So `high[i]` needs 2 bits.
#    We can pack 4 items' high bits into 1 byte.
#    400 items * 2 bits = 800 bits = 100 bytes.
#    Total bytes needed: 400 (low) + 100 (high) = 500 bytes.
#    We have 1024 bytes. This fits easily!
#    
#    Plan:
#    State layout:
#    Bytes 0..399: `low[i]` (8 bits) for item i.
#    Bytes 400..499: `high[i]` (2 bits) for item i. 
#       Byte 400 contains high bits for items 0,1,2,3.
#       Item 0: bits 0-1
#       Item 1: bits 2-3
#       Item 2: bits 4-5
#       Item 3: bits 6-7
#       ...
#    Total 500 bytes used. Remaining 524 bytes are zero.
#    
#    Update:
#    1. Read `low[item]`.
#    2. Read `high[item]` from the packed byte.
#    3. Compute `count = (high << 8) + low`.
#    4. Increment `count`.
#    5. If `count >= 256`:
#         `low[item] = count & 0xFF`
#         `high[item] = (count >> 8) & 0x03` (since max is 1024, 2 bits are enough: 0,1,2,3,4? 1024 is 4*256. So we need 0..4. 3 bits? 1024/256 = 4. 4 fits in 3 bits (0-7). 2 bits only go to 3. 
#         Wait, 1024 items. If all 1024 are the same item, count=1024.
#         1024 = 4 * 256.
#         So `high` can be 4.
#         2 bits can represent 0,1,2,3. 4 requires 3 bits.
#         Let's use 3 bits per item for `high`.
#         400 items * 3 bits = 1200 bits = 150 bytes.
#         Total: 400 + 150 = 550 bytes. Still fits.
#         Actually, let's just use 4 bits to be safe and simple?
#         4 bits * 400 = 1600 bits = 200 bytes.
#         Total: 400 + 200 = 600 bytes. Fits.
#         Or even simpler: Just use 1 byte for `high`?
#         400 bytes for `low`, 400 bytes for `high`. Total 800 bytes.
#         This is very simple and robust.
#         `count = (high[item] << 8) + low[item]`.
#         Max count 1024 fits in 16 bits (65535).
#         We have 1024 bytes. 800 bytes used. 224 bytes spare.
#    
#    This gives EXACT counts for all items, as long as count < 65536.
#    Since max count is 1024, this is always true.
#    Therefore, `query` will return the EXACT true count.
#    Mean Absolute Error will be 0.
#    Score will be 0.
#    
#    Why did the previous "lookup table" fail?
#    The note said "Measured: 11.76 -> 22.74".
#    Maybe it only used 1 byte per item (400 bytes total) and ignored the overflow?
#    If so, error for high-frequency items is huge.
#    By adding the `high` bytes, we fix the overflow.
#    
#    Let's implement this "Exact 16-bit Counter per Item" approach.
#    It is a structural change from "Probabilistic Sketch" to "Deterministic Exact Counter".
#    It uses the full state space efficiently.
#    
#    Implementation details:
#    - State is 1024 bytes.
#    - For item `i` (0..399):
#      - `low[i]` is at byte `i`.
#      - `high[i]` is at byte `400 + i`.
#    - Update:
#      - `low_val = state[i]`
#      - `high_val = state[400+i]`
#      - `count = (high_val << 8) | low_val`
#      - `count += 1`
#      - `new_low = count & 0xFF`
#      - `new_high = (count >> 8) & 0xFF`
#      - Write back to buffer.
#    - Query:
#      - `low_val = state[i]`
#      - `high_val = state[400+i]`
#      - `return (high_val << 8) | low_val`
#    
#    This is extremely fast and accurate.
#    
#    Let's double check constraints.
#    - Items 0..399.
#    - State 1024 bytes.
#    - Update returns bytes.
#    - Query returns nonnegative int.
#    
#    This should score 0.0 (or very close to it, limited by floating point? No, integers).
#    The score is mean absolute error / 1024. If error is 0, score is 0.
#    This is a massive improvement over 0.76.

def update(state, item):
    buf = bytearray(state)
    
    # Item is 0..399
    # Low byte at index item
    # High byte at index 400 + item
    
    low_idx = item
    high_idx = 400 + item
    
    low_val = buf[low_idx]
    high_val = buf[high_idx]
    
    count = (high_val << 8) | low_val
    count += 1
    
    new_low = count & 0xFF
    new_high = (count >> 8) & 0xFF
    
    buf[low_idx] = new_low
    buf[high_idx] = new_high
    
    return bytes(buf)

def query(state, item):
    low_idx = item
    high_idx = 400 + item
    
    low_val = state[low_idx]
    high_val = state[high_idx]
    
    return (high_val << 8) | low_val
