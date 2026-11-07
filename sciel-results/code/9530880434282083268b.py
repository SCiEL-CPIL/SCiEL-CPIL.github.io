"""Deterministic exact frequency table: 400 bytes for items 0..399."""

# EVOLVE-BLOCK-START
def update(state, item):
    state = bytearray(state)
    # Items are in 0..399. We use the first 400 bytes as exact counters.
    # Each byte can hold up to 255. Since stream length is 1024, and we have 3 streams,
    # wait, the state persists across the 3 streams? 
    # "Three insertion-only streams of 1024 items... Each 256-item chunk runs in a fresh interpreter... only the returned bytes persist"
    # The state is cumulative. Total items = 3 * 1024 = 3072.
    # Max count for a single item in the worst case (if all 3072 items were the same) would be 3072, which exceeds 255.
    # However, the streams are:
    # 1. Uniform on 0..399 (1024 items). Expected count per item ~ 1024/400 = 2.56. Max likely < 20.
    # 2. Zipf(1.6) modulo 400 (1024 items). Item 0 will be very hot. 
    #    Zipf(1.6) on 400 items. P(0) is high. 
    #    Sum_{i=1}^{400} 1/i^1.6 approx integral. 
    #    Let's check max possible count.
    #    If the distribution is Zipf, the most frequent item could appear many times.
    #    But wait, the task says "Items are ints in 0..399". 
    #    If I just use 1 byte, I might overflow.
    #    Let's re-read carefully: "state may never exceed 1024 bytes".
    #    "Three insertion-only streams of 1024 items".
    #    The state is updated sequentially. 
    #    Stream 1: 1024 items.
    #    Stream 2: 1024 items.
    #    Stream 3: 1024 items.
    #    Total 3072 updates.
    #    If an item appears in all 3 streams, its count adds up.
    #    In Stream 1 (Uniform), item 0 appears ~2-3 times.
    #    In Stream 2 (Zipf), item 0 is the most frequent. 
    #    In Stream 3 (Mixed: 512 uniform 0..19, 512 uniform 150..399), item 0 appears ~512/20 = 25.6 times on average.
    #    So item 0 could have count: 3 (stream 1) + High (stream 2) + 26 (stream 3).
    #    What is the max count in Zipf(1.6) of 1024 samples?
    #    The probability of item 0 is 1 / sum(1/i^1.6).
    #    sum_{i=1}^{400} i^-1.6 is roughly 1 + 0.35 + ... 
    #    Let's approximate. The sum is dominated by the first few terms.
    #    1^(-1.6) = 1
    #    2^(-1.6) = 0.33
    #    3^(-1.6) = 0.17
    #    4^(-1.6) = 0.10
    #    Sum ~ 1.6-1.8.
    #    So P(0) ~ 1/1.7 ~ 0.58.
    #    Expected count in Stream 2 for item 0 is 1024 * 0.58 ~ 594.
    #    This exceeds 255!
    #    So 1 byte per item is NOT safe for all items, specifically the hot ones in Zipf.
    
    # Strategy: Use a hybrid approach or wider counters for hot items?
    # Or, realize that the "fresh interpreter" per 256-item chunk means the state is passed as bytes.
    # If I use 2 bytes per item for the first few items (hot) and 1 byte for the rest?
    # Or simply use 2 bytes for ALL items? 400 items * 2 bytes = 800 bytes. Fits in 1024.
    # Let's use 2 bytes (uint16) per item. Max count 65535. Total max 3072. Safe.
    
    # Layout: 
    # Bytes 0-799: 400 uint16 little-endian counters.
    # Bytes 800-1023: unused.
    
    # Update:
    # Read current uint16, increment, write back.
    
    val = state[2*item] | (state[2*item+1] << 8)
    val += 1
    state[2*item] = val & 0xFF
    state[2*item+1] = (val >> 8) & 0xFF
    return bytes(state)

def query(state, item):
    # Read uint16 from state
    val = state[2*item] | (state[2*item+1] << 8)
    return float(val)
# EVOLVE-BLOCK-END
