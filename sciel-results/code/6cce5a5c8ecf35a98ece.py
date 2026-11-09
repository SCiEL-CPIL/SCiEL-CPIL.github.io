"""
Memory-bounded streaming frequency sketch using a hybrid approach:
- Exact counters for the top K items (tracked via a small sketch)
- Count-Min sketch for the remaining items

The key insight: for Zipf and bimodal distributions, a small number of items
dominate the frequency. By tracking these exactly, we eliminate error for the
most important items. The CM sketch handles the long tail.

State layout (1024 bytes):
- Bytes 0..63: 16 uint32 counters for exact tracking (top 16 items)
  Actually, we need to store which items are tracked. Let's use:
- Bytes 0..15: 16 tracked item IDs (uint8 each)
- Bytes 16..80: 16 uint32 exact counts (64 bytes)
- Bytes 80..1024: CM sketch: 12 rows x 64 uint32 counters = 12*64*4 = 3072... too big.

Let me recalculate. We have 1024 bytes total.
CM sketch: 4 rows x 64 cols x 4 bytes = 1024 bytes. That's the whole budget.

Alternative: use a smaller CM sketch and dedicate space to exact counters.
- 2 rows x 64 cols x 4 bytes = 512 bytes for CM
- Remaining 512 bytes for exact tracking.

Actually, let's think about this differently. The seed uses 4x64 CM sketch = 1024 bytes.
The hybrid approach that scored 14.13 (vs seed 1.0) used exact/sketch.

Wait, the score is seed/value, so 14.13 means value = seed/14.13, which is much better.
The seed value is the raw mean absolute error. So lower raw error = higher score.

Let me design a hybrid:
- Use a small number of exact counters for high-frequency items
- Use a CM sketch for the rest

State layout (1024 bytes):
- Bytes 0..79: 20 uint8 item IDs + 20 uint32 counts? No, that's 20+80=100 bytes.
  Let's do: 32 tracked items.
  - Bytes 0..31: 32 uint8 item IDs
  - Bytes 32..159: 32 uint32 counts (128 bytes)
  Total: 160 bytes for exact tracking.
- Bytes 160..1023: 864 bytes for CM sketch.
  864 / 4 = 216 uint32 counters.
  We can do 3 rows x 72 cols, or 4 rows x 54 cols, etc.
  Let's do 4 rows x 54 cols = 216 counters.

Actually, let me reconsider. The challenge is that we don't know in advance which items will be frequent. We need to discover them online.

Approach: Use a Count-Min sketch to estimate frequencies, and maintain a small set of "candidate" items that we track exactly. When an item's CM estimate exceeds a threshold, promote it to exact tracking.

But we have limited space. Let's use:
- 4 rows x 64 cols CM sketch = 1024 bytes... that's all we have.

We need to sacrifice some CM precision for exact tracking. Let's use:
- 3 rows x 64 cols CM sketch = 768 bytes
- 256 bytes for exact tracking:
  - 64 uint8 item IDs (64 bytes)
  - 64 uint32 counts (256 bytes) - wait, 64*4=256, total 320. Too much.
  
Let me use:
- 3 rows x 48 cols CM sketch = 576 bytes
- 448 bytes for exact:
  - 64 uint8 IDs (64 bytes)
  - 64 uint32 counts (256 bytes)
  - Remaining 128 bytes for... maybe a smaller secondary sketch or metadata.

Actually, let's keep it simpler. Use:
- 2 rows x 64 cols CM = 512 bytes
- 512 bytes exact:
  - 64 uint8 IDs
  - 64 uint32 counts = 256 bytes
  - Total 320 bytes, leaving 192 bytes unused or for a 3rd CM row.

Let me just use:
- 3 rows x 64 cols CM = 768 bytes
- 64 uint8 IDs + 64 uint32 counts = 320 bytes... 768+320=1088 > 1024.

OK: 2 rows x 64 cols CM = 512 bytes
- 64 uint8 IDs = 64 bytes
- 64 uint32 counts = 256 bytes
- Total: 832 bytes. 192 bytes left.

Let me add a 3rd row of 48 cols: 48*4=192 bytes. Total: 1024. 

So:
- Rows 0-1: 64 cols each (for CM)
- Row 2: 48 cols (for CM)
- 64 tracked item IDs (uint8)
- 64 tracked counts (uint32)

Hashing: use the same primes as the seed for consistency, but adjust for different col counts.

For query:
- If item is in the tracked set, return exact count.
- Otherwise, return min over all 3 CM rows.

For update:
- Increment the CM counters for the item.
- If item is in tracked set, increment its exact count.
- Otherwise, check if it should be promoted: if its CM estimate exceeds the minimum tracked count (and there's space), promote it.

Actually, promotion logic is tricky in a streaming setting. Let's simplify: just track the first 64 distinct items we see, OR use a simpler heuristic.

Better approach: Since we know items are 0..399, we can use a fixed assignment. But that doesn't help.

Let me try: track items whose CM estimate is above a threshold. On each update, after updating CM, if the item isn't tracked and its CM estimate > max_untracked_threshold, promote the least-frequent tracked item.

This is getting complex. Let me just implement a clean version.
"""

import struct
import time

# Primes for hashing (same as seed)
P0 = 1035493289
P1 = 775374203
P2 = 1495862053
P3 = 2015145641

# State layout:
# Bytes 0..63: 64 uint8 tracked item IDs
# Bytes 64..319: 64 uint32 tracked counts
# Bytes 320..831: 2 rows x 64 uint32 CM counters (row 0: 320..575, row 1: 576..831)
# Bytes 832..1023: 1 row x 48 uint32 CM counters (row 2: 832..1023)

TRACKED_COUNT = 64
CM_COLS_0 = 64
CM_COLS_1 = 64
CM_COLS_2 = 48

def _hash0(item, row):
    return ((item * P0 + 31 * row) % 2147483647) % CM_COLS_0

def _hash1(item, row):
    return ((item * P1 + 31 * row) % 2147483647) % CM_COLS_1

def _hash2(item, row):
    return ((item * P2 + 31 * row) % 2147483647) % CM_COLS_2

def update(state, item):
    """Update the sketch state with a new item."""
    state = bytearray(state)
    
    # Parse tracked items and counts
    tracked_ids = list(state[0:64])
    tracked_counts = list(struct.unpack_from('<64I', state, 64))
    
    # Parse CM rows
    cm_row0 = list(struct.unpack_from('<64I', state, 320))
    cm_row1 = list(struct.unpack_from('<64I', state, 576))
    cm_row2 = list(struct.unpack_from('<48I', state, 832))
    
    # Update CM counters
    h0 = _hash0(item, 0)
    h1 = _hash1(item, 0)
    h2 = _hash2(item, 0)
    cm_row0[h0] += 1
    cm_row1[h1] += 1
    cm_row2[h2] += 1
    
    # Check if item is already tracked
    if item in tracked_ids:
        idx = tracked_ids.index(item)
        tracked_counts[idx] += 1
    else:
        # Check if we should promote this item
        # Get current CM estimate for this item
        cm_est = min(cm_row0[h0], cm_row1[h1], cm_row2[h2])
        
        # Find the least frequent tracked item
        min_count = min(tracked_counts)
        min_idx = tracked_counts.index(min_count)
        
        # Promote if new item's estimate exceeds the least tracked count
        if cm_est > min_count:
            tracked_ids[min_idx] = item
            tracked_counts[min_idx] = cm_est  # Start with the CM estimate
    
    # Rebuild state
    new_state = bytearray(1024)
    new_state[0:64] = bytes(tracked_ids)
    struct.pack_into('<64I', new_state, 64, *tracked_counts)
    struct.pack_into('<64I', new_state, 320, *cm_row0)
    struct.pack_into('<64I', new_state, 576, *cm_row1)
    struct.pack_into('<48I', new_state, 832, *cm_row2)
    
    return bytes(new_state)

def query(state, item):
    """Query the estimated frequency of an item."""
    state = bytes(state)
    
    # Parse tracked items and counts
    tracked_ids = list(state[0:64])
    tracked_counts = list(struct.unpack_from('<64I', state, 64))
    
    # Check if item is tracked
    if item in tracked_ids:
        idx = tracked_ids.index(item)
        return float(tracked_counts[idx])
    
    # Parse CM rows
    cm_row0 = list(struct.unpack_from('<64I', state, 320))
    cm_row1 = list(struct.unpack_from('<64I', state, 576))
    cm_row2 = list(struct.unpack_from('<48I', state, 832))
    
    h0 = _hash0(item, 0)
    h1 = _hash1(item, 0)
    h2 = _hash2(item, 0)
    
    return float(min(cm_row0[h0], cm_row1[h1], cm_row2[h2]))
