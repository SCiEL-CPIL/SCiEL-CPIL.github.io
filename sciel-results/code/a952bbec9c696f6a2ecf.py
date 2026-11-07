"""Hybrid frequency estimator: 4-bit counters for hot items + Count-Min sketch for others.

Layout (1024 bytes total):
- Bytes 0..127: 128 x 4-bit counters (256 items total) for items 0..19 and 150..399
  (the "hot" items under the known stream distributions).
  Each pair of bytes stores two 4-bit counters.
  Item mapping:
    items 0..19  -> slot 0..19
    items 150..399 -> slot 20..249
- Bytes 128..1023: 896 bytes for a Count-Min sketch.
  We use 7 rows x 32 columns of uint32 little-endian = 7*32*4 = 896 bytes.
  The CM sketch handles items 20..149 (130 items) and also provides overflow
  detection/correction for hot items if their 4-bit counter saturates.

4-bit counters cap at 15. For the third stream, items 0..19 get ~25.6 on average,
so they will saturate. For Zipf(1.6), the top items may also exceed 15.
When a 4-bit counter saturates at 15, we fall back to the CM sketch estimate
for that item, which should be more accurate than a flat 15.

Actually, let me reconsider. The key insight from the current program is that
8-bit direct addressing gives score 29.7. The main error source is saturation
of 8-bit counters for Zipf hot items (counts > 255).

Let me use a different approach:
- Use the full 1024 bytes more cleverly.
- 400 items need at most ~10 bits each (max count 1024).
- 400 * 10 = 4000 bits = 500 bytes for exact 10-bit counters.
- Remaining 524 bytes for a CM sketch to correct for any edge cases.

Actually, 10-bit counters: 400 items * 10 bits = 4000 bits = 500 bytes exactly.
Max count per item in a 1024-item stream is 1024, which needs 11 bits (0..1024).
So 10 bits caps at 1023. Only an item appearing 1024 times would overflow,
which is extremely unlikely for any of the three streams.

Let me use 11-bit counters to be safe: 400 * 11 = 4400 bits = 550 bytes.
Remaining 474 bytes for a CM sketch.

Actually, the simplest and most accurate approach: use 16-bit counters.
400 * 16 = 6400 bits = 800 bytes. Remaining 224 bytes for a small CM sketch
or just padding. 16-bit counters go up to 65535, far exceeding any possible
count of 1024. This gives EXACT counts for all items.

Wait - the current program already gets 29.7 with 8-bit exact counters. The issue
is that 8 bits cap at 255. If I use 16-bit counters, I get exact counts for all
items in all streams, which should be much better than 29.7.

Let me verify: 400 items * 2 bytes = 800 bytes. Remaining 224 bytes unused.
This should give near-zero error for all three streams.
"""

def update(state, item):
    """Increment the 16-bit counter for the given item."""
    ba = bytearray(state)
    # Each item gets 2 bytes (little-endian uint16)
    offset = item * 2
    current = ba[offset] | (ba[offset + 1] << 8)
    if current < 65535:
        current += 1
    ba[offset] = current & 0xFF
    ba[offset + 1] = (current >> 8) & 0xFF
    return bytes(ba)

def query(state, item):
    """Return the exact 16-bit count for the given item."""
    offset = item * 2
    return state[offset] | (state[offset + 1] << 8)
