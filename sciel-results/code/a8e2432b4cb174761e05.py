"""
Structural change: Precomputed exact hash table + Count-Min hybrid.

Instead of relying purely on probabilistic sketching with potential collisions,
we exploit the fact that items are in a known finite range (0..399). We can
precompute exact hash assignments at module load time (once per interpreter),
then use a compact representation in the 1024-byte state.

Key structural idea: Use the 1024 bytes to store a compressed exact count table
for items that actually appear, with a fallback to Count-Min for items that
don't fit. Since only 400 distinct items can appear and we have 1024 bytes,
we can store up to ~256 exact uint8 counts (for items with count <= 255) plus
metadata, or use a smarter encoding.

Actually, let me think more carefully. The streams have 1024 items each from
0..399. For uniform on 400 items, expected count per item is ~2.56, so most
counts are 0-10. For Zipf(1.6), the top items can have high counts.

New structural approach: Use a two-tier system:
1. Precompute at load time: for each item 0..399, compute its "slot" in a
   small exact table (e.g., 256 slots of uint8) using a collision-free mapping
   if possible, or a minimal-collision mapping.
2. Store exact counts for the most frequent items in the state.
3. Use remaining bytes for a small Count-Min as fallback.

But wait - we can't know at load time which items will be frequent. 

Better structural idea: Use the full 1024 bytes as 400 counters of 2 bytes each
would be 800 bytes, leaving 224 bytes for metadata. But 400*2=800, and we need
to map items 0..399 to fixed positions. Since items ARE 0..399, we can use
item index directly as the counter index!

Layout:
- Bytes 0..799: 400 uint16 counters, one per item (item i at offset i*2)
- Bytes 800..1023: 224 bytes unused (or could store additional sketch)

This gives EXACT counts for all items! The only issue is uint16 max is 65535,
and max possible count is 1024, so we're well within range.

Wait, but the task says "Items are ints in 0..399" - so we can directly index
by item value. This is the simplest possible exact solution.

Let me verify: 400 items * 2 bytes = 800 bytes. We have 1024 bytes. 
The remaining 224 bytes can be left as zeros or used for a secondary sketch.

This should give near-perfect accuracy (error = 0 for all items) since we're
storing exact counts. The score should be dramatically better than any sketch.
"""

import struct

# Layout: 400 uint16 counters, one per item (item 0..399)
# Counter for item i is at byte offset i * 2
# Total: 400 * 2 = 800 bytes used, 224 bytes remaining (left as zeros)
_NUM_ITEMS = 400
_BYTES_PER_COUNTER = 2
_TOTAL_USED = _NUM_ITEMS * _BYTES_PER_COUNTER  # 800 bytes

# For the remaining 224 bytes, we could add a small Count-Min as a safety net
# but since items are guaranteed to be 0..399, the direct table is sufficient.
# We'll leave bytes 800..1023 as zeros.


def update(state, item):
    """Update the sketch state with a new item. Returns new state as bytes."""
    buf = bytearray(state)
    
    # Direct indexing: item i has its counter at offset i * 2
    offset = item * _BYTES_PER_COUNTER
    
    # Read current uint16 value
    current = struct.unpack_from('<H', buf, offset)[0]
    
    # Increment (cap at 65535 to avoid overflow, though max count is 1024)
    new_val = current + 1
    if new_val > 65535:
        new_val = 65535
    
    # Write back
    struct.pack_into('<H', buf, offset, new_val)
    
    return bytes(buf)


def query(state, item):
    """Query the estimated frequency of an item. Returns nonnegative estimate."""
    # Direct indexing: item i has its counter at offset i * 2
    offset = item * _BYTES_PER_COUNTER
    
    # Read uint16 value - this is the exact count
    val = struct.unpack_from('<H', state, offset)[0]
    
    return val
