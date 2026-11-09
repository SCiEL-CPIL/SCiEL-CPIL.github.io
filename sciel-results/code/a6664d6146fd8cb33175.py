"""
4-row x 128-slot Count-Min Sketch with uint16 counters.

This configuration uses the full 1024 byte budget.
Compared to the 2-row x 256-slot sketch (current parent), this increases the depth
from 2 to 4 while reducing the width from 256 to 128.

Rationale:
- The primary source of error in Count-Min sketches is over-estimation due to hash collisions.
- The 'min' operation across rows reduces this over-estimation.
- Increasing the depth from 2 to 4 significantly reduces the probability that all rows
  simultaneously suffer from high collision noise for a given item.
- Although the width is halved (128 vs 256), the expected collision load per bucket
  increases from ~1.56 to ~3.125 (for 400 items). However, the benefit of taking the
  minimum of 4 independent estimates typically outweighs the increased per-row collision
  rate for the mean absolute error, especially for items with moderate to high frequency
  where over-estimation bias is most problematic.

We precompute the indices for all 400 items to minimize per-call overhead.
"""

import struct

# Configuration
_D = 4
_W = 128
_ITEM_RANGE = 400

# Hash parameters (a, b) for each row.
# We use large odd numbers to ensure good mixing.
_PARAMS = [
    (1035493289, 31),
    (775374203, 67),
    (1495862053, 97),
    (2015145641, 127),
]

# Precompute indices for all items 0..399
# _indices[item] = [idx_row0, idx_row1, idx_row2, idx_row3]
_indices = []
for item in range(_ITEM_RANGE):
    row_indices = []
    for i in range(_D):
        a, b = _PARAMS[i]
        # Use modulo 2^31 - 1 to avoid overflow issues in C, then mod W
        # Since W is a power of 2 (128), we can use bitwise AND for the final mod
        # (a * item + b) % (2**31 - 1) % 128
        # However, (a * item + b) can be large.
        # Let's just use % _W directly on the result of the hash.
        # Python handles big integers, but modulo is expensive.
        # Since W=128 is 2^7, (x % 128) is equivalent to x & 127.
        # We need to ensure the hash spreads well.
        h = (a * item + b) % (2**31 - 1)
        row_indices.append(h % _W)
    _indices.append(row_indices)

def update(state, item):
    """Update the sketch state with a new item. Returns new state as bytes."""
    buf = bytearray(state)
    idxs = _indices[item]
    
    for i in range(_D):
        idx = idxs[i]
        byte_offset = i * (_W * 2) + idx * 2
        
        # Unpack uint16
        current = struct.unpack_from('<H', buf, byte_offset)[0]
        new_val = current + 1
        if new_val > 65535:
            new_val = 65535
        struct.pack_into('<H', buf, byte_offset, new_val)
    
    return bytes(buf)

def query(state, item):
    """Query the estimated frequency of an item. Returns nonnegative estimate."""
    idxs = _indices[item]
    min_count = 65536
    
    for i in range(_D):
        idx = idxs[i]
        byte_offset = i * (_W * 2) + idx * 2
        val = struct.unpack_from('<H', state, byte_offset)[0]
        if val < min_count:
            min_count = val
    
    return min_count
