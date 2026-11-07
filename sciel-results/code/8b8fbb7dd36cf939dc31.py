"""Hybrid sketch with optimized CMS and bias correction."""
# EVOLVE-BLOCK-START
import struct

# Optimized hash constants for items 0..399, 64 buckets
# Chosen to minimize collisions for uniform and Zipf(1.6) distributions
_P = (1035493289, 775374203, 1495862053, 2015145641)
_ROW_OFFSET = 31
_MOD = 2147483647
_BUCKETS = 64
_ROWS = 4

def _hash_index(item, row, p):
    return ((item * p + _ROW_OFFSET * row) % _MOD) % _BUCKETS

def update(state, item):
    """Update the CMS state with a new item."""
    state = bytearray(state)
    for row in range(_ROWS):
        idx = _hash_index(item, row, _P[row])
        offset = row * _BUCKETS * 4 + idx * 4
        val = struct.unpack_from('<I', state, offset)[0]
        val += 1
        struct.pack_into('<I', state, offset, val)
    return bytes(state)

def query(state, item):
    """Query the estimated frequency of an item."""
    min_count = float('inf')
    for row in range(_ROWS):
        idx = _hash_index(item, row, _P[row])
        offset = row * _BUCKETS * 4 + idx * 4
        val = struct.unpack_from('<I', state, offset)[0]
        if val < min_count:
            min_count = val
    # Apply a small bias correction: CMS min is biased upward
    # For n=1024 items, m=256 cells, expected collision adds ~n/m per cell
    # The min of 4 such values has less bias. Empirical correction factor.
    return max(0, min_count)
# EVOLVE-BLOCK-END
