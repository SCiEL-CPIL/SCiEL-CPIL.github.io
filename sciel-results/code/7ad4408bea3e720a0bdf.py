"""Improved sketch: 2 rows x 256 uint16 counters, min-of-2 query."""
import struct
import time

# Precompute hash parameters for 2 rows
# Using different primes than seed to avoid correlation
_PRIMES = (1035493289, 775374203)
_M = 2147483647  # 2^31 - 1
_W = 256         # width (number of buckets per row)
_D = 2           # depth (number of rows)

def _compute_indices(item):
    """Compute the 2 bucket indices for an item."""
    idxs = []
    for row in range(_D):
        idx = ((item * _PRIMES[row] + 31 * row) % _M) % _W
        idxs.append(idx)
    return idxs

def update(state, item):
    """Update the sketch state by incrementing counters for the given item."""
    idxs = _compute_indices(item)
    # Work on a copy
    buf = bytearray(state)
    for i, idx in enumerate(idxs):
        # uint16 little-endian at offset i*256*2 + idx*2
        offset = i * (_W * 2) + idx * 2
        val = struct.unpack_from('<H', buf, offset)[0]
        if val < 65535:
            val += 1
        struct.pack_into('<H', buf, offset, val)
    return bytes(buf)

def query(state, item):
    """Query the estimated frequency of an item."""
    idxs = _compute_indices(item)
    min_val = 65536
    for i, idx in enumerate(idxs):
        offset = i * (_W * 2) + idx * 2
        val = struct.unpack_from('<H', state, offset)[0]
        if val < min_val:
            min_val = val
    return min_val
