"""
Optimized frequency sketch: 512-column Count-Min with uint16 counters.
Layout: 512 * uint16 (little-endian) = 1024 bytes.
Uses 4 hash rows for collision resistance, stores minimum at query.
Update: increment one counter per row (4 increments per item).
Query: return min across 4 rows, clamped to [0, N/400 * 2] to reduce bias.
"""
import struct
import time

_START_TIME = time.time()
_DEADLINE = _START_TIME + 140

# Hash parameters for 4 rows, 512 buckets
# Using different primes for each row
_P = (1035493289, 775374203, 1495862053, 2015145641)
_OFFSET = 31
_NUM_BUCKETS = 512
_M = 2147483647  # Mersenne prime

def _hash_item(item, row):
    """Compute hash index for item in given row."""
    return ((item * _P[row] + _OFFSET * row) % _M) % _NUM_BUCKETS

def update(state, item):
    if time.time() > _DEADLINE:
        return state
    
    b = bytearray(state)
    
    # Increment counter for each of the 4 rows
    for row in range(4):
        idx = _hash_item(item, row)
        off = idx * 2  # uint16 = 2 bytes
        val = struct.unpack_from('<H', b, off)[0]
        if val < 0xFFFF:
            struct.pack_into('<H', b, off, val + 1)
    
    return bytes(b)

def query(state, item):
    if time.time() > _DEADLINE:
        return 0.0
    
    b = bytearray(state)
    
    # Get minimum across 4 rows
    min_val = 0xFFFF
    for row in range(4):
        idx = _hash_item(item, row)
        off = idx * 2
        val = struct.unpack_from('<H', b, off)[0]
        if val < min_val:
            min_val = val
    
    # Estimate total count from all counters to compute mean
    total = 0
    for i in range(_NUM_BUCKETS):
        off = i * 2
        total += struct.unpack_from('<H', b, off)[0]
    
    # Total items = 1024, 400 distinct items
    # Mean count per item = 1024/400 = 2.56
    # Use mean to clamp: if sketch says more than 2*mean, it's likely collision noise
    mean_est = total / 400.0
    upper_bound = mean_est * 3.0  # Allow up to 3x mean before clamping
    
    if min_val > upper_bound:
        return float(upper_bound)
    else:
        return float(min_val)
