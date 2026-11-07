"""
Hybrid exact + CM sketch with optimized column count.

State layout (1024 bytes):
- Bytes 0-255: 256 dedicated uint8 counters for items 0-255 (exact counting).
- Bytes 256-1023: 768 bytes for a Count-Min sketch (12 rows x 64 columns of uint8).

Rationale:
- Stream 1 (uniform 0..399): Items 0-255 are exact. Items 256-399 use CM.
  With 64 columns, expected collision per cell is ~368/64 ≈ 5.75.
  True count per item: ~2.56. Estimate ≈ 2.56 + 5.75 ≈ 8.3. Error ≈ 5.75.
  
- Stream 2 (Zipf 1.6): Items 0-255 capture most mass (exact). Items 256-399
  have very low counts. Total mass in sketch is small (~15 items), so
  collision is ~15/64 ≈ 0.23. Very accurate.
  
- Stream 3: First 512 uniform on 0-19 (all exact). Second 512 uniform on 150-399.
  Items 150-255 are exact. Items 256-399 use CM. Total mass in sketch:
  512 * 144/250 ≈ 295. Collision: 295/64 ≈ 4.6. True count: ~2.05.
  Estimate ≈ 6.65. Error ≈ 4.6.

The 12x64 layout gives much better collision properties than 12x16 while
still using the same 768 bytes for the sketch portion.
"""

import struct

_NUM_EXACT = 256
_SKETCH_ROWS = 12
_SKETCH_COLS = 64
_SKETCH_OFFSET = 256

# Primes for hashing
_PRIMES = [
    1035493289,
    775374203,
    1495862053,
    2015145641,
    1234567891,
    987654323,
    314159265,
    271828183,
    410332281,
    524287001,
    628318531,
    734003371,
]
_SEEDS = [
    31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79
]
_MOD = 2147483647


def _sketch_hash(item, row):
    """Hash item to a column in the sketch."""
    p = _PRIMES[row]
    s = _SEEDS[row]
    return ((item * p + s) % _MOD) % _SKETCH_COLS


def update(state, item):
    """
    state: bytes of length 1024
    item: int in 0..399
    returns: new state as bytes
    """
    buf = bytearray(state)
    
    if item < _NUM_EXACT:
        # Exact counter for items 0-255
        val = buf[item]
        if val < 255:
            buf[item] = val + 1
    else:
        # CM sketch for items 256-399
        for row in range(_SKETCH_ROWS):
            col = _sketch_hash(item, row)
            offset = _SKETCH_OFFSET + row * _SKETCH_COLS + col
            val = buf[offset]
            if val < 255:
                buf[offset] = val + 1
    
    return bytes(buf)


def query(state, item):
    """
    state: bytes of length 1024
    item: int in 0..399
    returns: nonnegative frequency estimate
    """
    if item < _NUM_EXACT:
        # Exact count
        return state[item]
    else:
        # CM sketch: take min of all rows
        best = 255
        for row in range(_SKETCH_ROWS):
            col = _sketch_hash(item, row)
            offset = _SKETCH_OFFSET + row * _SKETCH_COLS + col
            val = state[offset]
            if val < best:
                best = val
        return best
