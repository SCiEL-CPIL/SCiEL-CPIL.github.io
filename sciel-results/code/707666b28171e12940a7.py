"""Memory-bounded streaming frequency sketches - 4-row x 128-col Count-Min sketch."""

# EVOLVE-BLOCK-START

import struct

# 4 rows of 128 uint32 counters = 4 * 128 * 4 = 2048 bytes... too big.
# Use 4 rows of 64 uint32 counters = 4 * 64 * 4 = 1024 bytes.
_ROWS = 4
_COLS = 64
_HASH_P = (1035493289, 775374203, 1495862053, 2015145641)
_HASH_OFF = (31, 57, 97, 131)
_MOD = 2147483647

def update(state, item):
    """Increment one counter per row for item in the 4x64 CM sketch."""
    b = bytearray(state)
    for row in range(_ROWS):
        idx = ((item * _HASH_P[row] + _HASH_OFF[row]) % _MOD) % _COLS
        off = (row * _COLS + idx) * 4
        val = struct.unpack_from('<I', b, off)[0]
        struct.pack_into('<I', b, off, (val + 1) & 0xFFFFFFFF)
    return bytes(b)

def query(state, item):
    """Return the min across 4 rows for item from the CM sketch."""
    best = 0xFFFFFFFF
    for row in range(_ROWS):
        idx = ((item * _HASH_P[row] + _HASH_OFF[row]) % _MOD) % _COLS
        off = (row * _COLS + idx) * 4
        val = struct.unpack_from('<I', state, off)[0]
        if val < best:
            best = val
    return float(best)

# EVOLVE-BLOCK-END
