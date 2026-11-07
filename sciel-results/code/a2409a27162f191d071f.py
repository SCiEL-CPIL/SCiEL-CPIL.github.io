"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import struct

PRIMES = (1035493289, 775374203, 1495862053, 2015145641)
MOD = 2147483647
WIDTH = 64
ROWS = 4

def _get_offset(state, item, row):
    p = PRIMES[row]
    idx = ((item * p + 31 * row) % MOD) % WIDTH
    # Each counter is 4 bytes (uint32).
    # Row r starts at r * WIDTH * 4
    return row * WIDTH * 4 + idx * 4

def update(state, item):
    state = bytearray(state)
    for row in range(ROWS):
        offset = _get_offset(state, item, row)
        val = struct.unpack_from('<I', state, offset)[0]
        struct.pack_into('<I', state, offset, val + 1)
    return bytes(state)

def query(state, item):
    min_val = float('inf')
    for row in range(ROWS):
        offset = _get_offset(state, item, row)
        val = struct.unpack_from('<I', state, offset)[0]
        if val < min_val:
            min_val = val
    return min_val
# EVOLVE-BLOCK-END
