"""Count-Min sketch with 8 rows of 64 uint16 counters (1024 bytes total)."""
import struct

# Precompute hash parameters for 8 rows
MULTS = [
    1035493289,
    775374203,
    1495862053,
    2015145641,
    2654435761,
    1103515245,
    1234567891,
    2166136261,
]
HASH_MOD = 2147483647
W = 64
D = 8


def update(state, item):
    """Increment counters for item in the sketch state."""
    b = bytearray(state)
    for row in range(D):
        idx = ((item * MULTS[row] + 31 * row) % HASH_MOD) % W
        offset = row * W * 2 + idx * 2
        val = struct.unpack_from('<H', b, offset)[0]
        if val < 65535:
            val += 1
        struct.pack_into('<H', b, offset, val)
    return bytes(b)


def query(state, item):
    """Return the minimum count across all rows for the given item."""
    min_val = 65536
    for row in range(D):
        idx = ((item * MULTS[row] + 31 * row) % HASH_MOD) % W
        offset = row * W * 2 + idx * 2
        val = struct.unpack_from('<H', state, offset)[0]
        if val < min_val:
            min_val = val
    return min_val
