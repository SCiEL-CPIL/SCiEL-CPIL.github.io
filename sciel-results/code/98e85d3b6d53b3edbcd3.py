"""Combined: 8-row CMS with precomputed XOR-shift hash table."""
import struct

DEPTH = 8
WIDTH = 64
COUNTER_SIZE = 2  # bytes per counter (uint16)

# Primes chosen for good distribution
PRIMES = [
    1035493289,
    775374203,
    1495862053,
    2015145641,
    2285881789,
    2473401169,
    2600822929,
    2807566739,
]

# Row salts for mixing
ROW_SALTS = [
    0x9e3779b9,
    0x85ebca6b,
    0xc2b2ae35,
    0x27d4eb2f,
    0x165667b1,
    0x85ebca6b,
    0xc2b2ae35,
    0x27d4eb2f,
]

# Precompute the 400 x 8 index table using XOR-shift bit mixing
_IDX_TABLE = []
for item in range(400):
    row_indices = []
    for r in range(DEPTH):
        # Combine item and row salt with prime, then use XOR-shift for mixing
        h = (item * PRIMES[r] + ROW_SALTS[r]) & 0xFFFFFFFF
        # XOR-shift to mix bits
        h ^= (h >> 16)
        h &= 0xFFFF
        row_indices.append(h % WIDTH)
    _IDX_TABLE.append(row_indices)


def update(state, item):
    """
    Updates the sketch state with a new item.
    state: bytes (1024 bytes)
    item: int
    returns: bytes (1024 bytes)
    """
    b = bytearray(state)
    item_indices = _IDX_TABLE[item]
    for r in range(DEPTH):
        idx = item_indices[r]
        offset = (r * WIDTH + idx) * COUNTER_SIZE
        val = struct.unpack_from('<H', b, offset)[0]
        val += 1
        if val > 65535:
            val = 65535
        struct.pack_into('<H', b, offset, val)
    return bytes(b)


def query(state, item):
    """
    Queries the sketch state for the frequency estimate of an item.
    state: bytes (1024 bytes)
    item: int
    returns: int (non-negative frequency estimate)
    """
    item_indices = _IDX_TABLE[item]
    min_val = 65535
    for r in range(DEPTH):
        idx = item_indices[r]
        offset = (r * WIDTH + idx) * COUNTER_SIZE
        val = struct.unpack_from('<H', state, offset)[0]
        if val < min_val:
            min_val = val
    return min_val
