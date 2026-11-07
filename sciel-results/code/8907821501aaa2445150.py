import struct

# EVOLVE-BLOCK-START
# Count-Min Sketch: 8 rows x 64 columns of uint16 counters = 1024 bytes.
# Precomputed index table for items 0..399 to speed up update/query.
# Uses well-distributed hash primes with a clean modulo-64 mapping.

DEPTH = 8
WIDTH = 64
COUNTER_SIZE = 2

# Primes chosen for good distribution when mod 64
PRIMES = [
    2654435761,  # 2^32 * phi, common hash prime
    40503,       # small prime
    69069,       # product of small primes, good mixing
    2862933555777941757 % 2**32,  # large prime mod 2^32
    3037000493,
    2246822519,
    3266489917,
    668265263,
]

# Precompute index table: idx_table[item][row] = column index
# Using: (item * prime + row_salt) & 63  (bitwise AND for power-of-2 width)
# This is fast and well-distributed for power-of-2 widths.
ROW_SALTS = [
    0x9e3779b9,
    0x85ebca6b,
    0xc2b2ae35,
    0x27d4eb2f,
    0x165667b1,
    0x9e3779b9,  # different prime handles distribution
    0x85ebca6b,
    0xc2b2ae35,
]

# Precompute the 400 x 8 index table
_IDX_TABLE = []
for item in range(400):
    row_indices = []
    for r in range(DEPTH):
        # Combine item and row salt, then use high bits for better mixing
        h = (item * PRIMES[r] + ROW_SALTS[r]) & 0xFFFFFFFF
        # XOR-shift to mix bits
        h ^= (h >> 16)
        h &= 0xFFFF
        row_indices.append(h % WIDTH)
    _IDX_TABLE.append(row_indices)

def update(state, item):
    b = bytearray(state)
    item_indices = _IDX_TABLE[item]
    for r in range(DEPTH):
        idx = item_indices[r]
        offset = (r * WIDTH + idx) * COUNTER_SIZE
        val = struct.unpack_from('<H', b, offset)[0]
        val += 1
        if val >= 65536:
            val = 65535
        struct.pack_into('<H', b, offset, val)
    return bytes(b)

def query(state, item):
    item_indices = _IDX_TABLE[item]
    min_val = 65535
    for r in range(DEPTH):
        idx = item_indices[r]
        offset = (r * WIDTH + idx) * COUNTER_SIZE
        val = struct.unpack_from('<H', state, offset)[0]
        if val < min_val:
            min_val = val
    return min_val
# EVOLVE-BLOCK-END
