import struct

# Use a better hash with multiple independent hashes for Count-Min style
# 4 rows x 64 counters = 256 uint32 = 1024 bytes
# Each counter is 4 bytes (uint32)

# Hash constants - using different primes for each row
P = (1035493289, 775374203, 1495862053, 2015145641)
M = 2147483647  # 2^31 - 1

def _hash(item, row):
    """Compute hash index for given item and row."""
    return ((item * P[row] + 31 * row) % M) % 64

def update(state, item):
    b = bytearray(state)
    for row in range(4):
        idx = _hash(item, row)
        offset = (row * 64 + idx) * 4
        val = struct.unpack_from('<I', b, offset)[0]
        val = (val + 1) & 0xFFFFFFFF
        struct.pack_into('<I', b, offset, val)
    return bytes(b)

def query(state, item):
    estimates = []
    for row in range(4):
        idx = _hash(item, row)
        offset = (row * 64 + idx) * 4
        val = struct.unpack_from('<I', state, offset)[0]
        estimates.append(val)
    return min(estimates)
