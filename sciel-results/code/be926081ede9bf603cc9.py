import struct

# EVOLVE-BLOCK-START
# Configuration for an improved Count-Min Sketch
# Using 8 rows of 64 16-bit counters (512 counters total)
# This fits exactly in 1024 bytes (512 * 2 = 1024)
# Increasing depth from 4 to 8 reduces overestimation due to collisions.

PRIMES = [
    1035493289,
    775374203,
    1495862053,
    2015145641,
    2285881789,
    2473401169,
    2600822929,
    2807566739
]
MOD = 2147483647
WIDTH = 64
DEPTH = 8
COUNTER_SIZE = 2  # bytes per counter (uint16)

def update(state, item):
    b = bytearray(state)
    for i in range(DEPTH):
        p = PRIMES[i]
        # Compute index for this row
        # Using the same hashing scheme as the seed but with more rows
        idx = ((item * p + 31 * i) % MOD) % WIDTH
        offset = i * WIDTH * COUNTER_SIZE + idx * COUNTER_SIZE
        val = struct.unpack_from('<H', b, offset)[0]
        val += 1
        struct.pack_into('<H', b, offset, val)
    return bytes(b)

def query(state, item):
    min_val = 65535
    for i in range(DEPTH):
        p = PRIMES[i]
        idx = ((item * p + 31 * i) % MOD) % WIDTH
        offset = i * WIDTH * COUNTER_SIZE + idx * COUNTER_SIZE
        val = struct.unpack_from('<H', state, offset)[0]
        if val < min_val:
            min_val = val
    return min_val
# EVOLVE-BLOCK-END
