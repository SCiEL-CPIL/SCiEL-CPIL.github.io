import struct

ROWS = 4
WIDTH = 128
# 4 primes
PRIMES = (
    1035493289, 775374203, 1495862053, 2015145641
)
MOD = 2147483647

def _row_col(item, row):
    p = PRIMES[row]
    return (item * p + 31 * row) % MOD % WIDTH

def update(state, item):
    buf = bytearray(state)
    for row in range(ROWS):
        col = _row_col(item, row)
        idx = (row * WIDTH + col) * 2
        val = struct.unpack_from('<H', buf, idx)[0]
        val += 1
        struct.pack_into('<H', buf, idx, val)
    return bytes(buf)

def query(state, item):
    best = 65535
    for row in range(ROWS):
        col = _row_col(item, row)
        idx = (row * WIDTH + col) * 2
        val = struct.unpack_from('<H', state, idx)[0]
        if val < best:
            best = val
    return best
