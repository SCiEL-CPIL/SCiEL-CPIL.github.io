import struct

ROWS = 16
WIDTH = 32
# 16 odd primes for hashing
# Using large primes to ensure good mixing
PRIMES = (
    1035493289, 775374203, 1495862053, 2015145641,
    1103515245, 1234567891, 1357924681, 1478967303,
    1597463007, 1680700013, 1794246733, 1899884863,
    1987654321, 2048362973, 2147483629, 2236962103
)
# Offset constant for row decorrelation
ROW_OFFSET = 31

def _row_col(item, row):
    p = PRIMES[row]
    # Fast hash: (item * p + row * offset) % WIDTH
    # Since WIDTH is 32 (2^5), we can use bitwise AND if we want, 
    # but modulo is fine and handles any WIDTH.
    return (item * p + ROW_OFFSET * row) % WIDTH

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
