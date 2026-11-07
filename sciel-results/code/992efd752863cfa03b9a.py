import struct

# 4 rows, 128 columns, uint16
ROWS = 4
COLS = 128
# Primes for hashing
PRIMES = [1035493289, 775374203, 1495862053, 2015145641]
LARGE_PRIME = 2147483647

def update(state, item):
    # state is 1024 bytes: 4 rows * 128 cols * 2 bytes (uint16)
    counters = list(struct.unpack('<' + 'H' * (ROWS * COLS), state))
    for row in range(ROWS):
        idx = ((item * PRIMES[row] + 31 * row) % LARGE_PRIME) % COLS
        pos = row * COLS + idx
        counters[pos] = min(counters[pos] + 1, 65535)
    return struct.pack('<' + 'H' * (ROWS * COLS), *counters)

def query(state, item):
    counters = struct.unpack('<' + 'H' * (ROWS * COLS), state)
    estimates = []
    for row in range(ROWS):
        idx = ((item * PRIMES[row] + 31 * row) % LARGE_PRIME) % COLS
        pos = row * COLS + idx
        estimates.append(counters[pos])
    return min(estimates)
