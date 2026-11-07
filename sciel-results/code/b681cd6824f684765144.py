import struct

# State layout:
# Bytes 0-19:   20 uint8 counters for items 0-19
# Bytes 20-269: 250 uint8 counters for items 150-399
# Bytes 270-1023: 754 bytes for CM sketch (for items 20-149)
#   CM: 10 rows x 75 cols = 750 bytes (4 bytes padding)

COLD_START = 20
COLD_END = 150  # items 20..149 are "cold"
HOT1_END = 20   # items 0..19
HOT2_START = 150  # items 150..399

NUM_COLD = COLD_END - COLD_START  # 130 items
NUM_ROWS = 10
NUM_COLS = 75

# Primes for CM sketch (odd numbers)
PRIMES = [
    1035493289,
    775374203,
    1495862053,
    2015145641,
    2654435761,
    3004317997,
    3221225473,
    3405833819,
    3607924513,
    3814234567,
]

MOD = 2147483647

def update(state, item):
    counters = bytearray(state)
    
    if item < HOT1_END:
        # Hot range 1: items 0-19, exact counter
        counters[item] += 1
    elif item >= HOT2_START:
        # Hot range 2: items 150-399, exact counter
        counters[HOT1_END + (item - HOT2_START)] += 1
    else:
        # Cold range: items 20-149, use CM sketch
        offset = HOT1_END + (HOT2_START - HOT1_END)  # 270
        for r in range(NUM_ROWS):
            p = PRIMES[r]
            idx = ((item * p + 31 * r) % MOD) % NUM_COLS
            pos = offset + r * NUM_COLS + idx
            counters[pos] += 1
            if counters[pos] > 255:
                counters[pos] = 255
    
    return bytes(counters)

def query(state, item):
    counters = state
    
    if item < HOT1_END:
        # Hot range 1: exact
        return counters[item]
    elif item >= HOT2_START:
        # Hot range 2: exact
        return counters[HOT1_END + (item - HOT2_START)]
    else:
        # Cold range: CM sketch min
        offset = HOT1_END + (HOT2_START - HOT1_END)  # 270
        min_val = 255
        for r in range(NUM_ROWS):
            p = PRIMES[r]
            idx = ((item * p + 31 * r) % MOD) % NUM_COLS
            pos = offset + r * NUM_COLS + idx
            val = counters[pos]
            if val < min_val:
                min_val = val
        return min_val
