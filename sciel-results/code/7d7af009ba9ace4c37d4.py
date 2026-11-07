import struct

ROWS = 4
WIDTH = 90
EXACT_ITEMS = 150
EXACT_BYTES = EXACT_ITEMS * 2
CMS_BYTES = ROWS * WIDTH * 2
TOTAL_BYTES = EXACT_BYTES + CMS_BYTES
# 150*2 + 4*90*2 = 300 + 720 = 1020
# We have 1024 bytes, so 4 bytes padding.

PRIMES = [1035493289, 775374203, 1495862053, 2015145641]
MOD = 2147483647

def _hash_index(item, row):
    p = PRIMES[row]
    return ((item * p + 31 * row) % MOD) % WIDTH

def update(state, item):
    buf = bytearray(state)
    if item < EXACT_ITEMS:
        offset = item * 2
        val = buf[offset] | (buf[offset + 1] << 8)
        val += 1
        if val > 65535:
            val = 65535
        buf[offset] = val & 0xFF
        buf[offset + 1] = (val >> 8) & 0xFF
    else:
        for row in range(ROWS):
            col = _hash_index(item, row)
            offset = EXACT_BYTES + row * WIDTH * 2 + col * 2
            val = buf[offset] | (buf[offset + 1] << 8)
            val += 1
            if val > 65535:
                val = 65535
            buf[offset] = val & 0xFF
            buf[offset + 1] = (val >> 8) & 0xFF
    return bytes(buf)

def query(state, item):
    buf = state
    if item < EXACT_ITEMS:
        offset = item * 2
        val = buf[offset] | (buf[offset + 1] << 8)
        return float(val)
    else:
        min_val = 65536
        for row in range(ROWS):
            col = _hash_index(item, row)
            offset = EXACT_BYTES + row * WIDTH * 2 + col * 2
            val = buf[offset] | (buf[offset + 1] << 8)
            if val < min_val:
                min_val = val
        return float(min_val)
