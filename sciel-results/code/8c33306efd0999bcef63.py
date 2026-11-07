import struct

# Layout: bytes 0..399 = exact 8-bit counters for items 0..399
# bytes 400..1023 = 624 bytes for a small CMS (3 rows x 208 cols, 8-bit)
# Actually let's use: 400 bytes exact + 624 bytes CMS
# CMS: 3 rows, 208 cols = 624 bytes
CMS_ROWS = 3
CMS_COLS = 208
CMS_OFFSET = 400

CMS_PRIMES = (
    1035493289, 775374203, 1495862053
)
CMS_MOD = 2147483647

def _cms_idx(item, row):
    p = CMS_PRIMES[row]
    return (item * p + 31 * row) % CMS_MOD % CMS_COLS

def update(state, item):
    buf = bytearray(state)
    # Exact counter
    if buf[item] < 255:
        buf[item] += 1
    # CMS
    for row in range(CMS_ROWS):
        col = _cms_idx(item, row)
        idx = CMS_OFFSET + row * CMS_COLS + col
        if buf[idx] < 255:
            buf[idx] += 1
    return bytes(buf)

def query(state, item):
    exact = state[item]
    # CMS minimum
    best = 255
    for row in range(CMS_ROWS):
        col = _cms_idx(item, row)
        idx = CMS_OFFSET + row * CMS_COLS + col
        val = state[idx]
        if val < best:
            best = val
    # Return the larger of exact and CMS min to handle overflow
    return max(exact, best)
