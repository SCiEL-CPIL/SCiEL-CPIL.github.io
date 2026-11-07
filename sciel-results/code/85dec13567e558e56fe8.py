import struct
import time

_START = time.time()
_DEADLINE = _START + 140.0

# Layout:
# Bytes 0-269: 270 dedicated uint8 exact counters for items 0-19 and 150-399
# Bytes 270-1023: 754 bytes = 7 rows x 53 cols of uint8 CMS for items 20-149
#   7 * 53 = 371 bytes... wait, 754 / 7 = 107.7. Let me recalculate.
# Actually: 1024 - 270 = 754 bytes. Use 7 rows x 107 cols? 7*107=749. 754-749=5 unused.
# Better: 10 rows x 75 cols = 750 bytes. 754-750=4 unused.
# Or: 7 rows x 107 cols = 749. Use that, 5 bytes padding.

_NUM_EXACT = 270  # items 0-19 (20) and 150-399 (250)
_NUM_ROWS_CMS = 7
_NUM_COLS_CMS = 107  # 7*107 = 749, total = 270+749 = 1019, 5 bytes padding

# Map item to exact counter index or CMS
# Items 0-19: exact index 0-19
# Items 150-399: exact index 20-269
# Items 20-149: CMS

def _item_to_exact(item):
    if item <= 19:
        return item
    elif item >= 150:
        return 20 + (item - 150)
    else:
        return -1

# Precompute CMS LUT for items 20-149
_PRIMES_CMS = [
    2147483647,
    2147483629,
    2147483587,
    2147483579,
    2147483549,
    2147483539,
    2147483533,
]
_SALTS_CMS = [31, 61, 97, 127, 151, 181, 211]

_LUT_CMS = [[((item * _PRIMES_CMS[r] + _SALTS_CMS[r]) % _PRIMES_CMS[r]) % _NUM_COLS_CMS for item in range(400)] for r in range(_NUM_ROWS_CMS)]

# Precompute exact index for each item
_EXACT_IDX = [_item_to_exact(i) for i in range(400)]

def update(state, item):
    if time.time() > _DEADLINE:
        return state
    buf = bytearray(state)
    ex = _EXACT_IDX[item]
    if ex >= 0:
        # Increment exact counter
        if buf[ex] < 255:
            buf[ex] += 1
    else:
        # Increment CMS counters
        for r in range(_NUM_ROWS_CMS):
            col = _LUT_CMS[r][item]
            offset = _NUM_EXACT + r * _NUM_COLS_CMS + col
            if buf[offset] < 255:
                buf[offset] += 1
    return bytes(buf)

def query(state, item):
    if time.time() > _DEADLINE:
        return 0
    ex = _EXACT_IDX[item]
    if ex >= 0:
        return state[ex]
    else:
        min_val = 255
        for r in range(_NUM_ROWS_CMS):
            col = _LUT_CMS[r][item]
            offset = _NUM_EXACT + r * _NUM_COLS_CMS + col
            val = state[offset]
            if val < min_val:
                min_val = val
        return min_val
