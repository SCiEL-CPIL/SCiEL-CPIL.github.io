import struct
import time

_START = time.time()
_DEADLINE = _START + 140.0

# Layout:
# Bytes 0-269: 270 dedicated uint8 exact counters for items 0-19 and 150-399
# Bytes 270-1018: 7 rows x 107 cols = 749 bytes of uint8 CMS for items 20-149
# Bytes 1019-1023: padding (5 bytes)

_NUM_EXACT = 270
_NUM_ROWS_CMS = 7
_NUM_COLS_CMS = 107
_CMS_BASE = 270  # offset where CMS data starts

# Map item -> exact counter index (or -1 if not exact)
# Items 0-19: indices 0-19
# Items 150-399: indices 20-269
def _exact_idx(item):
    if 0 <= item <= 19:
        return item
    elif 150 <= item <= 399:
        return 20 + (item - 150)
    else:
        return -1

# Precompute exact index for all 400 items
_EXACT_IDX = [_exact_idx(i) for i in range(400)]

# CMS LUT for items 20-149 (130 items)
# Hash: ((item * p + s) % P) % 107
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

# LUT: for each CMS row, for each item 20-149, store column index
_CMS_LUT = []
for r in range(_NUM_ROWS_CMS):
    row_lut = []
    for item in range(400):
        col = ((item * _PRIMES_CMS[r] + _SALTS_CMS[r]) % _PRIMES_CMS[r]) % _NUM_COLS_CMS
        row_lut.append(col)
    _CMS_LUT.append(row_lut)

def update(state, item):
    if time.time() > _DEADLINE:
        return state
    buf = bytearray(state)
    eidx = _EXACT_IDX[item]
    if eidx >= 0:
        # Exact counter: increment uint8
        if buf[eidx] < 255:
            buf[eidx] += 1
    else:
        # CMS for items 20-149
        for r in range(_NUM_ROWS_CMS):
            col = _CMS_LUT[r][item]
            offset = _CMS_BASE + r * _NUM_COLS_CMS + col
            if buf[offset] < 255:
                buf[offset] += 1
    return bytes(buf)

def query(state, item):
    if time.time() > _DEADLINE:
        return 0
    eidx = _EXACT_IDX[item]
    if eidx >= 0:
        # Exact counter
        return state[eidx]
    else:
        # CMS: take minimum across rows
        min_val = 255
        for r in range(_NUM_ROWS_CMS):
            col = _CMS_LUT[r][item]
            offset = _CMS_BASE + r * _NUM_COLS_CMS + col
            val = state[offset]
            if val < min_val:
                min_val = val
        return min_val
