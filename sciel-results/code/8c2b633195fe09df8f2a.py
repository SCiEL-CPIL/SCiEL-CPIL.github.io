"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START

# Hybrid: 128 exact uint8 counters + 4-row CMS for items 128..399
# Layout: bytes 0..127 = exact counts for items 0..127
#         bytes 128..1023 = 4 rows x 224 cols of uint8 CMS
#
# Rationale:
# - Zipf(1.6) mod 400: items 0..~50 have counts >10; 128 exact covers all heavy hitters
# - Stream 3: items 0..19 (512 uniform) and 150..399 (512 uniform); 128 exact covers 0..19
# - CMS only updated for items >=128, reducing collision noise
# - 224 columns > 256 seed columns... wait, 224 < 256. But we freed 128 bytes from exact.
#   Actually 4*224=896, total 128+896=1024. Columns=224 < 256 of pure CMS.
#   But the exact portion eliminates ALL error for 128 items, which is a big win.

_EXACT_N = 128
_ROWS = 4
_COLS = 224
_BASE = 128  # byte offset where CMS starts

_P = (1035493289, 775374203, 1495862053, 2015145641)
_MOD = 2147483647

def _col(item, row):
    return ((item * _P[row] + 31 * row) % _MOD) % _COLS

def update(state, item):
    b = bytearray(state)
    if item < _EXACT_N:
        b[item] = (b[item] + 1) & 0xFF
    else:
        # Only update CMS for items >= 128 to avoid collision noise
        for r in range(_ROWS):
            c = _col(item, r)
            off = _BASE + r * _COLS + c
            b[off] = (b[off] + 1) & 0xFF
    return bytes(b)

def query(state, item):
    if item < _EXACT_N:
        return state[item]
    # CMS minimum over 4 rows
    m = 255
    for r in range(_ROWS):
        c = _col(item, r)
        v = state[_BASE + r * _COLS + c]
        if v < m:
            m = v
    return m

# EVOLVE-BLOCK-END
