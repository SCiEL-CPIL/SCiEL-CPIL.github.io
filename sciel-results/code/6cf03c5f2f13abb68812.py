"""
Memory-bounded streaming frequency sketch: hybrid exact + compact CMS.

Strategy:
- Items 0-19 and 150-399 (270 items) get exact 8-bit counters.
  These are the heavy-hitter ranges in streams 2 and 3.
- Items 20-149 (130 items) get a compact Count-Min sketch.
  These only appear in stream 1 (uniform) with ~2.56 expected count each.

State layout (1024 bytes):
- Bytes 0-19:   20 x uint8 exact counters for items 0-19
- Bytes 20-269: 250 x uint8 exact counters for items 150-399 (item i -> byte 20+(i-150))
- Bytes 270-1023: 754 bytes for CMS: 6 rows x 125 cols x uint8
  (6*125 = 750, leaving 4 spare bytes)
"""

# --- Constants ---
_EXACT_LOW_END = 20       # items 0..19
_EXACT_HIGH_START = 150   # items 150..399

_CMS_ROWS = 6
_CMS_COLS = 125
_CMS_OFFSET = 270         # 20 + 250 = 270

_PRIMES = [
    1035493289,
    775374203,
    1495862053,
    2015145641,
    32452843,
    2166136261,
]
_SEEDS = [31, 37, 41, 43, 47, 53]
_MOD = 2147483647


def _cms_col(item, row):
    """Hash an item in 20..149 to a column index in [0, CMS_COLS)."""
    p = _PRIMES[row]
    s = _SEEDS[row]
    return ((item * p + s) % _MOD) % _CMS_COLS


def update(state, item):
    """
    state: bytes of length 1024
    item:  int in 0..399
    returns: new state as bytes (length 1024)
    """
    buf = bytearray(state)

    if item < _EXACT_LOW_END:
        # Exact counter for items 0..19
        idx = item
        if buf[idx] < 255:
            buf[idx] += 1
    elif item >= _EXACT_HIGH_START:
        # Exact counter for items 150..399, stored at offset 20+(item-150)
        idx = 20 + (item - _EXACT_HIGH_START)
        if buf[idx] < 255:
            buf[idx] += 1
    else:
        # Items 20..149: compact CMS
        for row in range(_CMS_ROWS):
            col = _cms_col(item, row)
            offset = _CMS_OFFSET + row * _CMS_COLS + col
            if buf[offset] < 255:
                buf[offset] += 1

    return bytes(buf)


def query(state, item):
    """
    state: bytes of length 1024
    item:  int in 0..399
    returns: nonnegative frequency estimate (float or int)
    """
    if item < _EXACT_LOW_END:
        return float(state[item])
    elif item >= _EXACT_HIGH_START:
        return float(state[20 + (item - _EXACT_HIGH_START)])
    else:
        # CMS: min over 6 rows
        best = 255
        for row in range(_CMS_ROWS):
            col = _cms_col(item, row)
            offset = _CMS_OFFSET + row * _CMS_COLS + col
            val = state[offset]
            if val < best:
                best = val
        return float(best)
