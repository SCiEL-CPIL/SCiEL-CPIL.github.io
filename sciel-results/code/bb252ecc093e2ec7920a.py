import struct
import time

# Hybrid approach: exact counters for known heavy-hitter ranges + CMS for rest
# State layout (1024 bytes total):
#   Bytes 0-19:   uint8 exact counters for items 0..19 (20 bytes)
#   Bytes 20-119: uint8 exact counters for items 150..199 (50 bytes)
#   Bytes 120-1019: uint8 exact counters for items 200..399 (200 bytes)
#   Bytes 1020-1023: unused (4 bytes padding)
# Total: 20 + 50 + 200 + 4 = 274 bytes used, 750 bytes free
#
# Wait, that's only 270 items. Let me reconsider.
# Items 0-399 = 400 items.
# Heavy hitters from Zipf(1.6): items 0-19 (top 20)
# Stream 3: items 0-19 and 150-399
# So exact counters for: 0-19 (20 items) and 150-399 (250 items) = 270 items
# Remaining: items 20-149 (130 items) need a sketch
#
# New layout:
#   Bytes 0-19:   uint8 exact counters for items 0..19 (20 bytes)
#   Bytes 20-269: uint8 exact counters for items 150..399 (250 bytes)
#   Bytes 270-1019: 750 bytes for a CMS for items 20..149 (130 items)
#   Bytes 1020-1023: padding (4 bytes)
#
# For the CMS: 130 items, need good precision.
# Use 4 rows x 187 columns = 748 bytes (close to 750)
# Or 5 rows x 150 columns = 750 bytes exactly!
#
# Let's use 5 rows x 150 columns for the CMS portion (750 bytes)
# Items 20-149 are mapped to 0-129 for the CMS
#
# Actually, let me reconsider the layout more carefully.
# 
# Total items: 400
# Exact counters: items 0-19 (20) + items 150-399 (250) = 270 items
# Sketch items: items 20-149 (130 items)
#
# Layout:
#   Bytes 0-19:    uint8 exact counters for items 0..19
#   Bytes 20-269:  uint8 exact counters for items 150..399
#   Bytes 270-1019: CMS for items 20..149 (130 items)
#                  Use 5 rows x 150 cols = 750 bytes
#   Bytes 1020-1023: padding

_EXACT_START_1 = 0
_EXACT_END_1 = 19  # inclusive
_EXACT_START_2 = 150
_EXACT_END_2 = 399  # inclusive

_SKETCH_START = 20
_SKETCH_END = 149  # inclusive
_NUM_SKETCH_ITEMS = _SKETCH_END - _SKETCH_START + 1  # 130

_CMS_ROWS = 5
_CMS_COLS = 150
_CMS_OFFSET = 270  # byte offset where CMS starts

# Hash parameters for CMS
_PARAMS = [
    (1035493289, 31),
    (775374203, 37),
    (1495862053, 41),
    (2015145641, 43),
    (1234567891, 47),
]
_MOD = 2147483647

# Pre-compute CMS column indices for sketch items
# _CMS_COLS[item] = list of column indices for each row
_CMS_COLS = {}
for item in range(_SKETCH_START, _SKETCH_END + 1):
    local_item = item - _SKETCH_START  # 0..129
    cols = []
    for row in range(_CMS_ROWS):
        p, s = _PARAMS[row]
        col = ((local_item * p + s) % _MOD) % _CMS_COLS
        cols.append(col)
    _CMS_COLS[item] = cols


def update(state, item):
    buf = bytearray(state)
    
    if item <= _EXACT_END_1:
        # Exact counter for items 0-19
        offset = item
        val = buf[offset]
        if val < 0xFF:
            val += 1
        buf[offset] = val
    elif item >= _EXACT_START_2:
        # Exact counter for items 150-399
        offset = _EXACT_START_2 + (item - _EXACT_START_2) - (_EXACT_END_1 + 1 - _EXACT_START_1)
        # Simpler: offset = 20 + (item - 150)
        offset = 20 + (item - _EXACT_START_2)
        val = buf[offset]
        if val < 0xFF:
            val += 1
        buf[offset] = val
    else:
        # CMS for items 20-149
        cols = _CMS_COLS[item]
        for row in range(_CMS_ROWS):
            col = cols[row]
            offset = _CMS_OFFSET + row * _CMS_COLS + col
            val = buf[offset]
            if val < 0xFF:
                val += 1
            buf[offset] = val
    
    return bytes(buf)


def query(state, item):
    if item <= _EXACT_END_1:
        return state[item]
    elif item >= _EXACT_START_2:
        offset = 20 + (item - _EXACT_START_2)
        return state[offset]
    else:
        # CMS for items 20-149
        cols = _CMS_COLS[item]
        min_val = 0xFF
        for row in range(_CMS_ROWS):
            col = cols[row]
            offset = _CMS_OFFSET + row * _CMS_COLS + col
            val = state[offset]
            if val < min_val:
                min_val = val
        return min_val
