import struct

# State layout:
# Bytes 0-19: exact uint8 counters for items 0-19
# Bytes 20-269: exact uint8 counters for items 150-399 (250 items)
# Bytes 270-973: 11 rows x 64 cols uint8 CM sketch for items 20-149 (130 items)
# Bytes 974-1023: padding (50 bytes)
# Total: 20 + 250 + 704 + 50 = 1024 bytes

_EXACT_ITEMS_LOW = list(range(0, 20))       # 20 items
_EXACT_ITEMS_HIGH = list(range(150, 400))   # 250 items
_SKETCH_ITEMS = list(range(20, 150))        # 130 items

_EXACT_LOW_OFFSET = 0
_EXACT_HIGH_OFFSET = 20
_SKETCH_OFFSET = 270
_NUM_ROWS = 11
_NUM_COLS = 64

# Mapping from item -> exact offset or None
_EXACT_MAP = {}
for i in _EXACT_ITEMS_LOW:
    _EXACT_MAP[i] = _EXACT_LOW_OFFSET + i
for i in _EXACT_ITEMS_HIGH:
    _EXACT_MAP[i] = _EXACT_HIGH_OFFSET + (i - 150)

# Hash parameters for CM sketch (for items 20-149)
_PRIMES = [
    1035493289, 775374203, 1495862053, 2015145641,
    1234567891, 987654323, 314159265, 271828183,
    410332281, 524287001, 628318531
]
_SEEDS = [
    31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73
]
_MOD = 2147483647

def update(state, item):
    buf = bytearray(state)
    
    if item in _EXACT_MAP:
        offset = _EXACT_MAP[item]
        val = buf[offset]
        if val < 0xFF:
            val += 1
        buf[offset] = val
    else:
        # CM sketch for items 20-149
        for row in range(_NUM_ROWS):
            p = _PRIMES[row]
            s = _SEEDS[row]
            col = ((item * p + s) % _MOD) % _NUM_COLS
            offset = _SKETCH_OFFSET + row * _NUM_COLS + col
            val = buf[offset]
            if val < 0xFF:
                val += 1
            buf[offset] = val
    
    return bytes(buf)

def query(state, item):
    if item in _EXACT_MAP:
        return state[_EXACT_MAP[item]]
    else:
        min_val = 0xFF
        for row in range(_NUM_ROWS):
            p = _PRIMES[row]
            s = _SEEDS[row]
            col = ((item * p + s) % _MOD) % _NUM_COLS
            offset = _SKETCH_OFFSET + row * _NUM_COLS + col
            val = state[offset]
            if val < min_val:
                min_val = val
        return min_val
