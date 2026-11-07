<<<<<<< SEARCH
import struct

_NUM_EXACT_LOW = 20       # items 0-19
_NUM_EXACT_HIGH = 250     # items 150-399
_EXACT_LOW_OFFSET = 0
_EXACT_HIGH_OFFSET = 20   # bytes 20-269

_CMS_ROWS = 3
_CMS_COLS = 250
_CMS_OFFSET = 270

# Items 20-149 map to CMS
_CMS_ITEM_MIN = 20
_CMS_ITEM_MAX = 149

# Primes and seeds for 3 CMS rows
_PRIMES = [1035493289, 775374203, 1495862053]
_SEEDS = [31, 37, 41]
_MOD = 2147483647


def _cms_hash(item, row):
    """Hash item (20-149) to a column in the CMS."""
    p = _PRIMES[row]
    s = _SEEDS[row]
    return ((item * p + s) % _MOD) % _CMS_COLS


def update(state, item):
    """
    state: bytes of length 1024
    item: int in 0..399
    returns: new state as bytes
    """
    buf = bytearray(state)
    
    if item < _NUM_EXACT_LOW:
        # Exact counter for items 0-19
        idx = _EXACT_LOW_OFFSET + item
        val = buf[idx]
        if val < 255:
            buf[idx] = val + 1
    elif item >= 150:
        # Exact counter for items 150-399
        idx = _EXACT_HIGH_OFFSET + (item - 150)
        val = buf[idx]
        if val < 255:
            buf[idx] = val + 1
    else:
        # CMS for items 20-149
        for row in range(_CMS_ROWS):
            col = _cms_hash(item, row)
            offset = _CMS_OFFSET + row * _CMS_COLS + col
            val = buf[offset]
            if val < 255:
                buf[offset] = val + 1
    
    return bytes(buf)


def query(state, item):
    """
    state: bytes of length 1024
    item: int in 0..399
    returns: nonnegative frequency estimate
    """
    if item < _NUM_EXACT_LOW:
        return state[_EXACT_LOW_OFFSET + item]
    elif item >= 150:
        return state[_EXACT_HIGH_OFFSET + (item - 150)]
    else:
        # CMS: take min of 3 rows
        best = 255
        for row in range(_CMS_ROWS):
            col = _cms_hash(item, row)
            offset = _CMS_OFFSET + row * _CMS_COLS + col
            val = state[offset]
            if val < best:
                best = val
        return best
=======
import struct

# State layout: 400 items × 2 bytes (uint16 LE) = 800 bytes, rest zero.
# Each item gets an exact 16-bit counter (max 65535 >> 1024 max count).

def update(state, item):
    """
    state: bytes of length 1024
    item: int in 0..399
    returns: new state as bytes
    """
    buf = bytearray(state)
    offset = item * 2
    lo = buf[offset]
    hi = buf[offset + 1]
    val = lo | (hi << 8)
    val += 1
    buf[offset] = val & 0xFF
    buf[offset + 1] = (val >> 8) & 0xFF
    return bytes(buf)


def query(state, item):
    """
    state: bytes of length 1024
    item: int in 0..399
    returns: nonnegative frequency estimate
    """
    offset = item * 2
    return state[offset] | (state[offset + 1] << 8)
>>>>>>> REPLACE
