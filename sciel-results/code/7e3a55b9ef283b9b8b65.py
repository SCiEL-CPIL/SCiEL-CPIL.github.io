# EVOLVE-BLOCK-START

_NUM_DEDICATED = 100
_NUM_FINE = 128
_NUM_COARSE = 32
_NUM_FINE_ROWS = 5
_NUM_COARSE_ROWS = 3

# Layout:
# Bytes 0..99: dedicated 10-bit counters (100 items, 2 bytes each = 200 bytes)
# Bytes 200..819: fine CM sketch (5 rows × 128 cols = 640 bytes)
# Bytes 820..899: coarse CM sketch (3 rows × 32 cols = 96 bytes)
# Bytes 900..1023: padding

_FINE_BASE = _NUM_DEDICATED * 2  # 200
_COARSE_BASE = _FINE_BASE + _NUM_FINE_ROWS * _NUM_FINE  # 200 + 640 = 840

_FINE_PRIMES = [1035493289, 775374203, 1495862053, 2015145641, 1234567891]
_COARSE_PRIMES = [16807, 48271, 69621]

_MOD = 2147483647


def _fine_idx(item, row):
    return ((item * _FINE_PRIMES[row] + 31 * (row + 1)) % _MOD) % _NUM_FINE


def _coarse_idx(item, row):
    return ((item * _COARSE_PRIMES[row] + 17 * (row + 1)) % _MOD) % _NUM_COARSE


def update(state, item):
    buf = bytearray(state)

    if item < _NUM_DEDICATED:
        # 10-bit counter: low byte at 2*item, high byte at 2*item+1
        lo = buf[2 * item]
        hi = buf[2 * item + 1]
        val = lo + (hi << 8)
        if val < 1023:
            val += 1
        buf[2 * item] = val & 0xFF
        buf[2 * item + 1] = (val >> 8) & 0xFF
    else:
        # Fine CM sketch
        for row in range(_NUM_FINE_ROWS):
            idx = _fine_idx(item, row)
            offset = _FINE_BASE + row * _NUM_FINE + idx
            if buf[offset] < 255:
                buf[offset] += 1
        # Coarse CM sketch
        for row in range(_NUM_COARSE_ROWS):
            idx = _coarse_idx(item, row)
            offset = _COARSE_BASE + row * _NUM_COARSE + idx
            if buf[offset] < 255:
                buf[offset] += 1

    return bytes(buf)


def query(state, item):
    if item < _NUM_DEDICATED:
        # Exact 10-bit count
        return state[2 * item] + (state[2 * item + 1] << 8)
    else:
        # Fine sketch: min of 5 rows
        best = 255
        for row in range(_NUM_FINE_ROWS):
            idx = _fine_idx(item, row)
            offset = _FINE_BASE + row * _NUM_FINE + idx
            val = state[offset]
            if val < best:
                best = val
        # Coarse sketch: min of 3 rows
        for row in range(_NUM_COARSE_ROWS):
            idx = _coarse_idx(item, row)
            offset = _COARSE_BASE + row * _NUM_COARSE + idx
            val = state[offset]
            if val < best:
                best = val
        return best

# EVOLVE-BLOCK-END
