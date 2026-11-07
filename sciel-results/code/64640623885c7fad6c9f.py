# EVOLVE-BLOCK-START

_NUM_ROWS = 16
_NUM_COLS = 32

_PRIMES = [
    1035493289,
    775374203,
    1495862053,
    2015145641,
    1234567891,
    1919191911,
    16807,
    48271,
    69621,
    28174,
    99991,
    104729,
    109331,
    114257,
    120001,
    125797,
]
_MOD = 2147483647


def update(state, item):
    buf = bytearray(state)
    for row in range(_NUM_ROWS):
        p = _PRIMES[row]
        idx = ((item * p + 31 * row) % _MOD) % _NUM_COLS
        offset = (row * _NUM_COLS + idx) * 2
        lo = buf[offset]
        hi = buf[offset + 1]
        val = lo | (hi << 8)
        if val < 15:
            val += 1
        buf[offset] = val & 0xFF
        buf[offset + 1] = (val >> 8) & 0xFF
    return bytes(buf)


def query(state, item):
    best = 15
    for row in range(_NUM_ROWS):
        p = _PRIMES[row]
        idx = ((item * p + 31 * row) % _MOD) % _NUM_COLS
        offset = (row * _NUM_COLS + idx) * 2
        lo = state[offset]
        hi = state[offset + 1]
        val = lo | (hi << 8)
        if val < best:
            best = val
    return (1 << best) - 1

# EVOLVE-BLOCK-END
