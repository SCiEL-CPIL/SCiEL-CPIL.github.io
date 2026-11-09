import struct

# Full-width CM sketch: 8 rows x 128 cols, uint16 LE
# Total: 8 * 128 * 2 = 2048... wait, that's over 1024.
# 1024 / 2 = 512 counters total.
# Options: 8 rows x 64 cols, or 4 rows x 128 cols, or 2 rows x 256 cols.
# More rows = better min accuracy. 8 rows x 64 cols = 512 counters = 1024 bytes.
# With 64 bins and 336+ items, collision is high. 
# Let's do 4 rows x 128 cols = 512 counters = 1024 bytes.
# Or 6 rows x 85 cols = 510 counters = 1020 bytes (4 wasted).
# 4 rows x 128 cols is clean. 128 bins for 400 items -> load factor 3.125 per row.
# With 4 rows, min of 4 should be quite tight.

_DEPTH = 4
_WIDTH = 128
_BYTES_PER_COUNTER = 2
_TOTAL_COUNTERS = _DEPTH * _WIDTH  # 512
_STATE_SIZE = _TOTAL_COUNTERS * _BYTES_PER_COUNTER  # 1024

# Hash multipliers - use large odd primes for good distribution
# We need hash(item, row) -> index in [0, _WIDTH)
# Use: (a * item + b) % _WIDTH with different a,b per row
# To avoid correlation, use different multipliers

_A = [1035493289, 775374203, 1495862053, 2015145641]
_B = [31, 97, 193, 311]  # Different offsets per row

def _read_u16(buf, off):
    return buf[off] | (buf[off + 1] << 8)

def _write_u16(buf, off, val):
    buf[off] = val & 0xFF
    buf[off + 1] = (val >> 8) & 0xFF

def _hash(item, row):
    return (_A[row] * item + _B[row]) % _WIDTH

def update(state, item):
    buf = bytearray(state)
    for r in range(_DEPTH):
        idx = _hash(item, r)
        off = (r * _WIDTH + idx) * 2
        v = _read_u16(buf, off)
        if v < 65535:
            _write_u16(buf, off, v + 1)
    return bytes(buf)

def query(state, item):
    mn = 65535
    for r in range(_DEPTH):
        idx = _hash(item, r)
        off = (r * _WIDTH + idx) * 2
        v = _read_u16(buf := state, off)  # state is bytes, indexing works
        if v < mn:
            mn = v
    return mn
