import struct

# Configuration: 4 rows x 128 columns, uint16 little-endian counters
# Total size: 4 * 128 * 2 = 1024 bytes
_ROWS = 4
_WIDTH = 128
_PRIME = 2147483647  # 2^31 - 1
# Hash multipliers (large primes)
_MULTIPLIERS = (1035493289, 775374203, 1495862053, 2015145641)
# Salt offsets per row to decorrelate
_SALTS = (31, 97, 193, 311)

# Precompute indices for all 400 items to speed up update/query
# _IDX[item] = (idx_row0, idx_row1, idx_row2, idx_row3)
_IDX = []
for item in range(400):
    row_idxs = []
    for r in range(_ROWS):
        p = _MULTIPLIERS[r]
        s = _SALTS[r]
        # Use large prime modulus to avoid modulo bias for small item IDs
        idx = ((item * p + s) % _PRIME) % _WIDTH
        row_idxs.append(idx)
    _IDX.append(tuple(row_idxs))

def update(state, item):
    buf = bytearray(state)
    idxs = _IDX[item]
    for r in range(_ROWS):
        idx = idxs[r]
        # Byte offset for this counter: row * (width * 2) + idx * 2
        off = r * (_WIDTH * 2) + idx * 2
        # Read current value
        cur = buf[off] | (buf[off + 1] << 8)
        # Increment, cap at 65535 (though 1024 max count makes this safe)
        if cur < 65535:
            new_val = cur + 1
            buf[off] = new_val & 0xFF
            buf[off + 1] = (new_val >> 8) & 0xFF
    return bytes(buf)

def query(state, item):
    idxs = _IDX[item]
    min_val = 65535
    for r in range(_ROWS):
        idx = idxs[r]
        off = r * (_WIDTH * 2) + idx * 2
        val = state[off] | (state[off + 1] << 8)
        if val < min_val:
            min_val = val
    return min_val
