import struct

# Count-Min Sketch with 2 rows of 256 uint16 counters
W = 256
D = 2
PRIME = 2147483647
_PS = (1035493289, 775374203)

# Precompute indices for each item for each row
# _INDICES[item][row] = index
_INDICES = [[0] * D for _ in range(400)]
for item in range(400):
    for r in range(D):
        p = _PS[r]
        _INDICES[item][r] = ((item * p + 31 * r) % PRIME) % W

def update(state, item):
    buf = bytearray(state)
    idxs = _INDICES[item]
    for r in range(D):
        offset = r * (W * 2) + idxs[r] * 2
        cur = int.from_bytes(buf[offset:offset+2], 'little')
        if cur < 65535:
            buf[offset:offset+2] = (cur + 1).to_bytes(2, 'little')
    return bytes(buf)

def query(state, item):
    idxs = _INDICES[item]
    min_val = 65535
    for r in range(D):
        offset = r * (W * 2) + idxs[r] * 2
        val = int.from_bytes(state[offset:offset+2], 'little')
        if val < min_val:
            min_val = val
    return min_val
