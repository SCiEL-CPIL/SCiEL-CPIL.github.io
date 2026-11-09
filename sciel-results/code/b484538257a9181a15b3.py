import struct

# Hybrid: exact uint8 counters for items 0-127, 4-row CM sketch (224 slots, uint8) for items 128-399
# State layout (1024 bytes):
#   bytes 0..127:   exact counts for items 0..127
#   bytes 128..1023: 4 rows of 224 uint8 counters for items 128..399

W = 224
D = 4
PRIME = 2147483647
_PS = (1035493289, 775374203, 1495862053, 2015145641)

# Precompute sketch indices for items 128-399
# _INDICES[item][row] = index within the 224-slot row
_INDICES = [[0] * D for _ in range(400)]
for item in range(128, 400):
    for r in range(D):
        p = _PS[r]
        _INDICES[item][r] = ((item * p + 31 * r) % PRIME) % W

def update(state, item):
    buf = bytearray(state)
    if item < 128:
        # Exact counter
        cur = buf[item]
        if cur < 255:
            buf[item] = cur + 1
    else:
        # CM sketch increment
        idxs = _INDICES[item]
        base = 128
        for r in range(D):
            offset = base + r * W + idxs[r]
            cur = buf[offset]
            if cur < 255:
                buf[offset] = cur + 1
    return bytes(buf)

def query(state, item):
    if item < 128:
        return state[item]
    else:
        idxs = _INDICES[item]
        base = 128
        min_val = 255
        for r in range(D):
            val = state[base + r * W + idxs[r]]
            if val < min_val:
                min_val = val
        return min_val
