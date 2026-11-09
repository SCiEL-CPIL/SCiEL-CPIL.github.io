# 2 rows x 512 columns of uint8 counters = 1024 bytes
W = 512
D = 2
PRIMES = (1035493289, 775374203)
PRIME = 2147483647

# Precompute all indices: _IDX[item] = (idx0, idx1)
_IDX = []
for item in range(400):
    row_indices = []
    for r in range(D):
        p = PRIMES[r]
        idx = ((item * p + 31 * r) % PRIME) % W
        row_indices.append(idx)
    _IDX.append(tuple(row_indices))

def update(state, item):
    buf = bytearray(state)
    i0, i1 = _IDX[item]
    
    v = buf[i0]
    buf[i0] = v + 1 if v < 255 else 255
    
    v = buf[512 + i1]
    buf[512 + i1] = v + 1 if v < 255 else 255
    
    return bytes(buf)

def query(state, item):
    i0, i1 = _IDX[item]
    
    v0 = state[i0]
    v1 = state[512 + i1]
    
    m = v0 if v0 < v1 else v1
    
    return m
