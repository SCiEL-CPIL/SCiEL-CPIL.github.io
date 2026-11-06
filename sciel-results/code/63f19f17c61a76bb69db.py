import numpy as np

def construct():
    limit = 1000000
    phi = np.arange(limit + 1, dtype=np.int64)
    
    for i in range(2, limit + 1):
        if phi[i] == i:
            phi[i::i] -= phi[i::i] // i
            
    mask = phi % 120 == 0
    mask[0] = False
    indices = np.nonzero(mask)[0]
    if len(indices) > 0:
        return int(indices[0])
    return 1
