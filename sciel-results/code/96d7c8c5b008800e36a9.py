"""Matrix multiplication tensor decomposition for 4x5 * 5x5.
Block-structured decomposition splitting the middle dimension.
"""

DIMENSIONS = [4, 5, 5]

def construct():
    n, m, p = 4, 5, 5
    all_U = []
    all_V = []
    all_W = []
    
    # Block 1: j=0..3, k=0..3 (4x4 * 4x4)
    for i in range(4):
        for j in range(4):
            for k in range(4):
                u = [0] * 20
                v = [0] * 25
                w = [0] * 20
                u[i * 5 + j] = 1
                v[j * 5 + k] = 1
                w[k * 4 + i] = 1
                all_U.append(u)
                all_V.append(v)
                all_W.append(w)
    
    # Block 2: j=0..3, k=4 (4x4 * 4x1)
    for i in range(4):
        for j in range(4):
            k = 4
            u = [0] * 20
            v = [0] * 25
            w = [0] * 20
            u[i * 5 + j] = 1
            v[j * 5 + k] = 1
            w[k * 4 + i] = 1
            all_U.append(u)
            all_V.append(v)
            all_W.append(w)
    
    # Block 3: j=4, k=0..4 (4x1 * 1x5)
    for i in range(4):
        j = 4
        for k in range(5):
            u = [0] * 20
            v = [0] * 25
            w = [0] * 20
            u[i * 5 + j] = 1
            v[j * 5 + k] = 1
            w[k * 4 + i] = 1
            all_U.append(u)
            all_V.append(v)
            all_W.append(w)
    
    return {"U": all_U, "V": all_V, "W": all_W}
