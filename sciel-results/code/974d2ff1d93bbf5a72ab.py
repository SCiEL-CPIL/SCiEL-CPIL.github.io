import numpy as np

def construct():
    n, m, p = 2, 3, 10
    m1 = 2  # first block size
    m2 = 1  # second block size
    
    rank1 = m1 * p  # 20
    rank2 = 1
    rank = rank1 + rank2  # 21
    
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    col = 0
    # Block 1: j in {0,1}
    for j in range(m1):
        for k in range(p):
            for i in range(n):
                U[i*m + j, col] = 1
                W[k*n + i, col] = 1
            V[j*p + k, col] = 1
            col += 1
    
    # Block 2: j=2
    for i in range(n):
        U[i*m + 2, col] = 1
    for k in range(p):
        V[2*p + k, col] = 1
        for i in range(n):
            W[k*n + i, col] = 1
    col += 1
    
    # Verify
    for i in range(n):
        for j in range(m):
            for jp in range(m):
                for k in range(p):
                    s = sum(U[i*m+j, c] * V[jp*p+k, c] * W[k*n+i, c] for c in range(rank))
                    expected = 1 if j == jp else 0
                    if abs(s - expected) > 1e-10:
                        raise ValueError(f"Failed at i={i},j={j},jp={jp},k={k}: got {s}, expected {expected}")
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
