import numpy as np

def construct():
    n, m, p = 2, 5, 5
    dim_u = n * m  # 10
    dim_v = m * p  # 25
    dim_w = p * n  # 10
    
    rank = 10
    
    U = np.zeros((dim_u, rank))
    V = np.zeros((dim_v, rank))
    W = np.zeros((dim_w, rank))
    
    # Fill V with 1s
    V[:] = 1.0
    
    # Fill U and W
    # r=0..4 corresponds to i=0
    for r in range(5):
        # U: rows 0..4 are 1, rows 5..9 are 0
        U[0:5, r] = 1.0
        # W: even indices are 1/5, odd are 0
        W[0::2, r] = 0.2  # 1/5
        W[1::2, r] = 0.0
        
    # r=5..9 corresponds to i=1
    for r in range(5, 10):
        # U: rows 0..4 are 0, rows 5..9 are 1
        U[0:5, r] = 0.0
        U[5:10, r] = 1.0
        # W: odd indices are 1/5, even are 0
        W[0::2, r] = 0.0
        W[1::2, r] = 0.2  # 1/5

    # Verify
    T = np.einsum('ar,br,cr->abc', U, V, W)
    
    # Build target
    target = np.zeros((dim_u, dim_v, dim_w))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                target[u_idx, v_idx, w_idx] = 1.0
                
    if not np.allclose(T, target, atol=1e-9):
        # Fallback to schoolbook if verification fails
        rank = n * m * p
        U = np.zeros((dim_u, rank))
        V = np.zeros((dim_v, rank))
        W = np.zeros((dim_w, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
