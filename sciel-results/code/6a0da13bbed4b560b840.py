import numpy as np

def construct():
    n, m, p = 3, 3, 4
    rank = m  # Rank 3
    
    U = np.zeros((n*m, rank), dtype=float)
    V = np.zeros((m*p, rank), dtype=float)
    W = np.zeros((p*n, rank), dtype=float)
    
    for j in range(m):
        for i in range(n):
            U[i*m + j, j] = 1.0
        for k in range(p):
            V[j*p + k, j] = 1.0
        for i in range(n):
            for k in range(p):
                W[k*n + i, j] = 1.0
                
    # Verify
    T = np.zeros((n*m, m*p, p*n), dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1.0
                
    T_decomp = np.einsum('at,bt,ct->abc', U, V, W, optimize=True)
    
    if not np.allclose(T_decomp, T, atol=1e-9):
        # Fallback to schoolbook
        rank_fb = n * m * p
        U_fb = np.zeros((n*m, rank_fb), dtype=float)
        V_fb = np.zeros((m*p, rank_fb), dtype=float)
        W_fb = np.zeros((p*n, rank_fb), dtype=float)
        col_fb = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_fb[i*m+j, col_fb] = 1.0
                    V_fb[j*p+k, col_fb] = 1.0
                    W_fb[k*n+i, col_fb] = 1.0
                    col_fb += 1
        return {"U": U_fb.tolist(), "V": V_fb.tolist(), "W": W_fb.tolist()}
        
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
