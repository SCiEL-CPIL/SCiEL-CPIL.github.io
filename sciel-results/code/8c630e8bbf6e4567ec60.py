import numpy as np

DIMENSIONS = [2, 3, 3]

def _verify(U, V, W, n, m, p):
    """Check U,V,W of given shapes reconstruct the matmul tensor exactly."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    r = U.shape[1]
    
    # Build tensor T[k*n+i, j*m+? ... ] 
    # Convention: U[i*m+j, c], V[j*p+k, c], W[k*n+i, c]
    # T[i, j, k] = 1 (delta). Index U row = i*m+j, V row = j*p+k, W row = k*n+i.
    T = np.zeros((n * m, m * p, p * n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j, j * p + k, k * n + i] = 1.0
                
    # Reconstruct via rank-r: T[a,b,c] = sum_rank U[a,r]*V[b,r]*W[c,r]
    recon = np.einsum('ar,br,cr->abc', U, V, W)
    if recon.shape != T.shape:
        return False
    return np.allclose(recon, T, atol=1e-9)

def construct():
    n, m, p = DIMENSIONS

    def schoolbook():
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}

    # Return schoolbook as it is guaranteed correct and rank 18.
    # A rank-15 decomposition exists but requires specific coefficients.
    # Without verified coefficients, returning 18 is safer than an invalid 15.
    return schoolbook()
