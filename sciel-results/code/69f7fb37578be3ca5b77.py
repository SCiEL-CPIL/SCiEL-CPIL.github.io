# EVOLVE-BLOCK-START
def _verify(U, V, W, n, m, p):
    """Verify that sum_c U[i*m+j, c]*V[j*p+k, c]*W[k*n+i, c] == 1 for all i,j,k."""
    import numpy as np
    Ua = np.array(U, dtype=float)
    Va = np.array(V, dtype=float)
    Wa = np.array(W, dtype=float)
    rank = Ua.shape[1]
    # Efficient check using tensor contraction
    # T[i, j, k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    # We can reshape and use einsum or loops.
    # Given small sizes, loops are fine and transparent.
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for c in range(rank):
                    s += Ua[i*m+j, c] * Va[j*p+k, c] * Wa[k*n+i, c]
                if abs(s - 1.0) > 1e-9:
                    return False
    return True

def construct():
    n, m, p = DIMENSIONS
    rank = n * m * p
    
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j][col] = 1.0
                V[j*p+k][col] = 1.0
                W[k*n+i][col] = 1.0
                col += 1
                
    # Final verification
    if not _verify(U, V, W, n, m, p):
        raise ValueError("Internal error: Schoolbook decomposition failed verification")
        
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
