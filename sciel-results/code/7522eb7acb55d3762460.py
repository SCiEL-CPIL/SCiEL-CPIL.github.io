"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 3]

# EVOLVE-BLOCK-START
def construct():
    """
    Return a rank-27 exact tensor decomposition (schoolbook) for 3x3 matrix multiplication.
    
    The decomposition is: T = sum_{r=1}^{rank} u_r ⊗ v_r ⊗ w_r
    where U[i*m+j, r], V[j*p+k, r], W[k*n+i, r] with the trace convention W[k*n+i, r].
    
    We use the standard schoolbook multiplication with 27 terms.
    Each term corresponds to (i, j, k) and computes C[i,k] += A[i,j] * B[j,k].
    """
    import numpy as np
    
    n, m, p = DIMENSIONS
    rank = n * m * p  # 27 for 3x3x3
    
    # Build U, V, W as lists of lists (to match expected format)
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # U row index: i*m + j (which element of A)
                # V row index: j*p + k (which element of B)
                # W row index: k*n + i (which element of C, per trace convention W[k*n+i])
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
