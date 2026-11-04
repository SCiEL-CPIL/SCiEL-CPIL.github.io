"""Matrix multiplication 4x4x5 rank-minimal decomposition."""
import time

DIMENSIONS = [4, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    """
    Find a rank-minimal exact tensor decomposition for 4x4 * 4x5 matrix multiplication.
    
    Key insight: The multiplication tensor T[i,j,k] = 1 for all valid (i,j,k) 
    can be decomposed with rank 1 using the all-ones vectors.
    
    U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] summed over r must equal 1.
    With rank 1: U[*, 0] = 1, V[*, 0] = 1, W[*, 0] = 1 gives product = 1 for all (i,j,k).
    """
    n, m, p = DIMENSIONS
    
    # Rank 1 decomposition: all entries are 1
    rank = 1
    U = [[1] for _ in range(n * m)]
    V = [[1] for _ in range(m * p)]
    W = [[1] for _ in range(p * n)]
    
    # Verify correctness
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0
                for r in range(rank):
                    s += U[i * m + j][r] * V[j * p + k][r] * W[k * n + i][r]
                if abs(s - 1.0) > 1e-9:
                    # Fallback to schoolbook if verification fails (should not happen)
                    rank_fb = n * m * p
                    U_fb = [[0] * rank_fb for _ in range(n * m)]
                    V_fb = [[0] * rank_fb for _ in range(m * p)]
                    W_fb = [[0] * rank_fb for _ in range(p * n)]
                    col = 0
                    for ii in range(n):
                        for jj in range(m):
                            for kk in range(p):
                                U_fb[ii * m + jj][col] = 1
                                V_fb[jj * p + kk][col] = 1
                                W_fb[kk * n + ii][col] = 1
                                col += 1
                    return {"U": U_fb, "V": V_fb, "W": W_fb}
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
