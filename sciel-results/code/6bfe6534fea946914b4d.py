"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np

DIMENSIONS = [3, 3, 3]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    # Attempt Rank 23 (Winograd)
    # Coefficients are not available in memory, so we use a placeholder that will fail verification.
    # If we had the correct coefficients, this would return rank 23.
    rank_candidate = 23
    U_cand = np.zeros((n * m, rank_candidate))
    V_cand = np.zeros((m * p, rank_candidate))
    W_cand = np.zeros((p * n, rank_candidate))
    
    # Verification of candidate
    def verify(U, V, W):
        # Generate random test matrices
        np.random.seed(42)
        A = np.random.rand(n, m)
        B = np.random.rand(m, p)
        C_true = A @ B
        
        C_comp = np.zeros((n, p))
        for i in range(n):
            for k in range(p):
                val = 0
                for r in range(U.shape[1]):
                    # U index: i*m + j
                    # V index: j*p + k
                    # W index: k*n + i
                    x = 0
                    for j in range(m):
                        x += U[i * m + j, r] * A[i, j]
                    y = 0
                    for j in range(m):
                        y += V[j * p + k, r] * B[j, k]
                    val += x * y * W[k * n + i, r]
                C_comp[i, k] = val
                
        if not np.allclose(C_true, C_comp, atol=1e-9):
            return False
        return True

    if verify(U_cand, V_cand, W_cand):
        return {"U": U_cand.tolist(), "V": V_cand.tolist(), "W": W_cand.tolist()}

    # Fallback: Schoolbook Rank 27
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
                
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
