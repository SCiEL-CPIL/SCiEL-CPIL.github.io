"""Matrix multiplication 2,5,5 - rank minimization via decoupled 1x5x5 blocks."""
import numpy as np
import time

DIMENSIONS = [2, 5, 5]

def construct():
    n, m, p = DIMENSIONS
    
    # The 2x5 x 5x5 multiplication tensor T[i,j,k] = 1 for all valid i,j,k
    # Since the output C[i,k] = sum_j A[i,j] * B[j,k], the tensor entries are all 1.
    # 
    # Key insight: The tensor T[i,j,k] = 1 for all i,j,k can be decomposed as
    # T[i,j,k] = sum_{i'} delta_{i,i'} * 1 * 1  (summing over i')
    # This gives a rank-n decomposition where:
    #   U[i*m+j, c] = 1 if i = c (for c in range(n)), else 0
    #   V[j*p+k, c] = 1 for all j,k,c
    #   W[k*n+i, c] = 1 for all k,i,c
    # 
    # Let's verify: For fixed i,j,k:
    #   sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    #   = sum_c delta_{i,c} * 1 * 1
    #   = 1  (only c=i contributes)
    # 
    # This gives rank = n = 2.
    
    rank = n  # = 2
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    for i in range(n):
        for j in range(m):
            U[i*m + j, i] = 1.0  # U[i*m+j, c] = delta_{i,c}
        for j in range(m):
            for k in range(p):
                V[j*p + k, i] = 1.0  # V[j*p+k, c] = 1 for all c
        for k in range(p):
            W[k*n + i, i] = 1.0  # W[k*n+i, c] = delta_{i,c}... 
    
    # Wait, let me re-examine. W is indexed by k*n+i, so W[k*n+i, c].
    # We need: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1 for all i,j,k.
    #
    # With U[i*m+j, c] = delta_{i,c}, V[j*p+k, c] = 1, W[k*n+i, c] = delta_{i,c}:
    #   sum_c delta_{i,c} * 1 * delta_{i,c} = sum_c delta_{i,c} = 1.  ✓
    
    result = {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
    
    # Verify the decomposition
    max_err = 0.0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0.0
                for c in range(rank):
                    val += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                max_err = max(max_err, abs(val - 1.0))
    
    if max_err > 1e-10:
        # Fallback to schoolbook if something went wrong
        rank_sb = n * m * p
        U_sb = np.zeros((n*m, rank_sb))
        V_sb = np.zeros((m*p, rank_sb))
        W_sb = np.zeros((p*n, rank_sb))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_sb[i*m+j, col] = 1
                    V_sb[j*p+k, col] = 1
                    W_sb[k*n+i, col] = 1
                    col += 1
        result = {"U": U_sb.tolist(), "V": V_sb.tolist(), "W": W_sb.tolist()}
    
    return result
