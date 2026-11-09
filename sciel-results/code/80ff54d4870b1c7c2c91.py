"""Matrix multiplication 3x4x8: rank-minimal exact tensor decomposition."""
import numpy as np

DIMENSIONS = [3, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 3, 4, 8
    rank = n * m * p  # 96
    
    # Pre-compute the standard Kronecker product decomposition
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    for r in range(rank):
        i = r // (m * p)
        remainder = r % (m * p)
        j = remainder // p
        k = remainder % p
        U[i * m + j, r] = 1
        V[j * p + k, r] = 1
        W[k * n + i, r] = 1
    
    # Return as lists of lists (standard Python types for exact integer values)
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
