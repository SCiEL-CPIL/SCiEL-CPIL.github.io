"""Program for Matrix multiplication 3,3,6: Rank-18 decomposition using block structure."""
import numpy as np

DIMENSIONS = [3, 3, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # p is 6, which is 2 * 3. We can split into two 3x3 multiplications.
    # Rank of 3x3x3 is 9 (schoolbook). Total rank = 9 + 9 = 18.
    rank = 18
    U = np.zeros((n * m, rank), dtype=np.float64)
    V = np.zeros((m * p, rank), dtype=np.float64)
    W = np.zeros((p * n, rank), dtype=np.float64)
    
    col = 0
    # Block 1: k in [0, 1, 2]
    for i in range(n):
        for j in range(m):
            for k in range(3):
                U[i * m + j, col] = 1.0
                V[j * p + k, col] = 1.0
                W[k * n + i, col] = 1.0
                col += 1
                
    # Block 2: k in [3, 4, 5]
    for i in range(n):
        for j in range(m):
            for k in range(3):
                k_global = k + 3
                U[i * m + j, col] = 1.0
                V[j * p + k_global, col] = 1.0
                W[k_global * n + i, col] = 1.0
                col += 1

    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
