import numpy as np

DIMENSIONS = [2, 4, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    # We aim for a rank-minimal decomposition.
    # The lower bound is max(n*m, m*p, p*n) = 28.
    # Schoolbook rank is n*m*p = 56.
    # We will construct the schoolbook decomposition using NumPy for structural difference.
    
    rank = n * m * p
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # U maps A[i,j] to the col-th multiplication
                U[i * m + j, col] = 1.0
                # V maps B[j,k] to the col-th multiplication
                V[j * p + k, col] = 1.0
                # W maps the product to C[k*n + i] (trace convention W[k*n+i, r])
                W[k * n + i, col] = 1.0
                col += 1
                
    # Convert to list of lists to match the expected interface
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
