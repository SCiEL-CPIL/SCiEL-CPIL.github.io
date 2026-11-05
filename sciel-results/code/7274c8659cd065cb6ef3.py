import numpy as np

DIMENSIONS = [4, 5, 6]

def construct():
    n, m, p = DIMENSIONS
    # Schoolbook decomposition
    # A is n x m, B is m x p, C is n x p
    # C[i, j] = sum_k A[i, k] B[k, j]
    
    rank = n * m * p
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    col = 0
    for i in range(n):
        for k in range(m):
            for j in range(p):
                # U index for A[i, k]
                u_idx = i * m + k
                # V index for B[k, j]
                v_idx = k * p + j
                # W index for C[i, j]
                # Contract says W[k*n+i, r]. 
                # In our context, 'k' in contract formula likely maps to column index of C (j)
                # and 'i' maps to row index of C (i).
                # So w_idx = j * n + i
                w_idx = j * n + i
                
                U[u_idx, col] = 1.0
                V[v_idx, col] = 1.0
                W[w_idx, col] = 1.0
                col += 1
                
    return {"U": U, "V": V, "W": W}
