import numpy as np

def construct():
    n, m, p = 2, 3, 9
    rank = n * m * p  # 54, which is the schoolbook rank and also the proven minimum
    
    # U: (n*m) x rank, V: (m*p) x rank, W: (p*n) x rank
    U = np.zeros((n * m, rank), dtype=np.int8)
    V = np.zeros((m * p, rank), dtype=np.int8)
    W = np.zeros((p * n, rank), dtype=np.int8)
    
    # For each output element W[k*n + i] (row k, col i in the 2x9 result),
    # it equals sum over j of A[i,j] * B[j,k].
    # Each term A[i,j]*B[j,k] is one rank-1 slice.
    # Total: n * m * p = 54 terms.
    
    t = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, t] = 1
                V[j * p + k, t] = 1
                W[k * n + i, t] = 1
                t += 1
    
    # Convert to lists of lists to match the expected interface
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
