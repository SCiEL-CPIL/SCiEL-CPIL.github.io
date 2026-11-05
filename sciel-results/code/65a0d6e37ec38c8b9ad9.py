import numpy as np
import time
import sys

def construct():
    n, m, p = 3, 3, 8
    start_time = time.time()
    deadline = start_time + 130  # Safety margin
    
    # Schoolbook decomposition: rank = n*m*p = 3*3*8 = 72
    # U is (n*m, rank), V is (m*p, rank), W is (p*n, rank)
    # W[k*n + i, r] corresponds to C[i,k]
    # U[i*m + j, r] corresponds to A[i,j]
    # V[j*p + k, r] corresponds to B[j,k]
    
    rank_school = n * m * p  # 72
    U = np.zeros((n * m, rank_school))
    V = np.zeros((m * p, rank_school))
    W = np.zeros((p * n, rank_school))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    
    # Verify the schoolbook decomposition
    # Tensor T[i,j,k] = 1 for all i,j,k
    # sum_r U[i,j,r] * V[j,k,r] * W[k,i,r] should equal 1
    
    # Convert to list format as required
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
