import time
import numpy as np

def construct():
    n, m, p = 2, 5, 6
    start_time = time.time()
    deadline = start_time + 110.0

    # Rank 54 solution based on block decomposition
    # Split j into {0,1}, {2,3}, {4}
    # Split k into {0,1}, {2,3}, {4,5}
    # Blocks:
    # 6 blocks of type 2x2x2 (rank 7 each) -> 42
    # 3 blocks of type 2x1x2 (rank 4 each) -> 12
    # Total 54
    
    rank = 54
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    col = 0

    def add_strassen(j1, j2, k1, k2):
        nonlocal col
        # i is 0, 1
        # U indices: i*m + j
        # V indices: j*p + k
        # W indices: k*n + i
        
        u00 = 0*m + j1
        u01 = 0*m + j2
        u10 = 1*m + j1
        u11 = 1*m + j2
        
        v00 = j1*p + k1
        v01 = j1*p + k2
        v10 = j2*p + k1
        v11 = j2*p + k2
        
        w00 = k1*n + 0
        w01 = k2*n + 0
        w10 = k1*n + 1
        w11 = k2*n + 1
        
        # P1
        U[u00, col] += 1; U[u11, col] += 1
        V[v00, col] += 1; V[v11, col] += 1
        W[w00, col] += 1; W[w11, col] += 1
        col += 1
        
        # P2
        U[u10, col] += 1; U[u11, col] += 1
        V[v00, col] += 1
        W[w10, col] += 1; W[w11, col] += -1
        col += 1
        
        # P3
        U[u00, col] += 1
        V[v01, col] += 1; V[v11, col] += -1
        W[w01, col] += 1; W[w11, col] += 1
        col += 1
        
        # P4
        U[u11, col] += 1
        V[v10, col] += 1; V[v00, col] += -1
        W[w00, col] += 1; W[w10, col] += 1
        col += 1
        
        # P5
        U[u00, col] += 1; U[u01, col] += 1
        V[v11, col] += 1
        W[w00, col] += -1; W[w01, col] += 1
        col += 1
        
        # P6
        U[u10, col] += 1; U[u00, col] += -1
        V[v00, col] += 1; V[v01, col] += 1
        W[w11, col] += 1
        col += 1
        
        # P7
        U[u01, col] += 1; U[u11, col] += -1
        V[v10, col] += 1; V[v11, col] += 1
        W[w00, col] += 1
        col += 1

    def add_standard(i, j, k):
        nonlocal col
        U[i*m + j, col] = 1
        V[j*p + k, col] = 1
        W[k*n + i, col] = 1
        col += 1

    # 2x2x2 blocks
    add_strassen(0, 1, 0, 1)
    add_strassen(0, 1, 2, 3)
    add_strassen(0, 1, 4, 5)
    add_strassen(2, 3, 0, 1)
    add_strassen(2, 3, 2, 3)
    add_strassen(2, 3, 4, 5)
    
    # 2x1x2 blocks (j=4)
    for i in range(2):
        for k in range(6):
            add_standard(i, 4, k)
            
    # Verify
    def verify(U_mat, V_mat, W_mat):
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u_row = U_mat[i * m + j, :]
                    v_row = V_mat[j * p + k, :]
                    w_row = W_mat[k * n + i, :]
                    result = np.sum(u_row * v_row * w_row)
                    if abs(result - 1) > 1e-9:
                        return False
        return True

    if not verify(U, V, W):
        # Fallback to trivial rank 60
        rank_f = 60
        U_f = np.zeros((n * m, rank_f))
        V_f = np.zeros((m * p, rank_f))
        W_f = np.zeros((p * n, rank_f))
        col_f = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_f[i * m + j, col_f] = 1
                    V_f[j * p + k, col_f] = 1
                    W_f[k * n + i, col_f] = 1
                    col_f += 1
        return {"U": U_f, "V": V_f, "W": W_f}

    return {"U": U, "V": V, "W": W}
