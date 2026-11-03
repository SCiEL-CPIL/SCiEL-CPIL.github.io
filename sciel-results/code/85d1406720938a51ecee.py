import numpy as np
import time

DIMENSIONS = [3, 4, 7]

def construct():
    start_time = time.time()
    deadline = start_time + 110  # safety margin before 120s
    
    n, m, p = DIMENSIONS
    
    n_blocks = [(0, 2), (2, 1)]
    m_blocks = [(0, 2), (2, 2)]
    p_blocks = [(0, 2), (2, 2), (4, 2), (6, 1)]
    
    U_cols = []
    V_cols = []
    W_cols = []
    
    def add_term(U_local, V_local, W_local, i_off, j_off, k_off, i_size, j_size, k_size):
        rank_local = U_local.shape[1]
        for c in range(rank_local):
            u_col = [0] * (n * m)
            v_col = [0] * (m * p)
            w_col = [0] * (p * n)
            for ii in range(i_size):
                for jj in range(j_size):
                    u_col[(i_off + ii) * m + (j_off + jj)] = U_local[ii * j_size + jj, c]
            for jj in range(j_size):
                for kk in range(k_size):
                    v_col[(j_off + jj) * p + (k_off + kk)] = V_local[jj * k_size + kk, c]
            for kk in range(k_size):
                for ii in range(i_size):
                    w_col[(k_off + kk) * n + (i_off + ii)] = W_local[kk * i_size + ii, c]
            U_cols.append(u_col)
            V_cols.append(v_col)
            W_cols.append(w_col)
    
    def schoolbook(i_size, j_size, k_size):
        rank = i_size * j_size * k_size
        U = np.zeros((i_size * j_size, rank))
        V = np.zeros((j_size * k_size, rank))
        W = np.zeros((k_size * i_size, rank))
        c = 0
        for ii in range(i_size):
            for jj in range(j_size):
                for kk in range(k_size):
                    U[ii * j_size + jj, c] = 1
                    V[jj * k_size + kk, c] = 1
                    W[kk * i_size + ii, c] = 1
                    c += 1
        return U, V, W
    
    def strassen_222():
        U = np.zeros((4, 7))
        V = np.zeros((4, 7))
        W = np.zeros((4, 7))
        # Term 0: (A[0,0]+A[1,1])*(B[0,0]+B[1,1])
        U[0, 0] = 1; U[3, 0] = 1
        V[0, 0] = 1; V[3, 0] = 1
        W[0, 0] = 1; W[3, 0] = 1
        # Term 1: (A[1,0]+A[1,1])*B[0,0]
        U[2, 1] = 1; U[3, 1] = 1
        V[0, 1] = 1
        W[2, 1] = 1; W[3, 1] = -1
        # Term 2: A[0,0]*(B[0,1]-B[1,1])
        U[0, 2] = 1
        V[1, 2] = 1; V[3, 2] = -1
        W[1, 2] = 1; W[3, 2] = 1
        # Term 3: A[1,1]*(B[1,0]-B[0,0])
        U[3, 3] = 1
        V[2, 3] = 1; V[0, 3] = -1
        W[0, 3] = 1; W[2, 3] = 1
        # Term 4: (A[0,0]+A[0,1])*B[1,1]
        U[0, 4] = 1; U[1, 4] = 1
        V[3, 4] = 1
        W[0, 4] = -1; W[1, 4] = 1
        # Term 5: (A[1,0]-A[0,0])*(B[0,0]+B[0,1])
        U[2, 5] = 1; U[0, 5] = -1
        V[0, 5] = 1; V[1, 5] = 1
        W[3, 5] = 1
        # Term 6: (A[0,1]-A[1,1])*(B[1,0]+B[1,1])
        U[1, 6] = 1; U[3, 6] = -1
        V[2, 6] = 1; V[3, 6] = 1
        W[0, 6] = 1
        return U, V, W
    
    for (i_off, i_size) in n_blocks:
        for (j_off, j_size) in m_blocks:
            for (k_off, k_size) in p_blocks:
                if time.time() > deadline:
                    break
                if i_size == 2 and j_size == 2 and k_size == 2:
                    U_loc, V_loc, W_loc = strassen_222()
                else:
                    U_loc, V_loc, W_loc = schoolbook(i_size, j_size, k_size)
                add_term(U_loc, V_loc, W_loc, i_off, j_off, k_off, i_size, j_size, k_size)
    
    rank = len(U_cols)
    U_out = [[U_cols[c][r] for c in range(rank)] for r in range(n * m)]
    V_out = [[V_cols[c][r] for c in range(rank)] for r in range(m * p)]
    W_out = [[W_cols[c][r] for c in range(rank)] for r in range(p * n)]
    
    return {"U": U_out, "V": V_out, "W": W_out}
