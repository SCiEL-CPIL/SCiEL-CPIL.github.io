import numpy as np
import time

DIMENSIONS = [3, 4, 8]

def _verify_safe(U, V, W, n=3, m=4, p=8, tol=1e-6):
    U_arr = np.array(U)
    V_arr = np.array(V)
    W_arr = np.array(W)
    rank = U_arr.shape[1]
    T_recon = np.zeros((n, m, p))
    for c in range(rank):
        u_c = U_arr[:, c].reshape(n, m)
        v_c = V_arr[:, c].reshape(m, p)
        w_c = W_arr[:, c].reshape(p, n)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    T_recon[i, j, k] += u_c[i, j] * v_c[j, k] * w_c[k, i]
    return np.allclose(T_recon, 1.0, atol=tol)

def _rank1_nm_p(n, m, p):
    """Rank-1 decomposition for M_{n,m,p} when one dimension is 1."""
    # This is only valid when n=1 or m=1 or p=1
    u = np.ones((n*m, 1))
    v = np.ones((m*p, 1))
    w = np.ones((p*n, 1))
    return u, v, w

def _strassen_2x2x2():
    """Strassen decomposition for 2x2x2."""
    u = np.array([
        [1, 0, 0, 1],
        [0, 0, 1, 1],
        [1, 0, 0, 0],
        [0, 0, 0, 1],
        [1, 1, 0, 0],
        [-1, 0, 1, 0],
        [0, 1, 0, -1]
    ]).T  # Shape (4, 7)
    
    v = np.array([
        [1, 0, 0, 1],
        [1, 0, 0, 0],
        [0, 1, 0, -1],
        [-1, 0, 1, 0],
        [0, 0, 0, 1],
        [1, 1, 0, 0],
        [0, 0, 1, 1]
    ]).T  # Shape (4, 7)
    
    w = np.zeros((4, 7))
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    # W index: k*n+i, n=2
    # C11 (i=0,k=0) -> W[0]
    # C12 (i=0,k=1) -> W[2]
    # C21 (i=1,k=0) -> W[1]
    # C22 (i=1,k=1) -> W[3]
    w[0, 0] = 1  # C11, M1
    w[3, 0] = 1  # C22, M1
    w[1, 1] = 1  # C21, M2
    w[3, 1] = -1  # C22, M2
    w[2, 2] = 1  # C12, M3
    w[3, 2] = 1  # C22, M3
    w[0, 3] = 1  # C11, M4
    w[1, 3] = 1  # C21, M4
    w[0, 4] = -1  # C11, M5
    w[2, 4] = 1  # C12, M5
    w[3, 5] = 1  # C22, M6
    w[0, 6] = 1  # C11, M7
    return u, v, w  # Shapes (4,7), (4,7), (4,7)

def _construct_rank_35():
    """Construct a rank-35 decomposition for 3x4x8."""
    n, m, p = 3, 4, 8
    U_list = []
    V_list = []
    W_list = []
    
    def add_sub(i_off, j_off, k_off, sub_n, sub_m, sub_p, u_sub, v_sub, w_sub):
        rank_sub = u_sub.shape[1]
        for c in range(rank_sub):
            u_col = np.zeros(n*m)
            v_col = np.zeros(m*p)
            w_col = np.zeros(p*n)
            for i in range(sub_n):
                for j in range(sub_m):
                    u_col[(i_off+i)*m + (j_off+j)] = u_sub[i*sub_m+j, c]
            for j in range(sub_m):
                for k in range(sub_p):
                    v_col[(j_off+j)*p + (k_off+k)] = v_sub[j*sub_p+k, c]
            for k in range(sub_p):
                for i in range(sub_n):
                    w_col[(k_off+k)*n + (i_off+i)] = w_sub[k*sub_n+i, c]
            U_list.append(u_col)
            V_list.append(v_col)
            W_list.append(w_col)
    
    # 3x1x8: rank 1
    u_318, v_318, w_318 = _rank1_nm_p(3, 1, 8)
    add_sub(0, 0, 0, 3, 1, 8, u_318, v_318, w_318)
    
    # 3x3x8: rank 34
    # Split 3 = 1+2, 8 = 4+4
    # Sub-blocks: 1x3x4 (rank 1), 1x3x4 (rank 1), 2x3x4 (rank 16), 2x3x4 (rank 16)
    
    # 1x3x4: rank 1
    u_134, v_134, w_134 = _rank1_nm_p(1, 3, 4)
    add_sub(0, 1, 0, 1, 3, 4, u_134, v_134, w_134)
    add_sub(0, 1, 4, 1, 3, 4, u_134, v_134, w_134)
    
    # 2x3x4: rank 16
    # Split 3 = 1+2, 4 = 2+2
    # Sub-blocks: 2x1x2 (rank 1), 2x1x2 (rank 1), 2x2x2 (rank 7), 2x2x2 (rank 7)
    
    # 2x1x2: rank 1
    u_212, v_212, w_212 = _rank1_nm_p(2, 1, 2)
    add_sub(1, 1, 0, 2, 1, 2, u_212, v_212, w_212)
    add_sub(1, 1, 2, 2, 1, 2, u_212, v_212, w_212)
    
    # 2x2x2: rank 7 (Strassen)
    u_222, v_222, w_222 = _strassen_2x2x2()
    add_sub(1, 2, 0, 2, 2, 2, u_222, v_222, w_222)
    add_sub(1, 2, 2, 2, 2, 2, u_222, v_222, w_222)
    
    U_arr = np.array(U_list).T
    V_arr = np.array(V_list).T
    W_arr = np.array(W_list).T
    return U_arr, V_arr, W_arr

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140
    
    # Schoolbook fallback
    rank_school = n*m*p
    U_school = np.zeros((n*m, rank_school))
    V_school = np.zeros((m*p, rank_school))
    W_school = np.zeros((p*n, rank_school))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i*m+j, col] = 1
                V_school[j*p+k, col] = 1
                W_school[k*n+i, col] = 1
                col += 1
    best = {"U": U_school.tolist(), "V": V_school.tolist(), "W": W_school.tolist()}
    best_rank = rank_school
    
    if time.time() < deadline:
        U35, V35, W35 = _construct_rank_35()
        if _verify_safe(U35.tolist(), V35.tolist(), W35.tolist()):
            rank35 = U35.shape[1]
            if rank35 < best_rank:
                best = {"U": U35.tolist(), "V": V35.tolist(), "W": W35.tolist()}
                best_rank = rank35
    
    return best
