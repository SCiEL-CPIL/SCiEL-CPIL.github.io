import time
import numpy as np

DIMENSIONS = [3, 4, 7]

def construct():
    start_time = time.time()
    deadline = start_time + 110
    
    n, m, p = DIMENSIONS
    
    # Define blocks
    # i blocks: [0,1], [2]
    # j blocks: [0,1], [2,3]
    # k blocks: [0,1], [2,3], [4,5], [6]
    
    i_blocks = [[0, 1], [2]]
    j_blocks = [[0, 1], [2, 3]]
    k_blocks = [[0, 1], [2, 3], [4, 5], [6]]
    
    U = np.zeros((n * m, 1000), dtype=float) # Pre-allocate large enough
    V = np.zeros((m * p, 1000), dtype=float)
    W = np.zeros((p * n, 1000), dtype=float)
    
    col = 0
    
    # Helper to add a block decomposition
    def add_block(i_blk, j_blk, k_blk, u_local, v_local, w_local, rank):
        nonlocal col
        for t in range(rank):
            for i_l in range(len(i_blk)):
                for j_l in range(len(j_blk)):
                    val_u = u_local[i_l, j_l, t]
                    if val_u != 0:
                        i_g = i_blk[i_l]
                        j_g = j_blk[j_l]
                        U[i_g * m + j_g, col + t] += val_u
            for j_l in range(len(j_blk)):
                for k_l in range(len(k_blk)):
                    val_v = v_local[j_l, k_l, t]
                    if val_v != 0:
                        j_g = j_blk[j_l]
                        k_g = k_blk[k_l]
                        V[j_g * p + k_g, col + t] += val_v
            for k_l in range(len(k_blk)):
                for i_l in range(len(i_blk)):
                    val_w = w_local[k_l, i_l, t]
                    if val_w != 0:
                        k_g = k_blk[k_l]
                        i_g = i_blk[i_l]
                        W[k_g * n + i_g, col + t] += val_w
        col += rank

    # Strassen 2x2x2 decomposition
    # u, v, w are 2x2x7
    u_s = np.zeros((2, 2, 7))
    v_s = np.zeros((2, 2, 7))
    w_s = np.zeros((2, 2, 7))
    
    # M1
    u_s[0,0,0] = 1; u_s[1,1,0] = 1
    v_s[0,0,0] = 1; v_s[0,1,0] = 1; v_s[1,0,0] = 1; v_s[1,1,0] = 1
    w_s[0,0,0] = 1; w_s[1,1,0] = 1
    
    # M2
    u_s[1,0,1] = 1; u_s[1,1,1] = 1
    v_s[0,0,1] = 1
    w_s[0,1,1] = 1; w_s[1,1,1] = 1
    
    # M3
    u_s[0,0,2] = 1
    v_s[0,1,2] = 1; v_s[1,1,2] = -1
    w_s[0,0,2] = 1; w_s[1,0,2] = 1
    
    # M4
    u_s[1,1,3] = 1
    v_s[1,0,3] = 1; v_s[0,0,3] = -1
    w_s[0,0,3] = 1; w_s[0,1,3] = 1
    
    # M5
    u_s[0,0,4] = 1; u_s[0,1,4] = 1
    v_s[1,1,4] = 1
    w_s[0,0,4] = -1; w_s[1,0,4] = -1
    
    # M6
    u_s[1,0,5] = 1; u_s[0,0,5] = -1
    v_s[0,0,5] = 1; v_s[0,1,5] = 1
    w_s[1,1,5] = 1
    
    # M7
    u_s[0,1,6] = 1; u_s[1,1,6] = -1
    v_s[1,0,6] = 1; v_s[1,1,6] = 1
    w_s[0,0,6] = 1; w_s[1,1,6] = -1
    
    # Schoolbook for other sizes
    def schoolbook(n_l, m_l, p_l):
        u = np.zeros((n_l, m_l, n_l*m_l*p_l))
        v = np.zeros((m_l, p_l, n_l*m_l*p_l))
        w = np.zeros((p_l, n_l, n_l*m_l*p_l))
        c = 0
        for i in range(n_l):
            for j in range(m_l):
                for k in range(p_l):
                    u[i,j,c] = 1
                    v[j,k,c] = 1
                    w[k,i,c] = 1
                    c += 1
        return u, v, w, c

    # Process each block
    for i_blk in i_blocks:
        for j_blk in j_blocks:
            for k_blk in k_blocks:
                ni = len(i_blk)
                nj = len(j_blk)
                nk = len(k_blk)
                
                if ni == 2 and nj == 2 and nk == 2:
                    add_block(i_blk, j_blk, k_blk, u_s, v_s, w_s, 7)
                else:
                    u_l, v_l, w_l, rank = schoolbook(ni, nj, nk)
                    add_block(i_blk, j_blk, k_blk, u_l, v_l, w_l, rank)
    
    # Trim to actual rank
    U = U[:, :col]
    V = V[:, :col]
    W = W[:, :col]
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
