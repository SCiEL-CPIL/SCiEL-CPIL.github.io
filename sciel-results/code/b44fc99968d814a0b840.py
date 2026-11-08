import numpy as np
import time

DIMENSIONS = [3, 4, 5]
DEADLINE = 140  # seconds, with 10s safety margin before 150s kill

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    
    def time_remaining():
        return DEADLINE - (time.time() - start_time)
    
    # Total rank is fixed for this specific decomposition strategy
    # 2x4x5 block: 36
    # 1x4x5 block: 20
    # Total: 56
    TOTAL_RANK = 56
    
    U = np.zeros((n * m, TOTAL_RANK))
    V = np.zeros((m * p, TOTAL_RANK))
    W = np.zeros((p * n, TOTAL_RANK))
    
    rank_counter = [0] # Use list to allow modification in nested function
    
    def add_block(u_idx, v_idx, w_idx, U_blk, V_blk, W_blk):
        r_start = rank_counter[0]
        r_end = r_start + U_blk.shape[1]
        
        for r in range(U_blk.shape[1]):
            col = r_start + r
            for i, idx in enumerate(u_idx):
                U[idx, col] = U_blk[i, r]
            for i, idx in enumerate(v_idx):
                V[idx, col] = V_blk[i, r]
            for i, idx in enumerate(w_idx):
                W[idx, col] = W_blk[i, r]
                
        rank_counter[0] = r_end

    # Strassen 2x2x2: rank 7
    def strassen_2x2x2():
        U = np.zeros((4, 7))
        V = np.zeros((4, 7))
        W = np.zeros((4, 7))
        
        U[0, 0] = 1; U[3, 0] = 1
        V[0, 0] = 1; V[3, 0] = 1
        W[0, 0] = 1; W[3, 0] = 1
        
        U[2, 1] = 1; U[3, 1] = 1
        V[0, 1] = 1
        W[2, 1] = 1; W[3, 1] = -1
        
        U[0, 2] = 1
        V[1, 2] = 1; V[3, 2] = -1
        W[1, 2] = 1; W[3, 2] = 1
        
        U[3, 3] = 1
        V[2, 3] = 1; V[0, 3] = -1
        W[0, 3] = 1; W[2, 3] = 1
        
        U[0, 4] = 1; U[1, 4] = 1
        V[3, 4] = 1
        W[0, 4] = -1; W[1, 4] = 1
        
        U[2, 5] = 1; U[0, 5] = -1
        V[0, 5] = 1; V[1, 5] = 1
        W[3, 5] = 1
        
        U[1, 6] = 1; U[3, 6] = -1
        V[2, 6] = 1; V[3, 6] = 1
        W[0, 6] = 1
        
        return U, V, W

    # Schoolbook 2x2x1: rank 4
    def schoolbook_2x2x1():
        U = np.zeros((4, 4))
        V = np.zeros((2, 4))
        W = np.zeros((2, 4))
        
        U[0, 0] = 1; V[0, 0] = 1; W[0, 0] = 1
        U[1, 1] = 1; V[1, 1] = 1; W[0, 1] = 1
        U[2, 2] = 1; V[0, 2] = 1; W[1, 2] = 1
        U[3, 3] = 1; V[1, 3] = 1; W[1, 3] = 1
        
        return U, V, W

    # Schoolbook 1x2x2: rank 4
    def schoolbook_1x2x2():
        U = np.zeros((2, 4))
        V = np.zeros((4, 4))
        W = np.zeros((2, 4))
        
        U[0, 0] = 1; V[0, 0] = 1; W[0, 0] = 1
        U[1, 1] = 1; V[2, 1] = 1; W[0, 1] = 1
        U[0, 2] = 1; V[1, 2] = 1; W[1, 2] = 1
        U[1, 3] = 1; V[3, 3] = 1; W[1, 3] = 1
        
        return U, V, W

    # Schoolbook 1x2x1: rank 2
    def schoolbook_1x2x1():
        U = np.zeros((2, 2))
        V = np.zeros((2, 2))
        W = np.zeros((1, 2))
        
        U[0, 0] = 1; V[0, 0] = 1; W[0, 0] = 1
        U[1, 1] = 1; V[1, 1] = 1; W[0, 1] = 1
        
        return U, V, W

    # U indices: A is 3x4, flat index i*m+j where i=0..2, j=0..3
    # V indices: B is 4x5, flat index j*p+k where j=0..3, k=0..4
    # W indices: C is 5x3, flat index k*n+i where k=0..4, i=0..2

    # === 2x4x5 block (A rows 0-1) ===
    
    # Cols 0-1: 2x4x2, split B rows 0:4 as 2+2
    u_idx = [0, 1, 4, 5]
    v_idx = [0, 1, 5, 6]
    w_idx = [0, 3, 1, 4]
    add_block(u_idx, v_idx, w_idx, *strassen_2x2x2())
    
    u_idx = [2, 3, 6, 7]
    v_idx = [10, 11, 15, 16]
    w_idx = [0, 3, 1, 4]
    add_block(u_idx, v_idx, w_idx, *strassen_2x2x2())
    
    # Cols 2-3: 2x4x2
    u_idx = [0, 1, 4, 5]
    v_idx = [2, 3, 7, 8]
    w_idx = [6, 9, 7, 10]
    add_block(u_idx, v_idx, w_idx, *strassen_2x2x2())
    
    u_idx = [2, 3, 6, 7]
    v_idx = [12, 13, 17, 18]
    w_idx = [6, 9, 7, 10]
    add_block(u_idx, v_idx, w_idx, *strassen_2x2x2())
    
    # Col 4: 2x4x1
    u_idx = [0, 1, 4, 5]
    v_idx = [4, 9]
    w_idx = [12, 13]
    add_block(u_idx, v_idx, w_idx, *schoolbook_2x2x1())
    
    u_idx = [2, 3, 6, 7]
    v_idx = [14, 19]
    w_idx = [12, 13]
    add_block(u_idx, v_idx, w_idx, *schoolbook_2x2x1())
    
    # === 1x4x5 block (A row 2) ===
    
    # B rows 0:2
    u_idx = [8, 9]
    v_idx = [0, 1, 5, 6]
    w_idx = [2, 5]
    add_block(u_idx, v_idx, w_idx, *schoolbook_1x2x2())
    
    u_idx = [8, 9]
    v_idx = [2, 3, 7, 8]
    w_idx = [8, 11]
    add_block(u_idx, v_idx, w_idx, *schoolbook_1x2x2())
    
    u_idx = [8, 9]
    v_idx = [4, 9]
    w_idx = [14]
    add_block(u_idx, v_idx, w_idx, *schoolbook_1x2x1())
    
    # B rows 2:4
    u_idx = [10, 11]
    v_idx = [10, 11, 15, 16]
    w_idx = [2, 5]
    add_block(u_idx, v_idx, w_idx, *schoolbook_1x2x2())
    
    u_idx = [10, 11]
    v_idx = [12, 13, 17, 18]
    w_idx = [8, 11]
    add_block(u_idx, v_idx, w_idx, *schoolbook_1x2x2())
    
    u_idx = [10, 11]
    v_idx = [14, 19]
    w_idx = [14]
    add_block(u_idx, v_idx, w_idx, *schoolbook_1x2x1())
    
    # Verify correctness with vectorized tensor check
    if time_remaining() > 5:
        rank = rank_counter[0]
        # T[i,j,k] should equal sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
        U_t = U.reshape(n, m, rank)
        V_t = V.reshape(m, p, rank)
        W_t = W.reshape(p, n, rank)
        
        i_idx = np.repeat(np.arange(n), m * p)
        j_idx = np.tile(np.repeat(np.arange(m), p), n)
        k_idx = np.tile(np.arange(p), n * m)
        
        u_vals = U_t[i_idx, j_idx, :]
        v_vals = V_t[j_idx, k_idx, :]
        w_vals = W_t[k_idx, i_idx, :]
        
        tensor_vals = np.sum(u_vals * v_vals * w_vals, axis=1)
        
        max_err = np.max(np.abs(tensor_vals - 1))
        
        if max_err > 1e-10:
            # Fall back to schoolbook
            rank_school = n * m * p
            U_school = np.zeros((n * m, rank_school))
            V_school = np.zeros((m * p, rank_school))
            W_school = np.zeros((p * n, rank_school))
            col = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        U_school[i * m + j, col] = 1
                        V_school[j * p + k, col] = 1
                        W_school[k * n + i, col] = 1
                        col += 1
            return {"U": U_school.tolist(), "V": V_school.tolist(), "W": W_school.tolist()}
    
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
