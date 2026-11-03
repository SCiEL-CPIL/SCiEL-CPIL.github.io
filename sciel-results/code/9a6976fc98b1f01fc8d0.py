# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    deadline = time.time() + 140
    n, m, p = 4, 4, 9
    
    # 1. Define 2x2x2 Strassen decomposition (Rank 7)
    U_222 = np.zeros((4, 7))
    V_222 = np.zeros((4, 7))
    W_222 = np.zeros((4, 7))
    
    # m1: (a+d)(e+h) -> C00+C11
    U_222[0, 0] = 1; U_222[3, 0] = 1
    V_222[0, 0] = 1; V_222[3, 0] = 1
    W_222[0, 0] = 1; W_222[3, 0] = 1
    
    # m2: (c+d)e -> C10+C11
    U_222[2, 1] = 1; U_222[3, 1] = 1
    V_222[0, 1] = 1
    W_222[2, 1] = 1; W_222[3, 1] = 1
    
    # m3: a(f-h) -> C01+C11
    U_222[0, 2] = 1
    V_222[1, 2] = 1; V_222[3, 2] = -1
    W_222[1, 2] = 1; W_222[3, 2] = 1
    
    # m4: d(g-e) -> C00+C11
    U_222[3, 3] = 1
    V_222[2, 3] = 1; V_222[0, 3] = -1
    W_222[0, 3] = 1; W_222[3, 3] = 1
    
    # m5: (a+b)h -> C00-C01
    U_222[0, 4] = 1; U_222[1, 4] = 1
    V_222[3, 4] = 1
    W_222[0, 4] = 1; W_222[1, 4] = -1
    
    # m6: (c-a)(e+f) -> C10
    U_222[2, 5] = 1; U_222[0, 5] = -1
    V_222[0, 5] = 1; V_222[1, 5] = 1
    W_222[2, 5] = 1
    
    # m7: (b-d)(g+h) -> C00+C01
    U_222[1, 6] = 1; U_222[3, 6] = -1
    V_222[2, 6] = 1; V_222[3, 6] = 1
    W_222[0, 6] = 1; W_222[1, 6] = 1
    
    # 2. Construct 4x4x4 Strassen decomposition (Rank 49)
    U_444 = np.zeros((16, 49))
    V_444 = np.zeros((16, 49))
    W_444 = np.zeros((16, 49))
    
    def add_block_term(U_target, V_target, W_target, 
                       U_block, V_block, W_block,
                       a_blocks, b_blocks, c_blocks,
                       scale, col_offset):
        for s in range(7):
            col = col_offset + s
            
            for (br, bc, sign) in a_blocks:
                for i_local in range(2):
                    for j_local in range(2):
                        i_global = 2*br + i_local
                        j_global = 2*bc + j_local
                        u_idx = i_global * 4 + j_global
                        u_local_idx = i_local * 2 + j_local
                        U_target[u_idx, col] += sign * U_block[u_local_idx, s]
                        
            for (br, bc, sign) in b_blocks:
                for i_local in range(2):
                    for j_local in range(2):
                        i_global = 2*br + i_local
                        j_global = 2*bc + j_local
                        v_idx = i_global * 4 + j_global
                        v_local_idx = i_local * 2 + j_local
                        V_target[v_idx, col] += sign * V_block[v_local_idx, s]
                        
            for (br, bc, sign) in c_blocks:
                for i_local in range(2):
                    for j_local in range(2):
                        i_global = 2*br + i_local
                        j_global = 2*bc + j_local
                        w_idx = i_global * 4 + j_global
                        w_local_idx = i_local * 2 + j_local
                        W_target[w_idx, col] += sign * W_block[w_local_idx, s]

    add_block_term(U_444, V_444, W_444, U_222, V_222, W_222,
                   [(0,0,1), (1,1,1)], 
                   [(0,0,1), (1,1,1)], 
                   [(0,0,1), (1,1,1)],
                   1.0, 0)
                   
    add_block_term(U_444, V_444, W_444, U_222, V_222, W_222,
                   [(1,0,1), (1,1,1)], 
                   [(0,0,1)], 
                   [(1,0,1), (1,1,1)],
                   1.0, 7)
                   
    add_block_term(U_444, V_444, W_444, U_222, V_222, W_222,
                   [(0,0,1)], 
                   [(0,1,1), (1,1,-1)], 
                   [(0,1,1), (1,1,1)],
                   1.0, 14)
                   
    add_block_term(U_444, V_444, W_444, U_222, V_222, W_222,
                   [(1,1,1)], 
                   [(1,0,1), (0,0,-1)], 
                   [(0,0,1), (1,1,1)],
                   1.0, 21)
                   
    add_block_term(U_444, V_444, W_444, U_222, V_222, W_222,
                   [(0,0,1), (0,1,1)], 
                   [(1,1,1)], 
                   [(0,0,1), (0,1,-1)],
                   1.0, 28)
                   
    add_block_term(U_444, V_444, W_444, U_222, V_222, W_222,
                   [(1,0,1), (0,0,-1)], 
                   [(0,0,1), (0,1,1)], 
                   [(1,0,1)],
                   1.0, 35)
                   
    add_block_term(U_444, V_444, W_444, U_222, V_222, W_222,
                   [(0,1,1), (1,1,-1)], 
                   [(1,0,1), (1,1,1)], 
                   [(0,0,1), (0,1,1)],
                   1.0, 42)

    # 3. Create 4x4x3 decomposition by truncating 4x4x4
    # We take the first 3 columns of V and W from the 4x4x4 decomposition
    # This results in a valid 4x4x3 decomposition of rank 49
    U_443 = U_444.copy()
    V_443 = np.zeros((12, 49))
    W_443 = np.zeros((12, 49))
    
    for j in range(4):
        for k in range(3):
            V_443[j*3 + k, :] = V_444[j*4 + k, :]
            
    for k in range(3):
        for i in range(4):
            W_443[k*4 + i, :] = W_444[k*4 + i, :]

    # 4. Assemble full 4x4x9 decomposition using three 4x4x3 blocks
    total_rank = 3 * 49
    U_total = np.zeros((16, total_rank))
    V_total = np.zeros((36, total_rank))
    W_total = np.zeros((36, total_rank))
    
    for block_idx in range(3):
        col_start = block_idx * 49
        col_end = col_start + 49
        
        U_total[:, col_start:col_end] = U_443
        
        V_part = np.zeros((36, 49))
        for j in range(4):
            for k in range(3):
                V_part[j*9 + (block_idx*3 + k), :] = V_443[j*3 + k, :]
        V_total[:, col_start:col_end] = V_part
        
        W_part = np.zeros((36, 49))
        for k in range(3):
            for i in range(4):
                W_part[(block_idx*3 + k)*4 + i, :] = W_443[k*4 + i, :]
        W_total[:, col_start:col_end] = W_part
    
    # Fast validation using vectorized operations
    def validate_fast(U, V, W, tol=1e-9):
        U_arr = np.array(U, dtype=float)
        V_arr = np.array(V, dtype=float)
        W_arr = np.array(W, dtype=float)
        
        r = U_arr.shape[1]
        if V_arr.shape[1] != r or W_arr.shape[1] != r:
            return False
        
        # Check all (i,j,k) combinations using vectorized approach
        for i in range(n):
            for j in range(m):
                u_row = U_arr[i*m + j, :]
                for k in range(p):
                    v_row = V_arr[j*p + k, :]
                    w_row = W_arr[k*n + i, :]
                    s = np.dot(u_row, v_row * w_row)
                    if abs(s - 1.0) > tol:
                        return False
        return True

    if validate_fast(U_total, V_total, W_total):
        return {"U": U_total.tolist(), "V": V_total.tolist(), "W": W_total.tolist()}
    else:
        # Fallback to schoolbook
        rank_sb = n * m * p
        U_sb = np.zeros((n*m, rank_sb))
        V_sb = np.zeros((m*p, rank_sb))
        W_sb = np.zeros((p*n, rank_sb))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_sb[i*m + j, col] = 1
                    V_sb[j*p + k, col] = 1
                    W_sb[k*n + i, col] = 1
                    col += 1
        return {"U": U_sb.tolist(), "V": V_sb.tolist(), "W": W_sb.tolist()}
# EVOLVE-BLOCK-END
