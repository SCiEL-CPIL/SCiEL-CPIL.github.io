import numpy as np

def construct():
    # Dimensions
    n, m, p = 4, 4, 7
    
    # We will use a decomposition based on splitting B into three 4x2 blocks and one 4x1 block.
    # Block 1: 4x4x2 (Rank 28)
    # Block 2: 4x4x2 (Rank 28)
    # Block 3: 4x4x2 (Rank 28)
    # Block 4: 4x4x1 (Rank 16)
    # Total Rank = 28 * 3 + 16 = 100
    
    rank = 100
    
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    # Define local factors for 2x2x2 (Strassen)
    # U: 4x7, V: 4x7, W: 4x7
    U_222 = np.array([
        [1, 0, 1, 0, 1, -1, 0],
        [0, 0, 0, 0, 1, 0, 1],
        [0, 1, 0, 0, 0, 1, 0],
        [1, 1, 0, 1, 0, 0, -1]
    ]).T
    
    V_222 = np.array([
        [1, 1, 0, -1, 0, 1, 0],
        [0, 0, 1, 0, 0, 1, 0],
        [0, 0, 0, 1, 0, 0, 1],
        [1, 0, -1, 0, 1, 0, 1]
    ]).T
    
    W_222 = np.array([
        [1, 0, 0, 1, -1, 0, 1],
        [0, 0, 1, 0, 1, 0, 0],
        [0, 1, 0, 1, 0, 0, 0],
        [1, -1, 1, 0, 0, 1, 0]
    ]).T
    
    # Define local factors for 2x2x1 (Naive)
    # U: 4x4, V: 2x4, W: 2x4
    U_221 = np.eye(4).T
    V_221 = np.array([
        [1, 0, 1, 0],
        [0, 1, 0, 1]
    ]).T
    W_221 = np.array([
        [1, 1, 0, 0],
        [0, 0, 1, 1]
    ]).T
    
    current_rank = 0
    
    # Helper to add a 4x4x2 block
    # col_offset: which columns of B this block corresponds to (0, 1, 2, 3, 4, 5)
    # The block handles 2 columns of B.
    def add_4x4x2_block(col_start):
        nonlocal current_rank
        # 4 multiplications of 2x2x2
        # M1: A00 * B_top (rows 0-1 of B)
        # M2: A01 * B_bot (rows 2-3 of B)
        # M3: A10 * B_top (rows 0-1 of B)
        # M4: A11 * B_bot (rows 2-3 of B)
        
        # M1: A00 (i:0-1, j:0-1), B_top (j:0-1, k:col_start, col_start+1)
        # C_top (k:col_start, col_start+1, i:0-1)
        for t in range(7):
            u_idx = U_222[:, t]
            v_idx = V_222[:, t]
            w_idx = W_222[:, t]
            
            # Map U indices (4) to global U (16)
            # Local U index u_loc = i_loc * 2 + j_loc
            # Global U index = i_loc * 4 + j_loc
            for u_loc in range(4):
                val = u_idx[u_loc]
                if val != 0:
                    i_loc = u_loc // 2
                    j_loc = u_loc % 2
                    g_u = i_loc * 4 + j_loc
                    U[g_u, current_rank + t] = val
            
            # Map V indices (4) to global V (28)
            # Local V index v_loc = j_loc * 2 + k_loc
            # Global V index = (j_loc) * 2 + (k_loc + col_start)
            for v_loc in range(4):
                val = v_idx[v_loc]
                if val != 0:
                    j_loc = v_loc // 2
                    k_loc = v_loc % 2
                    g_v = j_loc * 2 + (k_loc + col_start)
                    V[g_v, current_rank + t] = val
            
            # Map W indices (4) to global W (28)
            # Local W index w_loc = k_loc * 2 + i_loc
            # Global W index = (k_loc + col_start) * 4 + i_loc
            for w_loc in range(4):
                val = w_idx[w_loc]
                if val != 0:
                    k_loc = w_loc // 2
                    i_loc = w_loc % 2
                    g_w = (k_loc + col_start) * 4 + i_loc
                    W[g_w, current_rank + t] = val
        
        current_rank += 7
        
        # M2: A01 (i:0-1, j:2-3), B_bot (j:2-3, k:col_start, col_start+1)
        # C_top (k:col_start, col_start+1, i:0-1)
        for t in range(7):
            u_idx = U_222[:, t]
            v_idx = V_222[:, t]
            w_idx = W_222[:, t]
            
            for u_loc in range(4):
                val = u_idx[u_loc]
                if val != 0:
                    i_loc = u_loc // 2
                    j_loc = u_loc % 2
                    g_u = i_loc * 4 + (j_loc + 2)
                    U[g_u, current_rank + t] = val
            
            for v_loc in range(4):
                val = v_idx[v_loc]
                if val != 0:
                    j_loc = v_loc // 2
                    k_loc = v_loc % 2
                    g_v = (j_loc + 2) * 2 + (k_loc + col_start)
                    V[g_v, current_rank + t] = val
            
            for w_loc in range(4):
                val = w_idx[w_loc]
                if val != 0:
                    k_loc = w_loc // 2
                    i_loc = w_loc % 2
                    g_w = (k_loc + col_start) * 4 + i_loc
                    W[g_w, current_rank + t] = val
        
        current_rank += 7
        
        # M3: A10 (i:2-3, j:0-1), B_top (j:0-1, k:col_start, col_start+1)
        # C_bot (k:col_start, col_start+1, i:2-3)
        for t in range(7):
            u_idx = U_222[:, t]
            v_idx = V_222[:, t]
            w_idx = W_222[:, t]
            
            for u_loc in range(4):
                val = u_idx[u_loc]
                if val != 0:
                    i_loc = u_loc // 2
                    j_loc = u_loc % 2
                    g_u = (i_loc + 2) * 4 + j_loc
                    U[g_u, current_rank + t] = val
            
            for v_loc in range(4):
                val = v_idx[v_loc]
                if val != 0:
                    j_loc = v_loc // 2
                    k_loc = v_loc % 2
                    g_v = j_loc * 2 + (k_loc + col_start)
                    V[g_v, current_rank + t] = val
            
            for w_loc in range(4):
                val = w_idx[w_loc]
                if val != 0:
                    k_loc = w_loc // 2
                    i_loc = w_loc % 2
                    g_w = (k_loc + col_start) * 4 + (i_loc + 2)
                    W[g_w, current_rank + t] = val
        
        current_rank += 7
        
        # M4: A11 (i:2-3, j:2-3), B_bot (j:2-3, k:col_start, col_start+1)
        # C_bot (k:col_start, col_start+1, i:2-3)
        for t in range(7):
            u_idx = U_222[:, t]
            v_idx = V_222[:, t]
            w_idx = W_222[:, t]
            
            for u_loc in range(4):
                val = u_idx[u_loc]
                if val != 0:
                    i_loc = u_loc // 2
                    j_loc = u_loc % 2
                    g_u = (i_loc + 2) * 4 + (j_loc + 2)
                    U[g_u, current_rank + t] = val
            
            for v_loc in range(4):
                val = v_idx[v_loc]
                if val != 0:
                    j_loc = v_loc // 2
                    k_loc = v_loc % 2
                    g_v = (j_loc + 2) * 2 + (k_loc + col_start)
                    V[g_v, current_rank + t] = val
            
            for w_loc in range(4):
                val = w_idx[w_loc]
                if val != 0:
                    k_loc = w_loc // 2
                    i_loc = w_loc % 2
                    g_w = (k_loc + col_start) * 4 + (i_loc + 2)
                    W[g_w, current_rank + t] = val
        
        current_rank += 7

    # Add 3 blocks of 4x4x2
    add_4x4x2_block(0)
    add_4x4x2_block(2)
    add_4x4x2_block(4)
    
    # Add 1 block of 4x4x1 (col 6)
    def add_4x4x1_block(col_idx):
        nonlocal current_rank
        # 4 multiplications of 2x2x1
        # M1: A00 * B_top (row 0-1 of B)
        # M2: A01 * B_bot (row 2-3 of B)
        # M3: A10 * B_top (row 0-1 of B)
        # M4: A11 * B_bot (row 2-3 of B)
        
        # M1: A00 (i:0-1, j:0-1), B_top (j:0-1, k=col_idx)
        # C_top (k=col_idx, i:0-1)
        for t in range(4):
            u_idx = U_221[:, t]
            v_idx = V_221[:, t]
            w_idx = W_221[:, t]
            
            for u_loc in range(4):
                val = u_idx[u_loc]
                if val != 0:
                    i_loc = u_loc // 2
                    j_loc = u_loc % 2
                    g_u = i_loc * 4 + j_loc
                    U[g_u, current_rank + t] = val
            
            for v_loc in range(2):
                val = v_idx[v_loc]
                if val != 0:
                    j_loc = v_loc
                    g_v = j_loc * 2 + col_idx
                    V[g_v, current_rank + t] = val
            
            for w_loc in range(2):
                val = w_idx[w_loc]
                if val != 0:
                    i_loc = w_loc
                    g_w = col_idx * 4 + i_loc
                    W[g_w, current_rank + t] = val
        
        current_rank += 4
        
        # M2: A01 (i:0-1, j:2-3), B_bot (j:2-3, k=col_idx)
        # C_top (k=col_idx, i:0-1)
        for t in range(4):
            u_idx = U_221[:, t]
            v_idx = V_221[:, t]
            w_idx = W_221[:, t]
            
            for u_loc in range(4):
                val = u_idx[u_loc]
                if val != 0:
                    i_loc = u_loc // 2
                    j_loc = u_loc % 2
                    g_u = i_loc * 4 + (j_loc + 2)
                    U[g_u, current_rank + t] = val
            
            for v_loc in range(2):
                val = v_idx[v_loc]
                if val != 0:
                    j_loc = v_loc
                    g_v = (j_loc + 2) * 2 + col_idx
                    V[g_v, current_rank + t] = val
            
            for w_loc in range(2):
                val = w_idx[w_loc]
                if val != 0:
                    i_loc = w_loc
                    g_w = col_idx * 4 + i_loc
                    W[g_w, current_rank + t] = val
        
        current_rank += 4
        
        # M3: A10 (i:2-3, j:0-1), B_top (j:0-1, k=col_idx)
        # C_bot (k=col_idx, i:2-3)
        for t in range(4):
            u_idx = U_221[:, t]
            v_idx = V_221[:, t]
            w_idx = W_221[:, t]
            
            for u_loc in range(4):
                val = u_idx[u_loc]
                if val != 0:
                    i_loc = u_loc // 2
                    j_loc = u_loc % 2
                    g_u = (i_loc + 2) * 4 + j_loc
                    U[g_u, current_rank + t] = val
            
            for v_loc in range(2):
                val = v_idx[v_loc]
                if val != 0:
                    j_loc = v_loc
                    g_v = j_loc * 2 + col_idx
                    V[g_v, current_rank + t] = val
            
            for w_loc in range(2):
                val = w_idx[w_loc]
                if val != 0:
                    i_loc = w_loc
                    g_w = col_idx * 4 + (i_loc + 2)
                    W[g_w, current_rank + t] = val
        
        current_rank += 4
        
        # M4: A11 (i:2-3, j:2-3), B_bot (j:2-3, k=col_idx)
        # C_bot (k=col_idx, i:2-3)
        for t in range(4):
            u_idx = U_221[:, t]
            v_idx = V_221[:, t]
            w_idx = W_221[:, t]
            
            for u_loc in range(4):
                val = u_idx[u_loc]
                if val != 0:
                    i_loc = u_loc // 2
                    j_loc = u_loc % 2
                    g_u = (i_loc + 2) * 4 + (j_loc + 2)
                    U[g_u, current_rank + t] = val
            
            for v_loc in range(2):
                val = v_idx[v_loc]
                if val != 0:
                    j_loc = v_loc
                    g_v = (j_loc + 2) * 2 + col_idx
                    V[g_v, current_rank + t] = val
            
            for w_loc in range(2):
                val = w_idx[w_loc]
                if val != 0:
                    i_loc = w_loc
                    g_w = col_idx * 4 + (i_loc + 2)
                    W[g_w, current_rank + t] = val
        
        current_rank += 4

    add_4x4x1_block(6)
    
    return {'U': U, 'V': V, 'W': W}
