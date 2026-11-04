import numpy as np

DIMENSIONS = [3, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    # Splits
    # n=3: 2, 1
    # m=5: 2, 2, 1
    # p=6: 2, 2, 2
    
    n_splits = [2, 1]
    m_splits = [2, 2, 1]
    p_splits = [2, 2, 2]
    
    # Global matrices
    # U: (n*m, rank)
    # V: (m*p, rank)
    # W: (p*n, rank)
    
    # We don't know the final rank yet, so we will collect blocks and then concatenate.
    # Or we can estimate max rank and use lists of lists.
    # Better: collect U_blocks, V_blocks, W_blocks.
    
    U_blocks = []
    V_blocks = []
    W_blocks = []
    
    # Helper to get decomposition for a specific block size (bn, bm, bp)
    def get_block_decomp(bn, bm, bp):
        # Returns U, V, W for this block
        # U shape: (bn*bm, rank_block)
        # V shape: (bm*bp, rank_block)
        # W shape: (bp*bn, rank_block)
        
        if bn == 2 and bm == 2 and bp == 2:
            # Strassen
            # Inputs A (2x2), B (2x2)
            # A indices: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
            # B indices: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
            # C indices: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
            
            # 7 multiplications
            # M1 = (A00+A11)(B00+B11)
            # M2 = (A10+A11)B00
            # M3 = A00(B10-B11)
            # M4 = A11(B01-B00)
            # M5 = (A00+A01)B11
            # M6 = (A10-A00)(B00+B10)
            # M7 = (A01-A11)(B11+B01)
            
            # U columns (linear forms in A)
            # 1: A00+A11 -> [1,0,0,1]
            # 2: A10+A11 -> [0,0,1,1]
            # 3: A00       -> [1,0,0,0]
            # 4: A11       -> [0,0,0,1]
            # 5: A00+A01   -> [1,1,0,0]
            # 6: A10-A00   -> [-1,0,1,0]
            # 7: A01-A11   -> [0,1,0,-1]
            
            u_cols = [
                [1, 0, 0, 1],
                [0, 0, 1, 1],
                [1, 0, 0, 0],
                [0, 0, 0, 1],
                [1, 1, 0, 0],
                [-1, 0, 1, 0],
                [0, 1, 0, -1]
            ]
            
            # V columns (linear forms in B)
            # 1: B00+B11 -> [1,0,0,1]
            # 2: B00       -> [1,0,0,0]
            # 3: B10-B11   -> [0,1,-1,0]  Wait, B10 is idx 2, B11 is idx 3. -> [0,0,1,-1]
            # 4: B01-B00   -> B01 is idx 1, B00 is idx 0. -> [-1,1,0,0]
            # 5: B11       -> [0,0,0,1]
            # 6: B00+B10   -> [1,0,1,0]
            # 7: B11+B01   -> [0,1,0,1]
            
            v_cols = [
                [1, 0, 0, 1],
                [1, 0, 0, 0],
                [0, 0, 1, -1],
                [-1, 1, 0, 0],
                [0, 0, 0, 1],
                [1, 0, 1, 0],
                [0, 1, 0, 1]
            ]
            
            # W columns (linear forms in C)
            # C00 = M1 + M4 - M5 + M7
            # C01 = M3 + M5
            # C10 = M2 + M4
            # C11 = M1 - M2 + M3 + M6
            
            # W is (bp*bn, rank) = (4, 7)
            # Row 0 (C00): M1(1), M4(1), M5(-1), M7(1) -> [1, 0, 0, 1, -1, 0, 1]
            # Row 1 (C01): M3(1), M5(1)              -> [0, 0, 1, 0, 1, 0, 0]
            # Row 2 (C10): M2(1), M4(1)              -> [0, 1, 0, 1, 0, 0, 0]
            # Row 3 (C11): M1(1), M2(-1), M3(1), M6(1) -> [1, -1, 1, 0, 0, 1, 0]
            
            w_rows = [
                [1, 0, 0, 1, -1, 0, 1],
                [0, 0, 1, 0, 1, 0, 0],
                [0, 1, 0, 1, 0, 0, 0],
                [1, -1, 1, 0, 0, 1, 0]
            ]
            
            U = [list(col) for col in u_cols] # Transpose: U is (bn*bm, rank). Here u_cols are rows of U^T?
            # My u_cols list has 7 items, each of length 4.
            # U should be (4, 7).
            # So U[i][r] = u_cols[r][i].
            U = [[u_cols[r][i] for r in range(7)] for i in range(4)]
            V = [[v_cols[r][i] for r in range(7)] for i in range(4)]
            W = [list(row) for row in w_rows] # W is (4, 7). w_rows are already rows.
            
            return U, V, W
        else:
            # Schoolbook for other small blocks
            rank = bn * bm * bp
            U = [[0] * rank for _ in range(bn * bm)]
            V = [[0] * rank for _ in range(bm * bp)]
            W = [[0] * rank for _ in range(bp * bn)]
            
            col = 0
            for i in range(bn):
                for j in range(bm):
                    for k in range(bp):
                        u_idx = i * bm + j
                        v_idx = j * bp + k
                        w_idx = k * bn + i
                        U[u_idx][col] = 1
                        V[v_idx][col] = 1
                        W[w_idx][col] = 1
                        col += 1
            return U, V, W

    # Assemble global decomposition
    # Index offsets
    # A is n x m. Row offset i0, Col offset j0.
    # B is m x p. Row offset j0, Col offset k0.
    # C is p x n. Row offset k0, Col offset i0.
    
    total_rank = 0
    global_U = []
    global_V = []
    global_W = []
    
    n_offsets = []
    n_acc = 0
    for ns in n_splits:
        n_offsets.append(n_acc)
        n_acc += ns
        
    m_offsets = []
    m_acc = 0
    for ms in m_splits:
        m_offsets.append(m_acc)
        m_acc += ms
        
    p_offsets = []
    p_acc = 0
    for ps in p_splits:
        p_offsets.append(p_acc)
        p_acc += ps
        
    for i_idx, n_i in enumerate(n_splits):
        i_off = n_offsets[i_idx]
        for k_idx, m_k in enumerate(m_splits):
            k_off = m_offsets[k_idx]
            for j_idx, p_j in enumerate(p_splits):
                j_off = p_offsets[j_idx]
                
                # Block size: n_i x m_k x p_j
                U_b, V_b, W_b = get_block_decomp(n_i, m_k, p_j)
                
                # Shift indices
                # U_b is (n_i * m_k, rank_b)
                # Global U row index: (i_row * m + k_col)
                # i_row in [0, n_i), k_col in [0, m_k)
                # Global row = (i_off + i_row) * m + (k_off + k_col)
                
                U_g = [[0] * len(U_b[0]) for _ in range(n * m)]
                V_g = [[0] * len(V_b[0]) for _ in range(m * p)]
                W_g = [[0] * len(W_b[0]) for _ in range(p * n)]
                
                rank_b = len(U_b[0])
                
                for r in range(rank_b):
                    # U
                    for u_row in range(len(U_b)):
                        i_row = u_row // m_k
                        k_col = u_row % m_k
                        g_u_row = (i_off + i_row) * m + (k_off + k_col)
                        U_g[g_u_row][r] = U_b[u_row][r]
                        
                    # V
                    for v_row in range(len(V_b)):
                        k_row = v_row // p_j
                        j_col = v_row % p_j
                        g_v_row = (k_off + k_row) * p + (j_off + j_col)
                        V_g[g_v_row][r] = V_b[v_row][r]
                        
                    # W
                    for w_row in range(len(W_b)):
                        j_row = w_row // n_i
                        i_col = w_row % n_i
                        g_w_row = (j_off + j_row) * n + (i_off + i_col)
                        W_g[g_w_row][r] = W_b[w_row][r]
                        
                # Add to global
                if not global_U:
                    global_U = U_g
                    global_V = V_g
                    global_W = W_g
                else:
                    for row in range(len(global_U)):
                        global_U[row].extend(U_g[row])
                    for row in range(len(global_V)):
                        global_V[row].extend(V_g[row])
                    for row in range(len(global_W)):
                        global_W[row].extend(W_g[row])
                
                total_rank += rank_b

    return {"U": global_U, "V": global_V, "W": global_W}
# EVOLVE-BLOCK-END
