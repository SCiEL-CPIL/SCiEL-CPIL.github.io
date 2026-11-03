import numpy as np

DIMENSIONS = [4, 5, 6]

def construct():
    n, m, p = DIMENSIONS
    # Target rank is 108 based on the blocking strategy:
    # A (4x5) split into cols 2,2,1
    # B (5x6) split into rows 2,2,1
    # C = A1B1 + A2B2 + A3B3
    # A1B1 (4x2 x 2x6) -> 2 * (2x2 x 2x6) -> 2 * 3 * (2x2 x 2x2) -> 6 * Rank(2x2x2) = 6*7 = 42
    # A2B2 (4x2 x 2x6) -> 42
    # A3B3 (4x1 x 1x6) -> 24
    # Total 42 + 42 + 24 = 108
    
    rank = 108
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    col = 0
    
    # Strassen decomposition for 2x2x2
    # Local indices:
    # U: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    # V: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    # W: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1) -- Note: W index is k*n+i? 
    # In 2x2x2, i in 0,1; j in 0,1; k in 0,1.
    # W local index: k*2 + i.
    
    # Strassen 7 terms for 2x2x2:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B10-B00)
    # M4 = A11(B01-B11)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B10)
    # M7 = (A01-A11)(B01+B11)
    
    # We need the vectors u, v, w for each term.
    # Term 1: u=(1,0,0,1), v=(1,1,1,1), w=(1,0,1,0) -- Wait, w is output C.
    # C00 = M1 - M2 - M4 + M7
    # C01 = M3 + M4
    # C10 = M1 - M3 - M5 + M6
    # C11 = M2 + M5 - M6 - M7
    
    # Let's define the 7 rank-1 components explicitly.
    # Each component r has u_r, v_r, w_r.
    
    # r=0 (M1): u=[1,0,0,1], v=[1,1,1,1], w=[1,0,1,0] (C00, C10 coeff 1)
    # r=1 (M2): u=[0,0,1,1], v=[1,0,1,0], w=[-1,0,1,0] (C00 coeff -1, C11 coeff 1) -> w=[-1,0,0,1]? No.
    # C00 gets -M2. C11 gets +M2.
    # w for M2 should have -1 at C00 (idx 0) and 1 at C11 (idx 3). w=[-1,0,0,1].
    
    # Let's list them properly.
    # C00 (idx 0): M1 - M2 - M4 + M7
    # C01 (idx 1): M3 + M4
    # C10 (idx 2): M1 - M3 - M5 + M6
    # C11 (idx 3): M2 + M5 - M6 - M7
    
    # Term 0 (M1): u=[1,0,0,1], v=[1,1,1,1], w=[1,0,1,0]
    # Term 1 (M2): u=[0,0,1,1], v=[1,0,1,0], w=[-1,0,0,1]
    # Term 2 (M3): u=[1,0,0,0], v=[0,1,1,0], w=[0,1,-1,0]
    # Term 3 (M4): u=[0,0,0,1], v=[0,1,0,1], w=[-1,1,0,0]
    # Term 4 (M5): u=[1,1,0,0], v=[0,0,1,1], w=[0,0,-1,1]
    # Term 5 (M6): u=[0,-1,1,0], v=[1,0,1,0], w=[0,0,1,-1]
    # Term 6 (M7): u=[0,1,0,-1], v=[0,1,0,1], w=[1,0,0,-1]
    
    strassen_u = np.array([
        [1,0,0,1],
        [0,0,1,1],
        [1,0,0,0],
        [0,0,0,1],
        [1,1,0,0],
        [0,-1,1,0],
        [0,1,0,-1]
    ])
    strassen_v = np.array([
        [1,1,1,1],
        [1,0,1,0],
        [0,1,1,0],
        [0,1,0,1],
        [0,0,1,1],
        [1,0,1,0],
        [0,1,0,1]
    ])
    strassen_w = np.array([
        [1,0,1,0],
        [-1,0,0,1],
        [0,1,-1,0],
        [-1,1,0,0],
        [0,0,-1,1],
        [0,0,1,-1],
        [1,0,0,-1]
    ])
    
    def add_block(i_off, j_off, k_off, u_mat, v_mat, w_mat, start_col):
        # i_off, j_off, k_off are offsets for the 2x2x2 block
        # u_mat, v_mat, w_mat are 4x7 matrices (rows are local indices 0..3)
        # Local u idx: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
        # Global u idx: (i_off + li) * m + (j_off + lj)
        
        for r in range(7):
            for li in range(2):
                for lj in range(2):
                    # u index
                    local_u_idx = li * 2 + lj
                    global_u_idx = (i_off + li) * m + (j_off + lj)
                    U[global_u_idx, start_col + r] = u_mat[local_u_idx, r]
                    
                    # v index
                    local_v_idx = li * 2 + lj
                    global_v_idx = (j_off + li) * p + (k_off + lj)
                    V[global_v_idx, start_col + r] = v_mat[local_v_idx, r]
                    
                    # w index
                    local_w_idx = li * 2 + lj
                    # W local index is k*n+i.
                    # Local k is lj, local i is li.
                    # So local_w_idx = lj * 2 + li.
                    # My strassen_w is defined with indices 0..3 corresponding to C00, C01, C10, C11.
                    # C00: i=0, k=0 -> idx 0
                    # C01: i=0, k=1 -> idx 1
                    # C10: i=1, k=0 -> idx 2
                    # C11: i=1, k=1 -> idx 3
                    # So local_w_idx = lj * 2 + li.
                    
                    global_w_idx = (k_off + lj) * n + (i_off + li)
                    W[global_w_idx, start_col + r] = w_mat[local_w_idx, r]
        
        return start_col + 7

    # Block A1B1: i in 0..3, j in 0..1, k in 0..5
    # Split i into 0..1 and 2..3
    # Split k into 0..1, 2..3, 4..5
    
    # Sub-block 1: i 0..1, j 0..1, k 0..1
    col = add_block(0, 0, 0, strassen_u, strassen_v, strassen_w, col)
    # Sub-block 2: i 0..1, j 0..1, k 2..3
    col = add_block(0, 0, 2, strassen_u, strassen_v, strassen_w, col)
    # Sub-block 3: i 0..1, j 0..1, k 4..5
    col = add_block(0, 0, 4, strassen_u, strassen_v, strassen_w, col)
    
    # Sub-block 4: i 2..3, j 0..1, k 0..1
    col = add_block(2, 0, 0, strassen_u, strassen_v, strassen_w, col)
    # Sub-block 5: i 2..3, j 0..1, k 2..3
    col = add_block(2, 0, 2, strassen_u, strassen_v, strassen_w, col)
    # Sub-block 6: i 2..3, j 0..1, k 4..5
    col = add_block(2, 0, 4, strassen_u, strassen_v, strassen_w, col)
    
    # Block A2B2: i in 0..3, j in 2..3, k in 0..5
    # Split i into 0..1 and 2..3
    # Split k into 0..1, 2..3, 4..5
    
    # Sub-block 7: i 0..1, j 2..3, k 0..1
    col = add_block(0, 2, 0, strassen_u, strassen_v, strassen_w, col)
    # Sub-block 8: i 0..1, j 2..3, k 2..3
    col = add_block(0, 2, 2, strassen_u, strassen_v, strassen_w, col)
    # Sub-block 9: i 0..1, j 2..3, k 4..5
    col = add_block(0, 2, 4, strassen_u, strassen_v, strassen_w, col)
    
    # Sub-block 10: i 2..3, j 2..3, k 0..1
    col = add_block(2, 2, 0, strassen_u, strassen_v, strassen_w, col)
    # Sub-block 11: i 2..3, j 2..3, k 2..3
    col = add_block(2, 2, 2, strassen_u, strassen_v, strassen_w, col)
    # Sub-block 12: i 2..3, j 2..3, k 4..5
    col = add_block(2, 2, 4, strassen_u, strassen_v, strassen_w, col)
    
    # Block A3B3: i in 0..3, j in 4, k in 0..5
    # A3 is 4x1, B3 is 1x6.
    # This is 24 rank-1 terms.
    # u: i*5 + 4
    # v: 4*6 + k
    # w: k*4 + i
    
    for i in range(4):
        for k in range(6):
            U[i * m + 4, col] = 1
            V[4 * p + k, col] = 1
            W[k * n + i, col] = 1
            col += 1
            
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
