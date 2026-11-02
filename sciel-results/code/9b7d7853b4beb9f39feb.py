import time
import numpy as np

DIMENSIONS = [4, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110
    
    # Rank 114 decomposition
    # Part 1: A_L (4x2) * B_T (2x6) using 6 Strassen 2x2x2 blocks. Rank 6*7 = 42.
    # Part 2: A_R (4x3) * B_B (3x6) using 6 Schoolbook 2x3x2 blocks. Rank 6*12 = 72.
    
    rank = 114
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    col = 0
    
    # Helper to add a rank-1 term
    def add_term(u, v, w, col_idx):
        U[:, col_idx] = u
        V[:, col_idx] = v
        W[:, col_idx] = w

    # Strassen 2x2x2 vectors (u, v, w) for indices 0..3
    # A indices: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    # B indices: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    # C indices: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    
    strassen_222 = [
        # M1
        ([1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1]),
        # M2
        ([0, 0, 1, 1], [1, 0, 0, 0], [0, 0, 1, 0]),
        # M3
        ([1, 0, 0, 0], [0, 1, 0, -1], [0, 1, 0, 0]),
        # M4
        ([0, 0, 0, 1], [0, 0, 1, -1], [1, 0, 1, 0]),
        # M5
        ([1, 1, 0, 0], [0, 0, 0, 1], [1, -1, 0, 0]),
        # M6
        ([-1, 0, 1, 0], [1, 1, 0, 0], [0, 0, 0, 1]),
        # M7
        ([0, 1, 0, -1], [0, 0, 1, 1], [1, 0, 0, 1])
    ]
    
    # Part 1: A_L (rows 0-1, 2-3) * B_T (cols 0-1, 2-3, 4-5)
    # A_L is A[:, 0:2]. B_T is B[0:2, :].
    # Blocks of A_L: A[0:2, 0:2] (idx 0-1 in U), A[2:4, 0:2] (idx 2-3 in U)
    # Blocks of B_T: B[0:2, 0:2] (idx 0-1 in V), B[0:2, 2:4] (idx 2-3 in V), B[0:2, 4:6] (idx 4-5 in V)
    # Result C blocks: C[0:2, 0:2], C[0:2, 2:4], C[0:2, 4:6] (top half)
    #                 C[2:4, 0:2], C[2:4, 2:4], C[2:4, 4:6] (bottom half)
    
    # We need to map local 2x2 indices to global indices.
    # U global index for A[i,j] is i*m + j.
    # V global index for B[j,k] is j*p + k.
    # W global index for C[k,i] is k*n + i.
    
    # For Part 1:
    # A block (r, c) where r in {0,1} (row block), c in {0,1,2} (col block of B_T)
    # A block is A[2*r : 2*r+2, 0:2]
    # B block is B[0:2, 2*c : 2*c+2]
    # C block is C[2*c : 2*c+2, 2*r : 2*r+2] ? No.
    # C = A B.
    # A is 4x2. B is 2x6.
    # A_L = [A_1; A_2] where A_1 is 2x2, A_2 is 2x2.
    # B_T = [B_1, B_2, B_3] where B_i is 2x2.
    # C_top = A_1 B_T = [A_1 B_1, A_1 B_2, A_1 B_3]
    # C_bot = A_2 B_T = [A_2 B_1, A_2 B_2, A_2 B_3]
    # C is 4x6.
    # C[0:2, 0:2] = A_1 B_1
    # C[0:2, 2:4] = A_1 B_2
    # C[0:2, 4:6] = A_1 B_3
    # C[2:4, 0:2] = A_2 B_1
    # C[2:4, 2:4] = A_2 B_2
    # C[2:4, 4:6] = A_2 B_3
    
    for r_block in range(2): # 0 for A_1, 1 for A_2
        for c_block in range(3): # 0 for B_1, 1 for B_2, 2 for B_3
            # Map local Strassen vectors to global U, V, W
            for u_loc, v_loc, w_loc in strassen_222:
                u_glob = np.zeros(n*m)
                v_glob = np.zeros(m*p)
                w_glob = np.zeros(p*n)
                
                # A_1 is A[0:2, 0:2]. A_2 is A[2:4, 0:2].
                # Local A indices: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
                # Global A indices for A_1: 0*m+0, 0*m+1, 1*m+0, 1*m+1
                # Global A indices for A_2: 2*m+0, 2*m+1, 3*m+0, 3*m+1
                
                if r_block == 0:
                    a_offsets = [0*m+0, 0*m+1, 1*m+0, 1*m+1]
                else:
                    a_offsets = [2*m+0, 2*m+1, 3*m+0, 3*m+1]
                
                for i in range(4):
                    u_glob[a_offsets[i]] = u_loc[i]
                
                # B_1 is B[0:2, 0:2]. B_2 is B[0:2, 2:4]. B_3 is B[0:2, 4:6].
                # Local B indices: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
                # Global B indices for B_c: 0*p + 2*c + 0, 0*p + 2*c + 1, 1*p + 2*c + 0, 1*p + 2*c + 1
                
                b_offsets = [0*p + 2*c_block + 0, 0*p + 2*c_block + 1, 1*p + 2*c_block + 0, 1*p + 2*c_block + 1]
                for i in range(4):
                    v_glob[b_offsets[i]] = v_loc[i]
                
                # C block for A_r B_c is C[2*c_block : 2*c_block+2, 2*r_block : 2*r_block+2]
                # Local C indices: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
                # Global C indices: k*n + i
                # k is in [2*c_block, 2*c_block+1]
                # i is in [2*r_block, 2*r_block+1]
                
                # Local 0: k=0, i=0 -> Global k=2*c_block, i=2*r_block
                # Local 1: k=0, i=1 -> Global k=2*c_block, i=2*r_block+1
                # Local 2: k=1, i=0 -> Global k=2*c_block+1, i=2*r_block
                # Local 3: k=1, i=1 -> Global k=2*c_block+1, i=2*r_block+1
                
                c_offsets = [
                    (2*c_block)*n + (2*r_block),
                    (2*c_block)*n + (2*r_block+1),
                    (2*c_block+1)*n + (2*r_block),
                    (2*c_block+1)*n + (2*r_block+1)
                ]
                for i in range(4):
                    w_glob[c_offsets[i]] = w_loc[i]
                
                add_term(u_glob, v_glob, w_glob, col)
                col += 1
                
    # Part 2: A_R (4x3) * B_B (3x6)
    # A_R = [A_3; A_4] where A_3 is 2x3, A_4 is 2x3.
    # B_B = [B_4, B_5, B_6] where B_i is 3x2.
    # C_top2 = A_3 B_B
    # C_bot2 = A_4 B_B
    
    # Schoolbook for 2x3x2.
    # A is 2x3. B is 3x2. C is 2x2.
    # C[i,k] = sum_j A[i,j] B[j,k]
    # 12 terms.
    
    for r_block in range(2): # 0 for A_3, 1 for A_4
        for c_block in range(3): # 0 for B_4, 1 for B_5, 2 for B_6
            # A_3 is A[0:2, 2:5]. A_4 is A[2:4, 2:5].
            # B_4 is B[2:5, 0:2]. B_5 is B[2:5, 2:4]. B_6 is B[2:5, 4:6].
            
            # A global indices for A_r:
            # A_3: rows 0,1. cols 2,3,4.
            # A_4: rows 2,3. cols 2,3,4.
            
            if r_block == 0:
                a_rows = [0, 1]
            else:
                a_rows = [2, 3]
            a_cols = [2, 3, 4]
            
            # B global indices for B_c:
            # rows 2,3,4. cols 2*c_block, 2*c_block+1.
            b_rows = [2, 3, 4]
            b_cols = [2*c_block, 2*c_block+1]
            
            # C block for A_r B_c is C[2*c_block : 2*c_block+2, 2*r_block : 2*r_block+2]
            # Wait, C is 4x6.
            # A is 4x5. B is 5x6.
            # A_R is 4x3 (cols 2,3,4). B_B is 3x6 (rows 2,3,4).
            # Product is 4x6.
            # C[0:2, 0:2] comes from A[0:2, 2:5] * B[2:5, 0:2].
            # So C block rows are 0:2 if r_block=0, 2:4 if r_block=1.
            # C block cols are 0:2 if c_block=0, 2:4 if c_block=1, 4:6 if c_block=2.
            
            c_rows = [2*r_block, 2*r_block+1]
            c_cols = [2*c_block, 2*c_block+1]
            
            for i in range(2): # A row
                for j in range(3): # A col / B row
                    for k in range(2): # B col / C col
                        u_glob = np.zeros(n*m)
                        v_glob = np.zeros(m*p)
                        w_glob = np.zeros(p*n)
                        
                        a_idx = a_rows[i] * m + a_cols[j]
                        b_idx = b_rows[j] * p + b_cols[k]
                        c_idx = c_cols[k] * n + c_rows[i]
                        
                        u_glob[a_idx] = 1
                        v_glob[b_idx] = 1
                        w_glob[c_idx] = 1
                        
                        add_term(u_glob, v_glob, w_glob, col)
                        col += 1

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
