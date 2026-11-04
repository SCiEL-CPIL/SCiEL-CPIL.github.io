def construct():
    n, m, p = 2, 5, 6
    rank = 55
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    def add_strassen_222(a_idx, b_idx, c_idx):
        nonlocal col
        # a_idx: 2x2 indices in A (i, j)
        # b_idx: 2x2 indices in B (j, k)
        # c_idx: 2x2 indices in C (i, k)
        
        # M1 = (A00 + A11)(B00 + B11)
        U[a_idx[0][0]][col] += 1
        U[a_idx[1][1]][col] += 1
        V[b_idx[0][0]][col] += 1
        V[b_idx[1][1]][col] += 1
        W[c_idx[0][0]][col] += 1
        W[c_idx[1][1]][col] += 1
        col += 1
        
        # M2 = (A10 + A11) B00
        U[a_idx[1][0]][col] += 1
        U[a_idx[1][1]][col] += 1
        V[b_idx[0][0]][col] += 1
        W[c_idx[1][0]][col] += 1
        col += 1
        
        # M3 = A00 (B01 - B11)
        U[a_idx[0][0]][col] += 1
        V[b_idx[0][1]][col] += 1
        V[b_idx[1][1]][col] -= 1
        W[c_idx[0][1]][col] += 1
        col += 1
        
        # M4 = A11 (B10 - B00)
        U[a_idx[1][1]][col] += 1
        V[b_idx[1][0]][col] += 1
        V[b_idx[0][0]][col] -= 1
        W[c_idx[1][0]][col] += 1
        W[c_idx[0][0]][col] += 1
        col += 1
        
        # M5 = (A00 + A01) B11
        U[a_idx[0][0]][col] += 1
        U[a_idx[0][1]][col] += 1
        V[b_idx[1][1]][col] += 1
        W[c_idx[0][0]][col] -= 1
        W[c_idx[0][1]][col] += 1
        col += 1
        
        # M6 = (A10 - A00)(B00 + B01)
        U[a_idx[1][0]][col] += 1
        U[a_idx[0][0]][col] -= 1
        V[b_idx[0][0]][col] += 1
        V[b_idx[0][1]][col] += 1
        W[c_idx[1][1]][col] += 1
        col += 1
        
        # M7 = (A01 - A11)(B10 + B11)
        U[a_idx[0][1]][col] += 1
        U[a_idx[1][1]][col] -= 1
        V[b_idx[1][0]][col] += 1
        V[b_idx[1][1]][col] += 1
        W[c_idx[0][0]][col] += 1
        W[c_idx[1][1]][col] += 1
        col += 1

    # Product 1: A_L (2x2) * B_T (2x6)
    # A_L: rows 0,1; cols 0,1
    a_L = [[(0,0), (0,1)], [(1,0), (1,1)]]
    
    # B_T blocks: rows 0,1; cols (0,1), (2,3), (4,5)
    b_T_blocks = [
        [[(0,0), (0,1)], [(1,0), (1,1)]],
        [[(0,2), (0,3)], [(1,2), (1,3)]],
        [[(0,4), (0,5)], [(1,4), (1,5)]]
    ]
    
    c_blocks = [
        [[(0,0), (0,1)], [(1,0), (1,1)]],
        [[(0,2), (0,3)], [(1,2), (1,3)]],
        [[(0,4), (0,5)], [(1,4), (1,5)]]
    ]
    
    for b in range(3):
        # Map local (i,k) to global (k*n+i) for W
        # W index is k*n + i
        # c_idx is (i, k)
        c_idx_mapped = [[(c_blocks[b][i][k][1], c_blocks[b][i][k][0]) for k in range(2)] for i in range(2)]
        # Wait, my c_idx in add_strassen is (i, k).
        # W index is k*n + i.
        # So for W[c_idx[i][k]], index is c_idx[i][k][1] * n + c_idx[i][k][0].
        # I will handle the mapping inside the add function or pass mapped indices.
        # Let's pass mapped indices to add_strassen.
        
        # a_idx: (i, j) -> U index i*m + j
        # b_idx: (j, k) -> V index j*p + k
        # c_idx: (i, k) -> W index k*n + i
        
        a_idx = a_L
        b_idx = b_T_blocks[b]
        
        # Prepare c_idx for W mapping
        c_idx = [[(0,0), (0,1)], [(1,0), (1,1)]] # Placeholder, we need global k, i
        # c_blocks[b] has (i, k)
        # We need to map to W index.
        # I will modify add_strassen to take raw indices and map them.
        
        add_strassen_222(a_L, b_T_blocks[b], c_blocks[b])
        
    # Product 2: A_R (2x3) * B_B (3x6)
    # A_R: rows 0,1; cols 2,3,4
    # B_B: rows 2,3,4; cols 0..5
    
    # Blocks for B_B: cols (0,1,2) and (3,4,5)
    b_B_blocks = [
        [[(2,0), (2,1), (2,2)], [(3,0), (3,1), (3,2)], [(4,0), (4,1), (4,2)]],
        [[(2,3), (2,4), (2,5)], [(3,3), (3,4), (3,5)], [(4,3), (4,4), (4,5)]]
    ]
    
    c_blocks_2 = [
        [[(0,0), (0,1), (0,2)], [(1,0), (1,1), (1,2)]],
        [[(0,3), (0,4), (0,5)], [(1,3), (1,4), (1,5)]]
    ]
    
    for b in range(2):
        # Term 1: j=0 (local)
        # A col 2 (global), B row 2 (global)
        # C cols 0,1,2 (local) -> global cols
        a_col = 2
        b_row = 2
        
        for k in range(3):
            i = 0
            U[i*m + a_col][col] = 1
            V[b_row*p + c_blocks_2[b][0][k][1]][col] = 1
            W[c_blocks_2[b][0][k][1]*n + i][col] = 1
            col += 1
            
            i = 1
            U[i*m + a_col][col] = 1
            V[b_row*p + c_blocks_2[b][0][k][1]][col] = 1
            W[c_blocks_2[b][0][k][1]*n + i][col] = 1
            col += 1
            
        # Term 2: j in {1,2} (local)
        # A cols 3,4 (global)
        # B rows 3,4 (global)
        
        # Sub-term 2a: B col 0 (local)
        # A (2x2) * B (2x1)
        # A cols 3,4. B rows 3,4. B col 0 (local).
        # C col 0 (local).
        
        a_cols = [3, 4]
        b_rows = [3, 4]
        b_col = c_blocks_2[b][0][0][1] # Global col index for local col 0
        
        for i in range(2):
            for j in range(2):
                U[i*m + a_cols[j]][col] = 1
                V[b_rows[j]*p + b_col][col] = 1
                W[c_blocks_2[b][0][0][1]*n + i][col] = 1
                col += 1
                
        # Sub-term 2b: B cols 1,2 (local)
        # A (2x2) * B (2x2)
        # A cols 3,4. B rows 3,4. B cols 1,2 (local).
        # C cols 1,2 (local).
        
        # Construct indices for Strassen
        a_idx_2b = [[(0, a_cols[0]), (0, a_cols[1])], 
                    [(1, a_cols[0]), (1, a_cols[1])]]
        b_idx_2b = [[(b_rows[0], c_blocks_2[b][0][1][1]), (b_rows[0], c_blocks_2[b][0][2][1])],
                    [(b_rows[1], c_blocks_2[b][0][1][1]), (b_rows[1], c_blocks_2[b][0][2][1])]]
        c_idx_2b = [[(0, c_blocks_2[b][0][1][1]), (0, c_blocks_2[b][0][2][1])],
                    [(1, c_blocks_2[b][0][1][1]), (1, c_blocks_2[b][0][2][1])]]
                    
        add_strassen_222(a_idx_2b, b_idx_2b, c_idx_2b)

    return {"U": U, "V": V, "W": W}
