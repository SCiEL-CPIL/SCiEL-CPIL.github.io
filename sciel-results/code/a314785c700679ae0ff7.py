import numpy as np

DIMENSIONS = [3, 4, 7]

def construct():
    n, m, p = DIMENSIONS
    
    # We will build U, V, W for the block decomposition
    # Rank 78
    
    # Helper to create Strassen 2x2x2 decomposition
    def strassen_222():
        # U: (4, 7), V: (4, 7), W: (4, 7)
        # A indices: 00, 01, 10, 11 -> 0, 1, 2, 3
        # B indices: 00, 01, 10, 11 -> 0, 1, 2, 3
        # C indices: 00, 01, 10, 11 -> 0, 1, 2, 3
        
        U = np.zeros((4, 7))
        V = np.zeros((4, 7))
        W = np.zeros((4, 7))
        
        # M1 = (A00+A11)(B00+B11)
        U[0, 0] = 1; U[3, 0] = 1
        V[0, 0] = 1; V[3, 0] = 1
        W[0, 0] = 1; W[1, 0] = 0; W[2, 0] = 0; W[3, 0] = 1
        
        # M2 = (A10+A11)B00
        U[2, 1] = 1; U[3, 1] = 1
        V[0, 1] = 1
        W[2, 1] = 1; W[3, 1] = 1
        
        # M3 = A00(B10-B11)
        U[0, 2] = 1
        V[2, 2] = 1; V[3, 2] = -1
        W[1, 2] = 1; W[3, 2] = 1
        
        # M4 = A11(B01-B00)
        U[3, 3] = 1
        V[1, 3] = 1; V[0, 3] = -1
        W[0, 3] = 1; W[2, 3] = 1
        
        # M5 = (A00+A01)B11
        U[0, 4] = 1; U[1, 4] = 1
        V[3, 4] = 1
        W[0, 4] = -1; W[1, 4] = 1
        
        # M6 = (A10-A00)(B00+B01)
        U[2, 5] = 1; U[0, 5] = -1
        V[0, 5] = 1; V[1, 5] = 1
        W[3, 5] = 1
        
        # M7 = (A01-A11)(B10+B11)
        U[1, 6] = 1; U[3, 6] = -1
        V[2, 6] = 1; V[3, 6] = 1
        W[1, 6] = 1
        
        return U, V, W

    def schoolbook_221():
        # 2x2 A, 2x1 B -> 2x1 C
        # Rank 4
        U = np.zeros((4, 4))
        V = np.zeros((2, 4)) # B is 2x1, so 2 elements
        W = np.zeros((2, 4)) # C is 2x1, so 2 elements
        
        # C0 = A00 B0 + A01 B1
        # C1 = A10 B0 + A11 B1
        
        U[0, 0] = 1; V[0, 0] = 1; W[0, 0] = 1
        U[1, 1] = 1; V[1, 1] = 1; W[0, 1] = 1
        U[2, 2] = 1; V[0, 2] = 1; W[1, 2] = 1
        U[3, 3] = 1; V[1, 3] = 1; W[1, 3] = 1
        
        return U, V, W

    def schoolbook_12p(p_dim):
        # 1x2 A, 2xp B -> 1xp C
        # Rank 2*p
        U = np.zeros((2, 2*p_dim))
        V = np.zeros((2*p_dim, 2*p_dim)) # B is 2xp, size 2*p
        W = np.zeros((p_dim, 2*p_dim)) # C is 1xp, size p
        
        # A indices: 0, 1
        # B indices: (row, col) -> row*p + col
        # C indices: col
        
        for col in range(p_dim):
            # C_col = A0 B0_col + A1 B1_col
            # Term 1: A0 * B0_col
            U[0, col*2] = 1
            V[0*p_dim + col, col*2] = 1
            W[col, col*2] = 1
            
            # Term 2: A1 * B1_col
            U[1, col*2 + 1] = 1
            V[1*p_dim + col, col*2 + 1] = 1
            W[col, col*2 + 1] = 1
            
        return U, V, W

    # Global arrays
    # U: (12, rank)
    # V: (28, rank)
    # W: (21, rank)
    
    rank_total = 0
    U_parts = []
    V_parts = []
    W_parts = []
    
    # Block 1: A11 (2x2) x B_top (2x7)
    # A11: rows 0,1; cols 0,1. Global A indices: 0,1,4,5 (i*m+j)
    # B_top: rows 0,1; cols 0..6. Global B indices: 0..6 (j*p+k) for j=0,1
    # C_top: rows 0,1; cols 0..6. Global C indices: k*n+i for i=0,1
    
    # We split B_top into 3 chunks of 2 cols and 1 chunk of 1 col
    # Chunk 0: cols 0,1. B indices: 0,1, 2,3 (j=0,1; k=0,1)
    # Chunk 1: cols 2,3. B indices: 4,5, 6,7
    # Chunk 2: cols 4,5. B indices: 8,9, 10,11
    # Chunk 3: col 6. B indices: 12, 13
    
    # Strassen for 2x2x2
    U_s, V_s, W_s = strassen_222()
    # Map to global indices
    # A indices: 0->0, 1->1, 2->4, 3->5
    a_map = [0, 1, 4, 5]
    # B indices for chunk 0: 0->0, 1->1, 2->2, 3->3
    b_map_c0 = [0, 1, 2, 3]
    # C indices for chunk 0 (cols 0,1): 
    # C is 2x2. Global C index k*n+i.
    # k=0, i=0 -> 0
    # k=0, i=1 -> 1
    # k=1, i=0 -> 3
    # k=1, i=1 -> 4
    c_map_c0 = [0, 1, 3, 4]
    
    # We need to create U, V, W for this specific block and add to global
    def add_block(U_b, V_b, W_b, a_idx, b_idx, c_idx):
        # U_b: (a_size, rank_b)
        # V_b: (b_size, rank_b)
        # W_b: (c_size, rank_b)
        # a_idx: list of global A indices
        # b_idx: list of global B indices
        # c_idx: list of global C indices
        rank_b = U_b.shape[1]
        U_g = np.zeros((n*m, rank_b))
        V_g = np.zeros((m*p, rank_b))
        W_g = np.zeros((p*n, rank_b))
        
        for i_local, i_g in enumerate(a_idx):
            U_g[i_g, :] = U_b[i_local, :]
        for j_local, j_g in enumerate(b_idx):
            V_g[j_g, :] = V_b[j_local, :]
        for k_local, k_g in enumerate(c_idx):
            W_g[k_g, :] = W_b[k_local, :]
            
        return U_g, V_g, W_g

    # Block 1.1: A11 x B_top_chunk0 (2x2x2)
    U11, V11, W11 = add_block(U_s, V_s, W_s, a_map, b_map_c0, c_map_c0)
    U_parts.append(U11)
    V_parts.append(V11)
    W_parts.append(W11)
    rank_total += 7
    
    # Block 1.2: A11 x B_top_chunk1 (2x2x2)
    b_map_c1 = [4, 5, 6, 7]
    c_map_c1 = [6, 7, 9, 10] # k=2,3 -> 2*3+0=6, 2*3+1=7, 3*3+0=9, 3*3+1=10
    U12, V12, W12 = add_block(U_s, V_s, W_s, a_map, b_map_c1, c_map_c1)
    U_parts.append(U12)
    V_parts.append(V12)
    W_parts.append(W12)
    rank_total += 7
    
    # Block 1.3: A11 x B_top_chunk2 (2x2x2)
    b_map_c2 = [8, 9, 10, 11]
    c_map_c2 = [12, 13, 15, 16] # k=4,5
    U13, V13, W13 = add_block(U_s, V_s, W_s, a_map, b_map_c2, c_map_c2)
    U_parts.append(U13)
    V_parts.append(V13)
    W_parts.append(W13)
    rank_total += 7
    
    # Block 1.4: A11 x B_top_chunk3 (2x2x1)
    U_221, V_221, W_221 = schoolbook_221()
    b_map_c3 = [12, 13] # j=0,1; k=6 -> 0*7+6=6? No.
    # B index j*p+k. j=0, k=6 -> 6. j=1, k=6 -> 13.
    # Wait, B_top is rows 0,1.
    # Chunk 3 is col 6.
    # B indices: 0*7+6=6, 1*7+6=13.
    b_map_c3 = [6, 13]
    c_map_c3 = [18, 19] # k=6, i=0 -> 18, i=1 -> 19
    U14, V14, W14 = add_block(U_221, V_221, W_221, a_map, b_map_c3, c_map_c3)
    U_parts.append(U14)
    V_parts.append(V14)
    W_parts.append(W14)
    rank_total += 4
    
    # Block 2: A12 (2x2) x B_bot (2x7)
    # A12: rows 0,1; cols 2,3. Global A indices: 2,3,6,7
    a_map_12 = [2, 3, 6, 7]
    # B_bot: rows 2,3; cols 0..6.
    # B indices: j*p+k. j=2,3.
    
    # Chunk 0: cols 0,1.
    b_map_b0 = [2*7+0, 2*7+1, 3*7+0, 3*7+1] # 14,15, 21,22
    c_map_b0 = [0, 1, 3, 4] # Same C as before (cols 0,1)
    U21, V21, W21 = add_block(U_s, V_s, W_s, a_map_12, b_map_b0, c_map_b0)
    U_parts.append(U21)
    V_parts.append(V21)
    W_parts.append(W21)
    rank_total += 7
    
    # Chunk 1: cols 2,3.
    b_map_b1 = [2*7+2, 2*7+3, 3*7+2, 3*7+3]
    c_map_b1 = [6, 7, 9, 10]
    U22, V22, W22 = add_block(U_s, V_s, W_s, a_map_12, b_map_b1, c_map_b1)
    U_parts.append(U22)
    V_parts.append(V22)
    W_parts.append(W22)
    rank_total += 7
    
    # Chunk 2: cols 4,5.
    b_map_b2 = [2*7+4, 2*7+5, 3*7+4, 3*7+5]
    c_map_b2 = [12, 13, 15, 16]
    U23, V23, W23 = add_block(U_s, V_s, W_s, a_map_12, b_map_b2, c_map_b2)
    U_parts.append(U23)
    V_parts.append(V23)
    W_parts.append(W23)
    rank_total += 7
    
    # Chunk 3: col 6.
    b_map_b3 = [2*7+6, 3*7+6] # 20, 27
    c_map_b3 = [18, 19]
    U24, V24, W24 = add_block(U_221, V_221, W_221, a_map_12, b_map_b3, c_map_b3)
    U_parts.append(U24)
    V_parts.append(V24)
    W_parts.append(W24)
    rank_total += 4
    
    # Block 3: A21 (1x2) x B_top (2x7)
    # A21: row 2; cols 0,1. Global A indices: 8, 9
    a_map_21 = [8, 9]
    # B_top: rows 0,1.
    # C_bot: row 2; cols 0..6.
    # C index k*n+i. i=2. k=0..6.
    # C indices: 0*3+2=2, 1*3+2=5, ..., 6*3+2=20.
    
    U_127, V_127, W_127 = schoolbook_12p(7)
    # B indices for B_top: 0..6, 7..13
    b_map_t_all = list(range(0, 14))
    c_map_bot_all = [2, 5, 8, 11, 14, 17, 20]
    
    U31, V31, W31 = add_block(U_127, V_127, W_127, a_map_21, b_map_t_all, c_map_bot_all)
    U_parts.append(U31)
    V_parts.append(V31)
    W_parts.append(W31)
    rank_total += 14
    
    # Block 4: A22 (1x2) x B_bot (2x7)
    # A22: row 2; cols 2,3. Global A indices: 10, 11
    a_map_22 = [10, 11]
    # B_bot: rows 2,3.
    # B indices: 14..20, 21..27
    
    b_map_b_all = list(range(14, 28))
    c_map_bot_all2 = [2, 5, 8, 11, 14, 17, 20] # Same C
    
    U41, V41, W41 = add_block(U_127, V_127, W_127, a_map_22, b_map_b_all, c_map_bot_all2)
    U_parts.append(U41)
    V_parts.append(V41)
    W_parts.append(W41)
    rank_total += 14
    
    U_final = np.hstack(U_parts)
    V_final = np.hstack(V_parts)
    W_final = np.hstack(W_parts)
    
    # Convert to lists
    return {"U": U_final.tolist(), "V": V_final.tolist(), "W": W_final.tolist()}
