import numpy as np

DIMENSIONS = [3, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = 3, 5, 6
    
    # We will build U, V, W by concatenating blocks.
    # U shape: (n*m, rank)
    # V shape: (m*p, rank)
    # W shape: (p*n, rank)
    
    rank = 0
    U_blocks = []
    V_blocks = []
    W_blocks = []
    
    def add_block(n_s, m_s, p_s, rank_s, U_b, V_b, W_b):
        # U_b: (n_s * m_s, rank_s)
        # V_b: (m_s * p_s, rank_s)
        # W_b: (p_s * n_s, rank_s)
        nonlocal rank
        U_blocks.append(U_b)
        V_blocks.append(V_b)
        W_blocks.append(W_b)
        rank += rank_s

    # Helper to create 0 matrices
    def zeros(r, c):
        return [[0.0] * c for _ in range(r)]
        
    # Decomposition of 2x2x2 (Strassen)
    # Inputs: A (2x2), B (2x2)
    # A indices: 0,1. B indices: 0,1. C indices: 0,1.
    # Rank 7
    def strassen_2x2x2():
        # U: 4x7, V: 4x7, W: 4x7
        U = zeros(4, 7)
        V = zeros(4, 7)
        W = zeros(4, 7)
        
        # Strassen steps
        # M1 = (A00 + A11) * (B00 + B11)
        # M2 = (A10 + A11) * B00
        # M3 = A00 * (B01 - B11)
        # M4 = A11 * (B10 - B00)
        # M5 = (A00 + A01) * B11
        # M6 = (A10 - A00) * (B00 + B01)
        # M7 = (A01 - A11) * (B10 + B11)
        
        # C00 = M1 + M4 - M5 + M7
        # C01 = M3 + M5
        # C10 = M2 + M4
        # C11 = M1 - M2 + M3 + M6
        
        # U corresponds to A linear forms
        # A is 2x2. Indices: 00, 01, 10, 11
        # U[row, col]
        
        # M1: A00 + A11
        U[0][0] = 1; U[3][0] = 1
        # B00 + B11
        V[0][0] = 1; V[3][0] = 1
        # C00 += 1, C11 += 1
        W[0][0] = 1; W[3][0] = 1
        
        # M2: (A10 + A11) * B00
        U[2][1] = 1; U[3][1] = 1
        V[0][1] = 1
        # C10 += 1
        W[2][1] = 1
        
        # M3: A00 * (B01 - B11)
        U[0][2] = 1
        V[1][2] = 1; V[3][2] = -1
        # C01 += 1, C11 -= 1
        W[1][2] = 1; W[3][2] = -1
        
        # M4: A11 * (B10 - B00)
        U[3][3] = 1
        V[2][3] = 1; V[0][3] = -1
        # C00 += 1, C10 += 1
        W[0][3] = 1; W[2][3] = 1
        
        # M5: (A00 + A01) * B11
        U[0][4] = 1; U[1][4] = 1
        V[3][4] = 1
        # C00 -= 1, C01 += 1
        W[0][4] = -1; W[1][4] = 1
        
        # M6: (A10 - A00) * (B00 + B01)
        U[2][5] = 1; U[0][5] = -1
        V[0][5] = 1; V[1][5] = 1
        # C11 += 1
        W[3][5] = 1
        
        # M7: (A01 - A11) * (B10 + B11)
        U[1][6] = 1; U[3][6] = -1
        V[2][6] = 1; V[3][6] = 1
        # C00 += 1
        W[0][6] = 1
        
        return U, V, W

    # Decomposition of 2x1x2 (Outer product 2x1 * 1x2 -> 2x2) Rank 4
    def outer_2x1x2():
        U = zeros(2, 4) # A is 2x1. Indices 0,1
        V = zeros(2, 4) # B is 1x2. Indices 0,1
        W = zeros(4, 4) # C is 2x2. Indices 0,1,2,3 (00, 01, 10, 11)
        
        # C00 = A0 * B0
        U[0][0] = 1; V[0][0] = 1; W[0][0] = 1
        # C01 = A0 * B1
        U[0][1] = 1; V[1][1] = 1; W[1][1] = 1
        # C10 = A1 * B0
        U[1][2] = 1; V[0][2] = 1; W[2][2] = 1
        # C11 = A1 * B1
        U[1][3] = 1; V[1][3] = 1; W[3][3] = 1
        
        return U, V, W

    # Decomposition of 1x2x2 (1x2 * 2x2 -> 1x2) Rank 4
    def matvec_1x2x2():
        U = zeros(2, 4) # A is 1x2. Indices 0,1
        V = zeros(4, 4) # B is 2x2. Indices 0,1,2,3
        W = zeros(2, 4) # C is 1x2. Indices 0,1
        
        # C0 = A0*B0 + A1*B2
        # C1 = A0*B1 + A1*B3
        U[0][0] = 1; V[0][0] = 1; W[0][0] = 1
        U[1][1] = 1; V[2][1] = 1; W[0][1] = 1
        U[0][2] = 1; V[1][2] = 1; W[1][2] = 1
        U[1][3] = 1; V[3][3] = 1; W[1][3] = 1
        
        return U, V, W

    # Decomposition of 1x1x2 (1x1 * 1x2 -> 1x2) Rank 2
    def matvec_1x1x2():
        U = zeros(1, 2) # A is 1x1
        V = zeros(2, 2) # B is 1x2
        W = zeros(2, 2) # C is 1x2
        
        U[0][0] = 1; V[0][0] = 1; W[0][0] = 1
        U[0][1] = 1; V[1][1] = 1; W[1][1] = 1
        
        return U, V, W

    # Mapping logical blocks to indices in n, m, p
    # n=3, m=5, p=6
    # I1: rows 0,1 of A. I2: row 2 of A.
    # J1: cols 0,1 of A. J2: cols 2,3 of A. J3: col 4 of A.
    # K1: cols 0,1 of B. K2: cols 2,3 of B. K3: cols 4,5 of B.
    
    # We define a function to embed a local block decomposition into the global U,V,W
    
    # Local U indices: (i_local * m_local + j_local)
    # Global U index: (i_global * m_global + j_global)
    
    n_g, m_g, p_g = n, m, p
    
    # Define the blocks to process
    # (n_start, n_end, m_start, m_end, p_start, p_end, type)
    # type: 'strassen', 'outer', 'matvec2', 'matvec1'
    
    blocks = []
    
    # I1 (2), J1 (2), K1 (2) -> Strassen
    blocks.append((0, 2, 0, 2, 0, 2, 'strassen'))
    # I1 (2), J1 (2), K2 (2) -> Strassen
    blocks.append((0, 2, 0, 2, 2, 4, 'strassen'))
    # I1 (2), J1 (2), K3 (2) -> Strassen
    blocks.append((0, 2, 0, 2, 4, 6, 'strassen'))
    
    # I1 (2), J2 (2), K1 (2) -> Strassen
    blocks.append((0, 2, 2, 4, 0, 2, 'strassen'))
    # I1 (2), J2 (2), K2 (2) -> Strassen
    blocks.append((0, 2, 2, 4, 2, 4, 'strassen'))
    # I1 (2), J2 (2), K3 (2) -> Strassen
    blocks.append((0, 2, 2, 4, 4, 6, 'strassen'))
    
    # I1 (2), J3 (1), K1 (2) -> Outer 2x1x2
    blocks.append((0, 2, 4, 5, 0, 2, 'outer'))
    # I1 (2), J3 (1), K2 (2) -> Outer 2x1x2
    blocks.append((0, 2, 4, 5, 2, 4, 'outer'))
    # I1 (2), J3 (1), K3 (2) -> Outer 2x1x2
    blocks.append((0, 2, 4, 5, 4, 6, 'outer'))
    
    # I2 (1), J1 (2), K1 (2) -> Matvec 1x2x2
    blocks.append((2, 3, 0, 2, 0, 2, 'matvec2'))
    # I2 (1), J1 (2), K2 (2) -> Matvec 1x2x2
    blocks.append((2, 3, 0, 2, 2, 4, 'matvec2'))
    # I2 (1), J1 (2), K3 (2) -> Matvec 1x2x2
    blocks.append((2, 3, 0, 2, 4, 6, 'matvec2'))
    
    # I2 (1), J2 (2), K1 (2) -> Matvec 1x2x2
    blocks.append((2, 3, 2, 4, 0, 2, 'matvec2'))
    # I2 (1), J2 (2), K2 (2) -> Matvec 1x2x2
    blocks.append((2, 3, 2, 4, 2, 4, 'matvec2'))
    # I2 (1), J2 (2), K3 (2) -> Matvec 1x2x2
    blocks.append((2, 3, 2, 4, 4, 6, 'matvec2'))
    
    # I2 (1), J3 (1), K1 (2) -> Matvec 1x1x2
    blocks.append((2, 3, 4, 5, 0, 2, 'matvec1'))
    # I2 (1), J3 (1), K2 (2) -> Matvec 1x1x2
    blocks.append((2, 3, 4, 5, 2, 4, 'matvec1'))
    # I2 (1), J3 (1), K3 (2) -> Matvec 1x1x2
    blocks.append((2, 3, 4, 5, 4, 6, 'matvec1'))

    # Process blocks
    for n_s, n_e, m_s, m_e, p_s, p_e, typ in blocks:
        n_l = n_e - n_s
        m_l = m_e - m_s
        p_l = p_e - p_s
        
        if typ == 'strassen':
            U_l, V_l, W_l = strassen_2x2x2()
            r_l = 7
        elif typ == 'outer':
            U_l, V_l, W_l = outer_2x1x2()
            r_l = 4
        elif typ == 'matvec2':
            U_l, V_l, W_l = matvec_1x2x2()
            r_l = 4
        elif typ == 'matvec1':
            U_l, V_l, W_l = matvec_1x1x2()
            r_l = 2
        else:
            raise ValueError("Unknown type")
            
        # Create global blocks
        U_g = zeros(n_g * m_g, r_l)
        V_g = zeros(m_g * p_g, r_l)
        W_g = zeros(p_g * n_g, r_l)
        
        # Fill U_g
        for i_loc in range(n_l):
            for j_loc in range(m_l):
                # Global indices
                i_g = n_s + i_loc
                j_g = m_s + j_loc
                idx_g = i_g * m_g + j_g
                idx_l = i_loc * m_l + j_loc
                for k in range(r_l):
                    U_g[idx_g][k] = U_l[idx_l][k]
                    
        # Fill V_g
        for j_loc in range(m_l):
            for k_loc in range(p_l):
                j_g = m_s + j_loc
                k_g = p_s + k_loc
                idx_g = j_g * p_g + k_g
                idx_l = j_loc * p_l + k_loc
                for r in range(r_l):
                    V_g[idx_g][r] = V_l[idx_l][r]
                    
        # Fill W_g
        for k_loc in range(p_l):
            for i_loc in range(n_l):
                # W index is k*n + i (Trace convention W[k*n+i, r])
                k_g = p_s + k_loc
                i_g = n_s + i_loc
                idx_g = k_g * n_g + i_g
                idx_l = k_loc * n_l + i_loc
                for r in range(r_l):
                    W_g[idx_g][r] = W_l[idx_l][r]
                    
        add_block(n_l, m_l, p_l, r_l, U_g, V_g, W_g)
        
    # Concatenate all blocks
    total_rank = rank
    U = []
    V = []
    W = []
    
    # Initialize U, V, W with zeros
    for i in range(n * m):
        U.append([0.0] * total_rank)
    for i in range(m * p):
        V.append([0.0] * total_rank)
    for i in range(p * n):
        W.append([0.0] * total_rank)
        
    col_offset = 0
    for i in range(len(U_blocks)):
        U_b = U_blocks[i]
        V_b = V_blocks[i]
        W_b = W_blocks[i]
        
        n_block_rows = len(U_b)
        m_block_rows = len(V_b)
        p_block_rows = len(W_b)
        r_block_cols = len(U_b[0]) if U_b else 0
        
        for r in range(r_block_cols):
            for i_u in range(n_block_rows):
                U[i_u][col_offset + r] += U_b[i_u][r]
            for i_v in range(m_block_rows):
                V[i_v][col_offset + r] += V_b[i_v][r]
            for i_w in range(p_block_rows):
                W[i_w][col_offset + r] += W_b[i_w][r]
                
        col_offset += r_block_cols
        
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
