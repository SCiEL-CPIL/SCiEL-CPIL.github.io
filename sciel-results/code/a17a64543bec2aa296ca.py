"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # We will construct a decomposition using block decomposition.
    # A is 2x5, B is 5x6.
    # Split A into A1 (2x2), A2 (2x2), A3 (2x1)
    # Split B into B1 (2x6), B2 (2x6), B3 (1x6)
    
    # Helper to add a decomposition to the main matrices
    # U, V, W are lists of lists
    # We will accumulate them
    
    # We need to track the current number of rank terms (columns)
    # Let's build U, V, W as lists of lists, appending columns.
    
    U = []
    V = []
    W = []
    
    def add_decomp(n_block, m_block, p_block, U_block, V_block, W_block, offset_i, offset_j, offset_k):
        # U_block is list of length n_block*m_block, each element list of length rank
        # V_block is list of length m_block*p_block, each element list of length rank
        # W_block is list of length p_block*n_block, each element list of length rank
        # offset_i: start index of this block in the full A (row-major)
        # offset_j: start index of this block in the full B (col-major? no, standard)
        # offset_k: start index of this block in the full C
        
        rank_block = len(U_block[0]) if U_block else 0
        
        for r in range(rank_block):
            u_col = [0.0] * (n * m)
            v_col = [0.0] * (m * p)
            w_col = [0.0] * (p * n)
            
            # Fill U
            for i in range(n_block * m_block):
                u_col[offset_i + i] = U_block[i][r]
                
            # Fill V
            # Note: B is m x p. Block is m_block x p_block.
            # The index in B is (j, k).
            # offset_j corresponds to the j-index start?
            # Let's be careful with indexing.
            # A is n x m. Index is i*m + j.
            # Block A is n_block x m_block.
            # i is 0..n_block-1, j is 0..m_block-1.
            # Global index: (i + i_start)*m + (j + j_start)
            
            # B is m x p. Index is j*p + k.
            # Block B is m_block x p_block.
            # j is 0..m_block-1, k is 0..p_block-1.
            # Global index: (j + j_start)*p + (k + k_start)
            
            # C is p x n. Index is k*n + i.
            # Block C is p_block x n_block.
            # k is 0..p_block-1, i is 0..n_block-1.
            # Global index: (k + k_start)*n + (i + i_start)
            
            for i in range(n_block * m_block):
                # Map local index i to global
                local_i = i // m_block
                local_j = i % m_block
                global_idx = (local_i + offset_i[0]) * m + (local_j + offset_i[1])
                u_col[global_idx] = U_block[i][r]
                
            for j in range(m_block * p_block):
                local_j = j // p_block
                local_k = j % p_block
                global_idx = (local_j + offset_j[0]) * p + (local_k + offset_j[1])
                v_col[global_idx] = V_block[j][r]
                
            for k in range(p_block * n_block):
                local_k = k // n_block
                local_i = k % n_block
                global_idx = (local_k + offset_k[0]) * n + (local_i + offset_k[1])
                w_col[global_idx] = W_block[k][r]
                
            U.append(u_col)
            V.append(v_col)
            W.append(w_col)
            
    # Define Strassen decomposition for 2x2x2
    def get_strassen_222():
        # Rank 7
        # U: 4 entries (2x2)
        # V: 4 entries (2x2)
        # W: 4 entries (2x2)
        
        # M1: (A11+A22)(B11+B22)
        # M2: (A21+A22)B11
        # M3: A11(B21-B22)
        # M4: A22(B12-B11)
        # M5: (A11+A12)B22
        # M6: (A21-A11)(B11+B12)
        # M7: (A12-A22)(B21+B22)
        
        # Indices for 2x2: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
        
        U = [[0.0]*7 for _ in range(4)]
        V = [[0.0]*7 for _ in range(4)]
        W = [[0.0]*7 for _ in range(4)]
        
        # U vectors (rows of A)
        # A11=0, A12=1, A21=2, A22=3
        
        # M1
        U[0][0] = 1; U[3][0] = 1
        V[0][0] = 1; V[3][0] = 1
        
        # M2
        U[2][1] = 1; U[3][1] = 1
        V[0][1] = 1
        
        # M3
        U[0][2] = 1
        V[2][2] = 1; V[3][2] = -1
        
        # M4
        U[3][3] = 1
        V[1][3] = 1; V[0][3] = -1
        
        # M5
        U[0][4] = 1; U[1][4] = 1
        V[3][4] = 1
        
        # M6
        U[2][5] = 1; U[0][5] = -1
        V[0][5] = 1; V[1][5] = 1
        
        # M7
        U[1][6] = 1; U[3][6] = -1
        V[2][6] = 1; V[3][6] = 1
        
        # W vectors (coefficients for C)
        # C11 = M1 + M4 - M5 + M7
        # C12 = M3 + M5
        # C21 = M2 + M4
        # C22 = M1 - M2 + M3 + M6
        
        W[0][0] = 1; W[0][3] = 1; W[0][4] = -1; W[0][6] = 1 # C11
        W[1][2] = 1; W[1][4] = 1 # C12
        W[2][1] = 1; W[2][3] = 1 # C21
        W[3][0] = 1; W[3][1] = -1; W[3][2] = 1; W[3][5] = 1 # C22
        
        return U, V, W

    # Define trivial decomposition for n x m x p
    def get_trivial(n_block, m_block, p_block):
        rank = n_block * m_block * p_block
        U = [[0.0]*rank for _ in range(n_block * m_block)]
        V = [[0.0]*rank for _ in range(m_block * p_block)]
        W = [[0.0]*rank for _ in range(p_block * n_block)]
        
        idx = 0
        for i in range(n_block):
            for j in range(m_block):
                for k in range(p_block):
                    U[i*m_block+j][idx] = 1.0
                    V[j*p_block+k][idx] = 1.0
                    W[k*n_block+i][idx] = 1.0
                    idx += 1
        return U, V, W

    # Block 1: 2x2x6
    # Split 6 into 2x3? Or just use trivial for 2x2x6?
    # Let's use 3 blocks of 2x2x2 for 2x2x6.
    # Rank 3*7 = 21.
    
    def get_226():
        U_list = []
        V_list = []
        W_list = []
        
        # We need to combine 3 decompositions of 2x2x2
        # The p dimension is 6. We split into 3 chunks of 2.
        # So we have 3 sub-problems.
        
        # We will build U, V, W for the 2x2x6 block
        # U size: 2*2 = 4
        # V size: 2*6 = 12
        # W size: 6*2 = 12
        
        rank_total = 21
        U = [[0.0]*rank_total for _ in range(4)]
        V = [[0.0]*rank_total for _ in range(12)]
        W = [[0.0]*rank_total for _ in range(12)]
        
        for chunk in range(3):
            U_sub, V_sub, W_sub = get_strassen_222()
            # V_sub is 2x2. We need to place it in V (2x6) at columns chunk*2 .. chunk*2+1
            # W_sub is 2x2. We need to place it in W (6x2) at rows chunk*2 .. chunk*2+1
            
            for r in range(7):
                # U
                for i in range(4):
                    U[i][chunk*7 + r] = U_sub[i][r]
                
                # V
                for j in range(2):
                    for k in range(2):
                        V[j*6 + (chunk*2 + k)][chunk*7 + r] = V_sub[j*2 + k][r]
                
                # W
                for k in range(2):
                    for i in range(2):
                        W[(chunk*2 + k)*2 + i][chunk*7 + r] = W_sub[k*2 + i][r]
                        
        return U, V, W

    # Block 2: 2x2x6 (same as Block 1)
    # Block 3: 2x1x6
    def get_216():
        return get_trivial(2, 1, 6)

    # Combine
    # A: 2x5.
    # Block 1: i=0..1, j=0..1
    # Block 2: i=0..1, j=2..3
    # Block 3: i=0..1, j=4..4
    
    # B: 5x6.
    # Block 1: j=0..1, k=0..5
    # Block 2: j=2..3, k=0..5
    # Block 3: j=4..4, k=0..5
    
    # C: 6x2.
    # Block 1: k=0..5, i=0..1
    # Block 2: k=0..5, i=0..1
    # Block 3: k=0..5, i=0..1
    
    U1, V1, W1 = get_226()
    add_decomp(2, 2, 6, U1, V1, W1, (0, 0), (0, 0), (0, 0))
    
    U2, V2, W2 = get_226()
    add_decomp(2, 2, 6, U2, V2, W2, (0, 2), (2, 0), (0, 0))
    
    U3, V3, W3 = get_216()
    add_decomp(2, 1, 6, U3, V3, W3, (0, 4), (4, 0), (0, 0))
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
