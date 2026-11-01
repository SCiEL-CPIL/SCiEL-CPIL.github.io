import time

DIMENSIONS = [2, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = 145  # seconds
    
    n, m, p = DIMENSIONS
    
    # Helper to create zero matrices
    def zero_mat(rows, cols):
        return [[0] * cols for _ in range(rows)]
    
    # Helper to get Strassen decomposition for 2x2x2
    # Returns U (4x7), V (4x7), W (4x7)
    # U index: i*2+j
    # V index: j*2+k
    # W index: k*2+i
    def get_strassen_222():
        # Standard Strassen 7 products
        # M1 = (A11 + A22)(B11 + B22)
        # M2 = (A21 + A22)B11
        # M3 = A11(B21 - B22)
        # M4 = A22(B12 - B11)
        # M5 = (A11 + A12)B22
        # M6 = (A21 - A11)(B11 + B12)
        # M7 = (A12 - A22)(B21 + B22)
        # C11 = M1 + M4 - M5 + M7
        # C12 = M3 + M5
        # C21 = M2 + M4
        # C22 = M1 - M2 + M3 + M6
        
        U = zero_mat(4, 7)
        V = zero_mat(4, 7)
        W = zero_mat(4, 7)
        
        # Indices:
        # U: 0(00), 1(01), 2(10), 3(11)
        # V: 0(00), 1(01), 2(10), 3(11)
        # W: 0(00), 1(01), 2(10), 3(11)
        
        # M1
        U[0][0] = 1; U[3][0] = 1
        V[0][0] = 1; V[3][0] = 1
        W[0][0] = 1; W[3][0] = 1
        
        # M2
        U[2][1] = 1; U[3][1] = 1
        V[0][1] = 1
        W[2][1] = 1; W[3][1] = 1
        
        # M3
        U[0][2] = 1
        V[2][2] = 1; V[3][2] = -1
        W[1][2] = 1; W[3][2] = 1
        
        # M4
        U[3][3] = 1
        V[1][3] = 1; V[0][3] = -1
        W[0][3] = 1; W[2][3] = 1
        
        # M5
        U[0][4] = 1; U[1][4] = 1
        V[3][4] = 1
        W[0][4] = -1; W[1][4] = 1
        
        # M6
        U[2][5] = 1; U[0][5] = -1
        V[0][5] = 1; V[1][5] = 1
        W[3][5] = 1
        
        # M7
        U[1][6] = 1; U[3][6] = -1
        V[2][6] = 1; V[3][6] = 1
        W[0][6] = 1
        
        return U, V, W

    # Helper to get schoolbook decomposition for n x m x p
    def get_schoolbook(n_l, m_l, p_l):
        rank = n_l * m_l * p_l
        U = zero_mat(n_l * m_l, rank)
        V = zero_mat(m_l * p_l, rank)
        W = zero_mat(p_l * n_l, rank)
        col = 0
        for i in range(n_l):
            for j in range(m_l):
                for k in range(p_l):
                    U[i * m_l + j][col] = 1
                    V[j * p_l + k][col] = 1
                    W[k * n_l + i][col] = 1
                    col += 1
        return U, V, W

    # Helper to embed a decomposition (U, V, W) for n x m x p into a larger problem
    # offset_i: shift for i (n dim)
    # offset_j: shift for j (m dim)
    # offset_k: shift for k (p dim)
    # Note: U index is i*m + j. If we shift j by offset_j, the index changes.
    # We assume the sub-problem starts at j=offset_j in the global m dimension.
    # And k=offset_k in the global p dimension.
    # i is not shifted if n is same? Yes, n is 2 everywhere.
    
    def embed(U_sub, V_sub, W_sub, n_sub, m_sub, p_sub, offset_j, offset_k):
        # U_sub has shape (n_sub * m_sub, rank)
        # Global U has shape (n * m, rank)
        # We return lists of (global_row, value, col) to add to global U
        # It's easier to just build the global matrices directly in the caller
        pass

    # We will build the final U, V, W by summing contributions from parts.
    # Global dimensions: n=2, m=5, p=5
    # U: 10 x R
    # V: 25 x R
    # W: 10 x R
    
    # Parts:
    # 1. M(2, 3, 5) -> j in [0, 2]
    # 2. M(2, 2, 5) -> j in [3, 4]
    
    # M(2, 3, 5) splits into:
    # A. M(2, 3, 2) -> k in [0, 1]
    # B. M(2, 3, 3) -> k in [2, 4]
    
    # M(2, 2, 5) splits into:
    # C. M(2, 2, 2) -> k in [0, 1]
    # D. M(2, 2, 3) -> k in [2, 4]
    
    # M(2, 3, 3) splits into:
    # B1. M(2, 2, 3) -> j_sub in [0, 1] (global j [0, 1])
    # B2. M(2, 1, 3) -> j_sub in [0] (global j [2])
    
    # M(2, 2, 3) (used in D and B1) splits into:
    # D1/B1a. M(2, 2, 2) -> k_sub in [0, 1]
    # D2/B1b. M(2, 2, 1) -> k_sub in [0]
    
    # M(2, 2, 2) is Strassen.
    # M(2, 2, 1) is Schoolbook (rank 4).
    # M(2, 1, 3) is Schoolbook (rank 6).
    # M(2, 3, 2) is Schoolbook (rank 12).
    
    # We need to accumulate into global U, V, W.
    # Let's create a function that adds a decomposition to the global ones.
    
    U_global = zero_mat(10, 100) # Max rank estimate 50, allocate 100
    V_global = zero_mat(25, 100)
    W_global = zero_mat(10, 100)
    current_rank = 0
    
    def add_decomp(U_sub, V_sub, W_sub, n_sub, m_sub, p_sub, offset_j, offset_k):
        nonlocal current_rank
        r_sub = len(U_sub[0])
        # Map sub indices to global indices
        # U: sub_row = i * m_sub + j_sub. Global row = i * 5 + (j_sub + offset_j)
        # V: sub_row = j_sub * p_sub + k_sub. Global row = (j_sub + offset_j) * 5 + (k_sub + offset_k)
        # W: sub_row = k_sub * n_sub + i. Global row = (k_sub + offset_k) * 2 + i
        
        for c in range(r_sub):
            gc = current_rank + c
            for i in range(n_sub):
                for j_sub in range(m_sub):
                    val = U_sub[i * m_sub + j_sub][c]
                    if val != 0:
                        g_row = i * 5 + (j_sub + offset_j)
                        U_global[g_row][gc] += val
                        
            for j_sub in range(m_sub):
                for k_sub in range(p_sub):
                    val = V_sub[j_sub * p_sub + k_sub][c]
                    if val != 0:
                        g_row = (j_sub + offset_j) * 5 + (k_sub + offset_k)
                        V_global[g_row][gc] += val
                        
            for k_sub in range(p_sub):
                for i in range(n_sub):
                    val = W_sub[k_sub * n_sub + i][c]
                    if val != 0:
                        g_row = (k_sub + offset_k) * 2 + i
                        W_global[g_row][gc] += val
                        
        current_rank += r_sub

    # Get basic decompositions
    U_222, V_222, W_222 = get_strassen_222()
    U_221, V_221, W_221 = get_schoolbook(2, 2, 1)
    U_213, V_213, W_213 = get_schoolbook(2, 1, 3)
    U_232, V_232, W_232 = get_schoolbook(2, 3, 2)
    
    # Build M(2, 2, 3)
    # Split p=3 into 2+1
    # Part 1: M(2, 2, 2) at offset_k=0
    # Part 2: M(2, 2, 1) at offset_k=2
    # We don't need to explicitly build M(2, 2, 3) matrix if we just add the parts
    # But M(2, 2, 3) is used in M(2, 3, 3) and M(2, 2, 5)
    
    # Let's add all leaf nodes directly.
    
    # 1. M(2, 3, 2) (Schoolbook)
    # j in [0, 2], k in [0, 1]
    add_decomp(U_232, V_232, W_232, 2, 3, 2, 0, 0)
    
    # 2. M(2, 3, 3) parts
    # B1: M(2, 2, 3) at offset_j=0
    #    - B1a: M(2, 2, 2) at offset_j=0, offset_k=2 (since k in [2, 4] for M(2,3,3), and M(2,2,2) takes k=0,1)
    #    Wait, M(2, 3, 3) has k in [0, 2] local. Global k in [2, 4].
    #    So local k=0 -> global k=2.
    #    M(2, 2, 2) takes local k=0,1.
    #    So add M(2, 2, 2) at offset_j=0, offset_k=2.
    add_decomp(U_222, V_222, W_222, 2, 2, 2, 0, 2)
    
    #    - B1b: M(2, 2, 1) at offset_j=0, offset_k=4 (local k=0 -> global k=4)
    add_decomp(U_221, V_221, W_221, 2, 2, 1, 0, 4)
    
    # B2: M(2, 1, 3) at offset_j=2 (global j=2), offset_k=2 (global k=2,3,4)
    # Local k=0,1,2 -> Global k=2,3,4
    add_decomp(U_213, V_213, W_213, 2, 1, 3, 2, 2)
    
    # 3. M(2, 2, 5) parts
    # j in [3, 4]
    # C: M(2, 2, 2) at offset_j=3, offset_k=0
    add_decomp(U_222, V_222, W_222, 2, 2, 2, 3, 0)
    
    # D: M(2, 2, 3) at offset_j=3, offset_k=2
    #    - D1: M(2, 2, 2) at offset_j=3, offset_k=2
    add_decomp(U_222, V_222, W_222, 2, 2, 2, 3, 2)
    
    #    - D2: M(2, 2, 1) at offset_j=3, offset_k=4
    add_decomp(U_221, V_221, W_221, 2, 2, 1, 3, 4)
    
    # Total rank:
    # M(2, 3, 2): 12
    # M(2, 2, 2) x 3: 7 * 3 = 21
    # M(2, 2, 1) x 2: 4 * 2 = 8
    # M(2, 1, 3) x 1: 6
    # Total: 12 + 21 + 8 + 6 = 47
    
    # Trim matrices to current_rank
    U_final = [row[:current_rank] for row in U_global]
    V_final = [row[:current_rank] for row in V_global]
    W_final = [row[:current_rank] for row in W_global]
    
    return {"U": U_final, "V": V_final, "W": W_final}

# EVOLVE-BLOCK-END
