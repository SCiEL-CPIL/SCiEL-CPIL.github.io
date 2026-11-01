import numpy as np
import time

DIMENSIONS = [3, 3, 5]

def construct():
    start_time = time.time()
    n, m, p = DIMENSIONS
    
    # We will build the decomposition by concatenating blocks.
    # U, V, W will be lists of columns.
    U_cols = []
    V_cols = []
    W_cols = []
    
    def add_block(n_b, m_b, p_b, U_block, V_block, W_block, i_off, j_off, k_off):
        # i_off, j_off, k_off are offsets for the global indices
        # Local indices: i in [0, n_b), j in [0, m_b), k in [0, p_b)
        # Global indices:
        # a = (i + i_off) * m + (j + j_off)  -- WAIT
        # The global index a is for the input matrix A (n x m).
        # A is 3x3.
        # Block A_b is n_b x m_b.
        # It corresponds to rows i_off..i_off+n_b-1 and cols j_off..j_off+m_b-1 of A.
        # So global row i_g = i + i_off.
        # Global col j_g = j + j_off.
        # Global index a = i_g * m + j_g.
        
        # Global index b is for B (m x p).
        # Block B_b is m_b x p_b.
        # It corresponds to rows j_off..j_off+m_b-1 and cols k_off..k_off+p_b-1 of B.
        # So global row j_g = j + j_off.
        # Global col k_g = k + k_off.
        # Global index b = j_g * p + k_g.
        
        # Global index c is for C (n x p).
        # Block C_b is n_b x p_b.
        # It corresponds to rows i_off..i_off+n_b-1 and cols k_off..k_off+p_b-1 of C.
        # So global row i_g = i + i_off.
        # Global col k_g = k + k_off.
        # Global index c = k_g * n + i_g.  (Trace convention W[k*n+i, r])
        
        for r in range(len(U_block[0])):
            u_col = [0.0] * (n * m)
            v_col = [0.0] * (m * p)
            w_col = [0.0] * (p * n)
            
            for i in range(n_b):
                a = (i + i_off) * m + (j_off) # Wait, U depends on i and j?
                # U_block[i, r] is the coefficient for A[i, j] in this term?
                # No, U_block has shape (n_b * m_b, rank_block).
                # The row index of U_block is i * m_b + j.
                
                # Let's re-structure.
                # U_block is a matrix of size (n_b * m_b) x rank_block.
                # We need to place U_block into the global U matrix.
                pass
        
        # This manual mapping is error-prone.
        # Better to build the full U, V, W matrices of size (n*m) x rank, etc.
        pass

    # Let's use a simpler approach: Build the full matrices directly.
    rank_total = 43
    U = np.zeros((n * m, rank_total))
    V = np.zeros((m * p, rank_total))
    W = np.zeros((p * n, rank_total))
    
    col = 0
    
    # Helper to add a block decomposition
    def add_block(n_b, m_b, p_b, U_b, V_b, W_b, i_off, j_off, k_off):
        nonlocal col
        # U_b shape: (n_b * m_b, r_b)
        # V_b shape: (m_b * p_b, r_b)
        # W_b shape: (p_b * n_b, r_b)
        
        for r in range(U_b.shape[1]):
            # Map U_b
            for i in range(n_b):
                for j in range(m_b):
                    a_local = i * m_b + j
                    a_global = (i + i_off) * m + (j + j_off)
                    U[a_global, col + r] = U_b[a_local, r]
                    
            # Map V_b
            for j in range(m_b):
                for k in range(p_b):
                    b_local = j * p_b + k
                    b_global = (j + j_off) * p + (k + k_off)
                    V[b_global, col + r] = V_b[b_local, r]
                    
            # Map W_b
            for k in range(p_b):
                for i in range(n_b):
                    c_local = k * n_b + i
                    c_global = (k + k_off) * n + (i + i_off)
                    W[c_global, col + r] = W_b[c_local, r]
                    
        col += U_b.shape[1]

    # Block 1: A11 (2x2) * B1 (2x5) -> C11 (2x5)
    # i_off=0, j_off=0, k_off=0
    # n_b=2, m_b=2, p_b=5
    # Decomposition of 2x2x5:
    # Split B1 into B1a (2x2) and B1b (2x3)
    # B1a: 2x2x2 (Strassen, rank 7)
    # B1b: 2x2x3 (Split into 2x2x2 and 2x2x1)
    #   2x2x2 (Strassen, rank 7)
    #   2x2x1 (Trivial, rank 4)
    
    # Strassen 2x2x2
    # M1 = (A11 + A22)(B11 + B22)
    # M2 = (A21 + A22) B11
    # M3 = A11 (B12 - B22)
    # M4 = A22 (B21 - B11)
    # M5 = (A11 + A12) B22
    # M6 = (A21 - A11) (B11 + B12)
    # M7 = (A12 - A22) (B21 + B22)
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    U_222 = np.zeros((4, 7))
    V_222 = np.zeros((4, 7))
    W_222 = np.zeros((4, 7))
    
    # Indices: A: 0,1,2,3 (0:00, 1:01, 2:10, 3:11)
    # B: 0,1,2,3
    # C: 0,1,2,3 (0:00, 1:10, 2:01, 3:11) -- WAIT
    # W convention: W[k*n+i, r].
    # For n=2, k in 0..1, i in 0..1.
    # k=0, i=0 -> 0. (C00)
    # k=1, i=0 -> 1. (C01)
    # k=0, i=1 -> 2. (C10)
    # k=1, i=1 -> 3. (C11)
    
    # M1
    U_222[0, 0] = 1; U_222[3, 0] = 1
    V_222[0, 0] = 1; V_222[3, 0] = 1
    W_222[0, 0] = 1; W_222[3, 0] = 1
    # M2
    U_222[2, 1] = 1; U_222[3, 1] = 1
    V_222[0, 1] = 1
    W_222[2, 1] = 1
    # M3
    U_222[0, 2] = 1
    V_222[1, 2] = 1; V_222[3, 2] = -1
    W_222[1, 2] = 1
    # M4
    U_222[3, 3] = 1
    V_222[2, 3] = 1; V_222[0, 3] = -1
    W_222[0, 3] = 1; W_222[2, 3] = 1
    # M5
    U_222[0, 4] = 1; U_222[1, 4] = 1
    V_222[3, 4] = 1
    W_222[3, 4] = -1
    # M6
    U_222[2, 5] = 1; U_222[0, 5] = -1
    V_222[0, 5] = 1; V_222[1, 5] = 1
    W_222[3, 5] = 1
    # M7
    U_222[1, 6] = 1; U_222[3, 6] = -1
    V_222[2, 6] = 1; V_222[3, 6] = 1
    W_222[1, 6] = 1

    # Add Block 1a: 2x2x2 at (0,0,0)
    add_block(2, 2, 2, U_222, V_222, W_222, 0, 0, 0)
    
    # Block 1b: 2x2x3 at (0,0,2)
    # Split into 2x2x2 at (0,0,2) and 2x2x1 at (0,0,4)
    
    # 2x2x2 at (0,0,2)
    add_block(2, 2, 2, U_222, V_222, W_222, 0, 0, 2)
    
    # 2x2x1 at (0,0,4)
    # Trivial 2x2x1
    U_221 = np.zeros((4, 4))
    V_221 = np.zeros((4, 4))
    W_221 = np.zeros((2, 4)) # p_b=1, n_b=2 -> 1*2=2
    for j in range(2):
        for i in range(2):
            idx = j * 2 + i
            U_221[i * 2 + j, idx] = 1
            V_221[j * 1 + 0, idx] = 1
            W_221[0 * 2 + i, idx] = 1
            
    add_block(2, 2, 1, U_221, V_221, W_221, 0, 0, 4)
    
    # Block 2: A12 (2x1) * B2 (1x5) -> C12 (2x5)
    # i_off=0, j_off=2, k_off=0
    # n_b=2, m_b=1, p_b=5
    # Trivial 2x1x5
    U_215 = np.zeros((2, 10))
    V_215 = np.zeros((5, 10))
    W_215 = np.zeros((10, 10))
    for k in range(5):
        for i in range(2):
            idx = k * 2 + i
            U_215[i * 1 + 0, idx] = 1
            V_215[0 * 5 + k, idx] = 1
            W_215[k * 2 + i, idx] = 1
            
    add_block(2, 1, 5, U_215, V_215, W_215, 0, 2, 0)
    
    # Block 3: A21 (1x2) * B1 (2x5) -> C21 (1x5)
    # i_off=2, j_off=0, k_off=0
    # n_b=1, m_b=2, p_b=5
    # Trivial 1x2x5
    U_125 = np.zeros((2, 10))
    V_125 = np.zeros((10, 10))
    W_125 = np.zeros((5, 10))
    for k in range(5):
        for j in range(2):
            idx = k * 2 + j
            U_125[0 * 2 + j, idx] = 1
            V_125[j * 5 + k, idx] = 1
            W_125[k * 1 + 0, idx] = 1
            
    add_block(1, 2, 5, U_125, V_125, W_125, 2, 0, 0)
    
    # Block 4: A22 (1x1) * B2 (1x5) -> C22 (1x5)
    # i_off=2, j_off=2, k_off=0
    # n_b=1, m_b=1, p_b=5
    # Trivial 1x1x5
    U_115 = np.zeros((1, 5))
    V_115 = np.zeros((5, 5))
    W_115 = np.zeros((5, 5))
    for k in range(5):
        U_115[0, k] = 1
        V_115[k, k] = 1
        W_115[k, k] = 1
        
    add_block(1, 1, 5, U_115, V_115, W_115, 2, 2, 0)
    
    # Check rank
    assert col == 43, f"Rank is {col}, expected 43"
    
    # Convert to lists
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}

