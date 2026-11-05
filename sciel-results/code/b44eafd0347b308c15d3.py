import numpy as np

DIMENSIONS = [3, 5, 7]

def construct():
    n, m, p = DIMENSIONS
    # We will build U, V, W as lists of lists or numpy arrays
    # Rank will be determined by the sum of ranks of blocks
    
    # Helper to add a block decomposition
    # block_n, block_m, block_p
    # U_block: (block_n * block_m, rank_block)
    # V_block: (block_m * block_p, rank_block)
    # W_block: (block_p * block_n, rank_block)
    # offsets: i_off, j_off, k_off
    
    U = []
    V = []
    W = []
    
    # We need to accumulate columns
    # Let's use numpy arrays for efficiency, then convert to lists if needed
    # But the return type expects dicts with lists or arrays? 
    # "factor matrices of shapes ... with integer/half-integer entries"
    # The seed program returns lists of lists. I will return lists of lists.
    
    # To make it easier, I'll build numpy arrays and convert at the end.
    
    # Define Strassen 2x2x2
    def get_strassen_222():
        # Returns U, V, W for 2x2x2 multiplication
        # n=2, m=2, p=2
        # 7 terms
        U = np.zeros((4, 7))
        V = np.zeros((4, 7))
        W = np.zeros((4, 7))
        
        # M1 = A00(B00+B11)
        U[0, 0] = 1; V[0, 0] = 1; V[3, 0] = 1
        # M2 = (A00+A10)B11
        U[0, 1] = 1; U[2, 1] = 1; V[3, 1] = 1
        # M3 = (A01+A11)B00
        U[1, 2] = 1; U[3, 2] = 1; V[0, 2] = 1
        # M4 = A11(B10+B00)
        U[3, 3] = 1; V[2, 3] = 1; V[0, 3] = 1
        # M5 = (A00+A01)B10
        U[0, 4] = 1; U[1, 4] = 1; V[2, 4] = 1
        # M6 = (A10-A00)(B00+B01)
        U[2, 5] = 1; U[0, 5] = -1; V[0, 5] = 1; V[1, 5] = 1
        # M7 = (A01-A11)(B10+B11)
        U[1, 6] = 1; U[3, 6] = -1; V[2, 6] = 1; V[3, 6] = 1
        
        # Output reconstruction
        # C00 = M1 - M4 + M6
        W[0, 0] = 1; W[0, 3] = -1; W[0, 5] = 1
        # C01 = M3 + M4
        W[1, 2] = 1; W[1, 3] = 1
        # C10 = M1 + M7
        W[2, 0] = 1; W[2, 6] = 1
        # C11 = M1 - M2 + M5
        W[3, 0] = 1; W[3, 1] = -1; W[3, 4] = 1
        
        return U, V, W

    def get_schoolbook_221():
        # 2x2x1
        # C0 = A00 B00 + A01 B10
        # C1 = A10 B00 + A11 B10
        U = np.zeros((4, 4))
        V = np.zeros((4, 4))
        W = np.zeros((2, 4)) # p*n = 1*2 = 2
        
        # M1 = A00 B00
        U[0, 0] = 1; V[0, 0] = 1
        # M2 = A01 B10
        U[1, 1] = 1; V[2, 1] = 1
        # M3 = A10 B00
        U[2, 2] = 1; V[0, 2] = 1
        # M4 = A11 B10
        U[3, 3] = 1; V[2, 3] = 1
        
        # C0 (k=0, i=0) -> W[0*2+0] = W[0]
        W[0, 0] = 1; W[0, 1] = 1
        # C1 (k=0, i=1) -> W[0*2+1] = W[1]
        W[1, 2] = 1; W[1, 3] = 1
        
        return U, V, W

    def get_schoolbook_127():
        # 1x2x7
        # C0k = A00 B0k + A01 B1k
        # Rank 14
        U = np.zeros((2, 14))
        V = np.zeros((14, 14)) # m*p = 2*7=14
        W = np.zeros((7, 14)) # p*n = 7*1=7
        
        for k in range(7):
            # M_{2k} = A00 B0k
            U[0, 2*k] = 1
            V[0*7+k, 2*k] = 1 # j=0, k=k
            W[k, 2*k] = 1 # k=k, i=0
            
            # M_{2k+1} = A01 B1k
            U[1, 2*k+1] = 1
            V[1*7+k, 2*k+1] = 1 # j=1, k=k
            W[k, 2*k+1] = 1
            
        return U, V, W

    def get_schoolbook_317():
        # 3x1x7
        # C_i k = A_i0 B_0k
        # Rank 21
        U = np.zeros((3, 21))
        V = np.zeros((7, 21)) # m*p = 1*7=7
        W = np.zeros((21, 21)) # p*n = 7*3=21
        
        idx = 0
        for i in range(3):
            for k in range(7):
                U[i, idx] = 1
                V[k, idx] = 1
                W[k*3+i, idx] = 1
                idx += 1
                
        return U, V, W

    # Assemble the full decomposition
    # Blocks:
    # 1. 3x2x7 block at i=0..2, j=0..1, k=0..6
    # 2. 3x2x7 block at i=0..2, j=2..3, k=0..6
    # 3. 3x1x7 block at i=0..2, j=4, k=0..6
    
    # We need to merge these into global U, V, W.
    # Global U size: 3*5 = 15
    # Global V size: 5*7 = 35
    # Global W size: 7*3 = 21
    
    total_rank = 0
    global_U = []
    global_V = []
    global_W = []
    
    def add_block(U_b, V_b, W_b, i_off, j_off, k_off, n_b, m_b, p_b):
        nonlocal total_rank
        rank_b = U_b.shape[1]
        total_rank += rank_b
        
        # Expand U_b to global size
        U_g = np.zeros((15, rank_b))
        V_g = np.zeros((35, rank_b))
        W_g = np.zeros((21, rank_b))
        
        # Map indices
        for r in range(rank_b):
            for u_idx in range(U_b.shape[0]):
                # u_idx = i_local * m_b + j_local
                i_local = u_idx // m_b
                j_local = u_idx % m_b
                i_global = i_off + i_local
                j_global = j_off + j_local
                U_g[i_global * 5 + j_global, r] = U_b[u_idx, r]
                
            for v_idx in range(V_b.shape[0]):
                # v_idx = j_local * p_b + k_local
                j_local = v_idx // p_b
                k_local = v_idx % p_b
                j_global = j_off + j_local
                k_global = k_off + k_local
                V_g[j_global * 7 + k_global, r] = V_b[v_idx, r]
                
            for w_idx in range(W_b.shape[0]):
                # w_idx = k_local * n_b + i_local
                k_local = w_idx // n_b
                i_local = w_idx % n_b
                k_global = k_off + k_local
                i_global = i_off + i_local
                W_g[k_global * 3 + i_global, r] = W_b[w_idx, r]
                
        global_U.append(U_g)
        global_V.append(V_g)
        global_W.append(W_g)

    # Block 1: 3x2x7 (j=0,1)
    # Split 3 rows into 2+1
    # Sub-block 1a: 2x2x7 (i=0,1; j=0,1; k=0..6)
    # Split 7 cols into 2,2,2,1
    # 2x2x2 (Strassen) x3, 2x2x1 (Schoolbook) x1
    
    # We need to build the 2x2x7 decomposition
    # Let's build it piece by piece and add to the global structure
    # Or build a 2x2x7 block and then add it.
    
    # Helper to build 2x2x7 block
    def build_227():
        U_parts = []
        V_parts = []
        W_parts = []
        
        # 3 Strassen blocks for k=0,1; 2,3; 4,5
        for k_start in [0, 2, 4]:
            U_s, V_s, W_s = get_strassen_222()
            # Shift k indices in V and W
            # V is (2*2, 7) -> local V is (2*2, 2)
            # We need to embed this 2x2x2 block into 2x2x7
            # Local V indices: j*2 + k_local
            # Global V indices (for 2x2x7): j*7 + k_global
            V_shift = np.zeros((4, 7))
            W_shift = np.zeros((6, 7)) # p*n = 2*3? No, 2x2x2 W is (2*2, 7)? No.
            # W for 2x2x2 is (2*2, 7)? No, W is (p*n, rank). p=2, n=2. Size 4.
            # My get_strassen_222 returns W of size (4, 7).
            # But for 2x2x7, n=2, p=7. W size is 14.
            
            # Let's just use the add_block logic inside build_227?
            # No, let's construct the 2x2x7 U,V,W directly.
            
        # This is getting complicated. I will stick to the add_block approach for the final assembly.
        # I will create the sub-blocks and add them with correct offsets.
        
        return None # Not used directly

    # Let's just add the sub-blocks directly to the global structure.
    
    # Block 1 (j=0,1):
    # i=0,1; j=0,1; k=0,1 -> Strassen
    U_s, V_s, W_s = get_strassen_222()
    add_block(U_s, V_s, W_s, i_off=0, j_off=0, k_off=0, n_b=2, m_b=2, p_b=2)
    
    # i=0,1; j=0,1; k=2,3 -> Strassen
    add_block(U_s, V_s, W_s, i_off=0, j_off=0, k_off=2, n_b=2, m_b=2, p_b=2)
    
    # i=0,1; j=0,1; k=4,5 -> Strassen
    add_block(U_s, V_s, W_s, i_off=0, j_off=0, k_off=4, n_b=2, m_b=2, p_b=2)
    
    # i=0,1; j=0,1; k=6 -> Schoolbook 2x2x1
    U_sb, V_sb, W_sb = get_schoolbook_221()
    add_block(U_sb, V_sb, W_sb, i_off=0, j_off=0, k_off=6, n_b=2, m_b=2, p_b=1)
    
    # i=2; j=0,1; k=0..6 -> Schoolbook 1x2x7
    U_127, V_127, W_127 = get_schoolbook_127()
    add_block(U_127, V_127, W_127, i_off=2, j_off=0, k_off=0, n_b=1, m_b=2, p_b=7)
    
    # Block 2 (j=2,3):
    # i=0,1; j=2,3; k=0,1 -> Strassen
    add_block(U_s, V_s, W_s, i_off=0, j_off=2, k_off=0, n_b=2, m_b=2, p_b=2)
    
    # i=0,1; j=2,3; k=2,3 -> Strassen
    add_block(U_s, V_s, W_s, i_off=0, j_off=2, k_off=2, n_b=2, m_b=2, p_b=2)
    
    # i=0,1; j=2,3; k=4,5 -> Strassen
    add_block(U_s, V_s, W_s, i_off=0, j_off=2, k_off=4, n_b=2, m_b=2, p_b=2)
    
    # i=0,1; j=2,3; k=6 -> Schoolbook 2x2x1
    add_block(U_sb, V_sb, W_sb, i_off=0, j_off=2, k_off=6, n_b=2, m_b=2, p_b=1)
    
    # i=2; j=2,3; k=0..6 -> Schoolbook 1x2x7
    add_block(U_127, V_127, W_127, i_off=2, j_off=2, k_off=0, n_b=1, m_b=2, p_b=7)
    
    # Block 3 (j=4):
    # i=0..2; j=4; k=0..6 -> Schoolbook 3x1x7
    U_317, V_317, W_317 = get_schoolbook_317()
    add_block(U_317, V_317, W_317, i_off=0, j_off=4, k_off=0, n_b=3, m_b=1, p_b=7)
    
    # Concatenate
    U_final = np.hstack(global_U)
    V_final = np.hstack(global_V)
    W_final = np.hstack(global_W)
    
    # Convert to lists
    return {
        "U": U_final.tolist(),
        "V": V_final.tolist(),
        "W": W_final.tolist()
    }
