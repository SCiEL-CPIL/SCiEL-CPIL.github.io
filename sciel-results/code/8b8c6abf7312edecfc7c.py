import numpy as np

DIMENSIONS = [3, 3, 7]

def construct():
    n, m, p = DIMENSIONS
    
    # We will build the U, V, W matrices by accumulating blocks.
    # Each block has a local rank and local dimensions.
    # We store the blocks as a list of tuples:
    # (n_b, m_b, p_b, i_off, j_off, k_off, U_b, V_b, W_b)
    
    blocks = []
    
    def add_block(n_b, m_b, p_b, i_off, j_off, k_off, U_b, V_b, W_b):
        blocks.append((n_b, m_b, p_b, i_off, j_off, k_off, U_b, V_b, W_b))

    # Helper to create rank 1 block for M_{n,m,1}, M_{1,m,p}, M_{n,1,p}
    # Actually, M_{n,m,1} is rank 1. M_{1,m,p} is rank 1. M_{n,1,p} is rank 1.
    # M_{n,m,p} with any dim 1 is rank 1.
    
    def get_rank1_block(n_b, m_b, p_b):
        U_b = np.ones((n_b * m_b, 1))
        V_b = np.ones((m_b * p_b, 1))
        W_b = np.ones((p_b * n_b, 1))
        return U_b, V_b, W_b

    # Helper for 2x2x2 Strassen block
    def get_strassen_block():
        # n=2, m=2, p=2
        # U: 4x7, V: 4x7, W: 4x7
        # Indices for U: i*2+j
        # Indices for V: j*2+k
        # Indices for W: k*2+i
        
        U_b = np.zeros((4, 7))
        V_b = np.zeros((4, 7))
        W_b = np.zeros((4, 7))
        
        # Term 1
        U_b[0, 0] = 1; U_b[3, 0] = 1
        V_b[0, 0] = 1; V_b[3, 0] = 1
        W_b[0, 0] = 1; W_b[3, 0] = 1
        
        # Term 2
        U_b[2, 1] = 1; U_b[3, 1] = 1
        V_b[0, 1] = 1
        W_b[2, 1] = 1
        
        # Term 3
        U_b[0, 2] = 1
        V_b[1, 2] = 1; V_b[3, 2] = -1
        W_b[1, 2] = 1
        
        # Term 4
        U_b[3, 3] = 1
        V_b[2, 3] = 1; V_b[0, 3] = -1
        W_b[2, 3] = 1
        
        # Term 5
        U_b[0, 4] = 1; U_b[1, 4] = 1
        V_b[3, 4] = 1
        W_b[1, 4] = 1
        
        # Term 6
        U_b[0, 5] = 1; U_b[2, 5] = -1
        V_b[0, 5] = 1; V_b[1, 5] = 1
        W_b[0, 5] = 1
        
        # Term 7
        U_b[1, 6] = 1; U_b[2, 6] = -1
        V_b[2, 6] = 1; V_b[3, 6] = 1
        W_b[0, 6] = 1
        
        return U_b, V_b, W_b

    # Decompose 3x3x7
    # Split n=3 into 2+1
    # Block A: n=2, m=3, p=7, i_off=0
    # Block B: n=1, m=3, p=7, i_off=2
    
    # Block B is rank 1
    U_b, V_b, W_b = get_rank1_block(1, 3, 7)
    add_block(1, 3, 7, 2, 0, 0, U_b, V_b, W_b)
    
    # Decompose Block A (2x3x7)
    # Split m=3 into 2+1
    # Block A1: n=2, m=2, p=7, i_off=0, j_off=0
    # Block A2: n=2, m=1, p=7, i_off=0, j_off=2
    
    # Block A2 is rank 1
    U_b, V_b, W_b = get_rank1_block(2, 1, 7)
    add_block(2, 1, 7, 0, 2, 0, U_b, V_b, W_b)
    
    # Decompose Block A1 (2x2x7)
    # Split p=7 into 2+2+2+1
    # Block A1a: n=2, m=2, p=2, i_off=0, j_off=0, k_off=0 (Strassen)
    # Block A1b: n=2, m=2, p=2, i_off=0, j_off=0, k_off=2 (Strassen)
    # Block A1c: n=2, m=2, p=2, i_off=0, j_off=0, k_off=4 (Strassen)
    # Block A1d: n=2, m=2, p=1, i_off=0, j_off=0, k_off=6 (Rank 1)
    
    U_s, V_s, W_s = get_strassen_block()
    add_block(2, 2, 2, 0, 0, 0, U_s, V_s, W_s)
    
    U_s, V_s, W_s = get_strassen_block()
    add_block(2, 2, 2, 0, 0, 2, U_s, V_s, W_s)
    
    U_s, V_s, W_s = get_strassen_block()
    add_block(2, 2, 2, 0, 0, 4, U_s, V_s, W_s)
    
    U_b, V_b, W_b = get_rank1_block(2, 2, 1)
    add_block(2, 2, 1, 0, 0, 6, U_b, V_b, W_b)
    
    # Now assemble the global matrices
    total_rank = sum(U_b.shape[1] for _, _, _, _, _, _, U_b, _, _ in blocks)
    
    U = np.zeros((n * m, total_rank))
    V = np.zeros((m * p, total_rank))
    W = np.zeros((p * n, total_rank))
    
    col_offset = 0
    for n_b, m_b, p_b, i_off, j_off, k_off, U_b, V_b, W_b in blocks:
        r_b = U_b.shape[1]
        
        # Fill U
        # Global row index: (i_off + i) * m + (j_off + j)
        # Local row index: i * m_b + j
        for i in range(n_b):
            for j in range(m_b):
                g_row = (i_off + i) * m + (j_off + j)
                l_row = i * m_b + j
                U[g_row, col_offset:col_offset + r_b] = U_b[l_row, :]
                
        # Fill V
        # Global row index: (j_off + j) * p + (k_off + k)
        # Local row index: j * p_b + k
        for j in range(m_b):
            for k in range(p_b):
                g_row = (j_off + j) * p + (k_off + k)
                l_row = j * p_b + k
                V[g_row, col_offset:col_offset + r_b] = V_b[l_row, :]
                
        # Fill W
        # Global row index: (k_off + k) * n + (i_off + i)
        # Local row index: k * n_b + i
        for k in range(p_b):
            for i in range(n_b):
                g_row = (k_off + k) * n + (i_off + i)
                l_row = k * n_b + i
                W[g_row, col_offset:col_offset + r_b] = W_b[l_row, :]
                
        col_offset += r_b
        
    # Convert to lists of lists as required
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
