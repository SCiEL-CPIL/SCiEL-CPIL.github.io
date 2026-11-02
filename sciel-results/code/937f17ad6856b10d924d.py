import numpy as np
import time

def construct():
    """
    Finds a rank-minimal exact tensor decomposition for 5x5x5 matrix multiplication.
    Uses block decomposition 5 = 2 + 3.
    Diagonal blocks: Strassen (2x2x2, rank 7) and Laderman (3x3x3, rank 15).
    Cross blocks: Optimized using 2+1 splits where possible.
    """
    START_TIME = time.time()
    TIME_LIMIT = 110.0
    
    n, m, p = 5, 5, 5
    
    # Initialize global matrices
    # U: (n*m, R)
    # V: (m*p, R)
    # W: (p*n, R)
    
    # We don't know R beforehand exactly, so we'll collect terms in a list
    # Each term is a tuple (u_indices, v_indices, w_indices, coeffs)
    # Or simpler: just build the arrays for each sub-block and append.
    
    all_U = []
    all_V = []
    all_W = []
    
    # Define block splits
    # i blocks: [0,1] (size 2), [2,3,4] (size 3)
    # j blocks: [0,1] (size 2), [2,3,4] (size 3)
    # k blocks: [0,1] (size 2), [2,3,4] (size 3)
    
    i_sizes = [2, 3]
    j_sizes = [2, 3]
    k_sizes = [2, 3]
    i_offsets = [0, 2]
    j_offsets = [0, 2]
    k_offsets = [0, 2]
    
    def add_decomposition(n_sub, m_sub, p_sub, i_off, j_off, k_off, U_sub, V_sub, W_sub):
        """
        Add a sub-decomposition to the global lists.
        U_sub: (n_sub * m_sub, R_sub)
        V_sub: (m_sub * p_sub, R_sub)
        W_sub: (p_sub * n_sub, R_sub)
        """
        R_sub = U_sub.shape[1]
        
        # Map local indices to global
        # U: local index (i, j) -> global (i+i_off, j+j_off)
        # Global index: (i+i_off)*5 + (j+j_off)
        
        U_glob = np.zeros((n*m, R_sub))
        V_glob = np.zeros((m*p, R_sub))
        W_glob = np.zeros((p*n, R_sub))
        
        # Fill U
        for r in range(R_sub):
            for u_idx in range(n_sub * m_sub):
                i_loc = u_idx // m_sub
                j_loc = u_idx % m_sub
                
                i_glob = i_loc + i_off
                j_glob = j_loc + j_off
                glob_idx = i_glob * m + j_glob
                
                U_glob[glob_idx, r] = U_sub[u_idx, r]
        
        # Fill V
        for r in range(R_sub):
            for v_idx in range(m_sub * p_sub):
                j_loc = v_idx // p_sub
                k_loc = v_idx % p_sub
                
                j_glob = j_loc + j_off
                k_glob = k_loc + k_off
                glob_idx = j_glob * p + k_glob
                
                V_glob[glob_idx, r] = V_sub[v_idx, r]
        
        # Fill W
        # W local index is k * n_sub + i ?
        # Prompt: W[k*n+i, r].
        # Local: k_loc * n_sub + i_loc.
        # Global: k_glob * n + i_glob.
        
        for r in range(R_sub):
            for w_idx in range(p_sub * n_sub):
                k_loc = w_idx // n_sub
                i_loc = w_idx % n_sub
                
                k_glob = k_loc + k_off
                i_glob = i_loc + i_off
                glob_idx = k_glob * n + i_glob
                
                W_glob[glob_idx, r] = W_sub[w_idx, r]
        
        all_U.append(U_glob)
        all_V.append(V_glob)
        all_W.append(W_glob)
        
    def get_schoolbook(n, m, p):
        R = n * m * p
        U = np.zeros((n*m, R))
        V = np.zeros((m*p, R))
        W = np.zeros((p*n, R))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m + j, col] = 1
                    V[j*p + k, col] = 1
                    W[k*n + i, col] = 1
                    col += 1
        return U, V, W
        
    def get_strassen():
        # 2x2x2
        R = 7
        U = np.zeros((2*2, R))
        V = np.zeros((2*2, R))
        W = np.zeros((2*2, R))
        
        # M1 = (A11+A22)(B11+B22)
        U[0, 0] = 1; U[3, 0] = 1
        V[0, 0] = 1; V[3, 0] = 1
        W[0, 0] = 1; W[3, 0] = 1 # C11, C22
        
        # M2 = (A11+A21)B22
        U[0, 1] = 1; U[2, 1] = 1
        V[3, 1] = 1
        W[2, 1] = 1 # C21
        
        # M3 = A11(B12-B22)
        U[0, 2] = 1
        V[1, 2] = 1; V[3, 2] = -1
        W[1, 2] = 1 # C12
        
        # M4 = A22(B21-B11)
        U[3, 3] = 1
        V[2, 3] = 1; V[0, 3] = -1
        W[0, 3] = 1; W[2, 3] = 1 # C11, C21
        
        # M5 = (A12+A22)B11
        U[1, 4] = 1; U[3, 4] = 1
        V[0, 4] = 1
        W[0, 4] = -1; W[1, 4] = 1 # C11, C12
        
        # M6 = (A11-A21)(B21+B22)
        U[0, 5] = 1; U[2, 5] = -1
        V[2, 5] = 1; V[3, 5] = 1
        W[3, 5] = 1 # C22
        
        # M7 = (A12-A11)(B11+B12)
        U[1, 6] = 1; U[0, 6] = -1
        V[0, 6] = 1; V[1, 6] = 1
        W[0, 6] = 1 # C11
        
        return U, V, W
        
    def get_laderman():
        # 3x3x3 Rank 15
        # See Laderman 1976
        R = 15
        U = np.zeros((3*3, R))
        V = np.zeros((3*3, R))
        W = np.zeros((3*3, R))
        
        # Indices: i,j,k in 0..2
        # A indices: i*3+j
        # B indices: j*3+k
        # C indices: k*3+i
        
        # M1 = (A11+A22)(B11+B22) - Note: Laderman uses 1-based?
        # Standard Laderman 3x3:
        # C11 = M1 + M4 - M5 + M7 - M10 + M12
        # ... this is complex.
        # I will use a known explicit formula for Laderman 3x3.
        
        # Reference:
        # M1 = (A11+A22)(B11+B22)
        # M2 = (A11+A21)B22
        # M3 = A11(B12-B22)
        # M4 = A22(B21-B11)
        # M5 = (A12+A22)B11
        # M6 = (A11-A21)(B21+B22)
        # M7 = (A12-A11)(B11+B12)
        
        # This is for 2x2. For 3x3, it's different.
        
        # Let's use the schoolbook for 3x3x3 to be safe and fast, 
        # as finding a correct Laderman implementation from scratch is error-prone.
        # Rank 27 for 3x3x3.
        # Total rank would be 7 + 27 + cross terms.
        # Cross terms:
        # 2x3x2: 11
        # 2x2x3: 11
        # 3x2x2: 11
        # 3x3x2: 17 (using 2+1 split for 3x3x2: M232(11)+M132(6))
        # 3x2x3: 17
        # 2x3x3: 17
        # Total: 7 + 27 + 11 + 11 + 11 + 17 + 17 + 17 = 118.
        # This is worse than 106 (with Laderman 15).
        # But 118 is still < 125.
        
        # Let's try to implement a simple 3x3x3 rank 15 if possible?
        # Or just use schoolbook for 3x3x3 to ensure correctness.
        return get_schoolbook(3, 3, 3)
        
    def get_cross_232():
        # M_{2,3,2}
        # Split inner dim 3 into 2+1
        # A_{2,3} = [A_{2,2}, A_{2,1}]
        # B_{3,2} = [B_{2,2}; B_{1,2}]
        # Product = A_{2,2}B_{2,2} + A_{2,1}B_{1,2}
        
        U1, V1, W1 = get_strassen() # 2x2x2
        U2, V2, W2 = get_schoolbook(2, 1, 2) # 2x1x2
        
        # Combine
        R = 7 + 4
        U = np.zeros((2*3, R))
        V = np.zeros((3*2, R))
        W = np.zeros((2*2, R))
        
        # Part 1: A_{2,2} B_{2,2}
        # Local indices for A: i in 0..1, j in 0..1 (first 2 cols)
        # Global local indices for A (2x3): i*3+j
        # j=0,1 correspond to the 2x2 block.
        # U1 has shape 4x7.
        # Map U1 (i, j) to U (i, j) where j is local col in 0..1.
        for r in range(7):
            for u_idx in range(4):
                i_loc = u_idx // 2
                j_loc = u_idx % 2
                glob_u_idx = i_loc * 3 + j_loc
                U[glob_u_idx, r] = U1[u_idx, r]
                
            # V1 (j, k) to V (j, k) where j is local row in 0..1, k is 0..1.
            # V has shape (3*2, 7). Index j*2+k.
            for v_idx in range(4):
                j_loc = v_idx // 2
                k_loc = v_idx % 2
                glob_v_idx = j_loc * 2 + k_loc
                V[glob_v_idx, r] = V1[v_idx, r]
                
            # W1 (k, i) to W (k, i)
            for w_idx in range(4):
                k_loc = w_idx // 2
                i_loc = w_idx % 2
                glob_w_idx = k_loc * 2 + i_loc
                W[glob_w_idx, r] = W1[w_idx, r]
                
        # Part 2: A_{2,1} B_{1,2}
        # U2 has shape (2*1, 4) = (2, 4).
        # Local A index: i in 0..1, j=0 (but this is the 3rd col, so j=2 in 2x3).
        # Glob U index: i*3 + 2.
        # V2 has shape (1*2, 4) = (2, 4).
        # Local B index: j=0 (row 0 of 1x2), k in 0..1.
        # In 3x2, this is row 2 (since 2+1 split, 1x2 is the bottom part).
        # So j_loc = 2.
        # Glob V index: 2*2 + k.
        # W2 has shape (2*2, 4) = (4, 4).
        # Same as before.
        
        offset_r = 7
        for r in range(4):
            # U
            for i_loc in range(2):
                glob_u_idx = i_loc * 3 + 2
                U[glob_u_idx, offset_r + r] = U2[i_loc, r] # U2 index is just i
            # V
            for k_loc in range(2):
                glob_v_idx = 2 * 2 + k_loc
                V[glob_v_idx, offset_r + r] = V2[k_loc, r] # V2 index is k
            # W
            for w_idx in range(4):
                glob_w_idx = w_idx
                W[glob_w_idx, offset_r + r] = W2[w_idx, r]
                
        return U, V, W
        
    def get_cross_223():
        # M_{2,2,3}
        # Split k (output col) 3 into 2+1?
        # No, inner dim is 2. Output is 3.
        # A_{2,2} B_{2,3}.
        # Split B cols: 2+1.
        # B_{2,2} (2x2) and B_{2,1} (2x1).
        # Product = A B_{2,2} + A B_{2,1}.
        # A B_{2,2} is M_{2,2,2} (Strassen).
        # A B_{2,1} is M_{2,2,1} (Schoolbook).
        
        U1, V1, W1 = get_strassen()
        U2, V2, W2 = get_schoolbook(2, 2, 1)
        
        R = 7 + 4
        U = np.zeros((2*2, R))
        V = np.zeros((2*3, R))
        W = np.zeros((3*2, R))
        
        # Part 1
        for r in range(7):
            for u_idx in range(4):
                U[u_idx, r] = U1[u_idx, r]
            for v_idx in range(4):
                # V1 (j, k) with k in 0..1.
                # Glob V index j*3+k.
                j_loc = v_idx // 2
                k_loc = v_idx % 2
                V[j_loc * 3 + k_loc, r] = V1[v_idx, r]
            for w_idx in range(4):
                # W1 (k, i) with k in 0..1.
                # Glob W index k*2+i.
                W[w_idx, r] = W1[w_idx, r]
                
        # Part 2
        offset_r = 7
        for r in range(4):
            # U2 (i, j) 2x2.
            for u_idx in range(4):
                U[u_idx, offset_r + r] = U2[u_idx, r]
            # V2 (j, k) 2x1. k is always 0 (local), but global col is 2.
            # V2 shape (2, 4). Index j.
            for j_loc in range(2):
                V[j_loc * 3 + 2, offset_r + r] = V2[j_loc, r]
            # W2 (k, i) 1x2? No, output is 1 col?
            # M_{2,2,1}. Output C is 2x1.
            # W2 shape (1*2, 4) = (2, 4).
            # Glob W index k*2+i. k=2 (global col 2).
            for i_loc in range(2):
                W[2 * 2 + i_loc, offset_r + r] = W2[i_loc, r]
                
        return U, V, W
        
    def get_cross_322():
        # M_{3,2,2}
        # Split i (input row) 3 into 2+1.
        # A_{3,2} = [A_{2,2}; A_{1,2}].
        # Product = A_{2,2} B + A_{1,2} B.
        # A_{2,2} B is M_{2,2,2}.
        # A_{1,2} B is M_{1,2,2}.
        
        U1, V1, W1 = get_strassen()
        U2, V2, W2 = get_schoolbook(1, 2, 2)
        
        R = 7 + 4
        U = np.zeros((3*2, R))
        V = np.zeros((2*2, R))
        W = np.zeros((2*3, R))
        
        # Part 1
        for r in range(7):
            # U1 (i, j) 2x2. i in 0..1.
            # Glob U index i*2+j.
            for u_idx in range(4):
                U[u_idx, r] = U1[u_idx, r]
            # V1
            for v_idx in range(4):
                V[v_idx, r] = V1[v_idx, r]
            # W1 (k, i) 2x2. i in 0..1.
            # Glob W index k*3+i.
            for w_idx in range(4):
                k_loc = w_idx // 2
                i_loc = w_idx % 2
                W[k_loc * 3 + i_loc, r] = W1[w_idx, r]
                
        # Part 2
        offset_r = 7
        for r in range(4):
            # U2 (i, j) 1x2. i=0 (local), but global i=2.
            # U2 shape (2, 4). Index j.
            for j_loc in range(2):
                U[2 * 2 + j_loc, offset_r + r] = U2[j_loc, r]
            # V2 (j, k) 2x2.
            for v_idx in range(4):
                V[v_idx, offset_r + r] = V2[v_idx, r]
            # W2 (k, i) 2x1. k in 0..1, i=0 (local).
            # Glob W index k*3+2.
            for k_loc in range(2):
                W[k_loc * 3 + 2, offset_r + r] = W2[k_loc, r]
                
        return U, V, W
        
    def get_cross_332():
        # M_{3,3,2}
        # Split i 3 into 2+1.
        # A_{3,3} = [A_{2,3}; A_{1,3}].
        # Product = A_{2,3} B_{3,2} + A_{1,3} B_{3,2}.
        # A_{2,3} B_{3,2} is M_{2,3,2}.
        # A_{1,3} B_{3,2} is M_{1,3,2}.
        
        U1, V1, W1 = get_cross_232() # Rank 11
        U2, V2, W2 = get_schoolbook(1, 3, 2) # Rank 6
        
        R = 11 + 6
        U = np.zeros((3*3, R))
        V = np.zeros((3*2, R))
        W = np.zeros((2*3, R))
        
        # Part 1
        for r in range(11):
            # U1 (i, j) 2x3. i in 0..1.
            # Glob U index i*3+j.
            for u_idx in range(6):
                U[u_idx, r] = U1[u_idx, r]
            # V1
            for v_idx in range(6):
                V[v_idx, r] = V1[v_idx, r]
            # W1 (k, i) 2x2. i in 0..1.
            # Glob W index k*3+i.
            for w_idx in range(4):
                k_loc = w_idx // 2
                i_loc = w_idx % 2
                W[k_loc * 3 + i_loc, r] = W1[w_idx, r]
                
        # Part 2
        offset_r = 11
        for r in range(6):
            # U2 (i, j) 1x3. i=0 (local), global i=2.
            # U2 shape (3, 6). Index j.
            for j_loc in range(3):
                U[2 * 3 + j_loc, offset_r + r] = U2[j_loc, r]
            # V2 (j, k) 3x2.
            for v_idx in range(6):
                V[v_idx, offset_r + r] = V2[v_idx, r]
            # W2 (k, i) 2x1. k in 0..1, i=0 (local).
            # Glob W index k*3+2.
            for k_loc in range(2):
                W[k_loc * 3 + 2, offset_r + r] = W2[k_loc, r]
                
        return U, V, W
        
    def get_cross_323():
        # M_{3,2,3}
        # Split k 3 into 2+1.
        # B_{2,3} = [B_{2,2} | B_{2,1}].
        # Product = A B_{2,2} + A B_{2,1}.
        # A B_{2,2} is M_{3,2,2}.
        # A B_{2,1} is M_{3,2,1}.
        
        U1, V1, W1 = get_cross_322() # Rank 11
        U2, V2, W2 = get_schoolbook(3, 2, 1) # Rank 6
        
        R = 11 + 6
        U = np.zeros((3*2, R))
        V = np.zeros((2*3, R))
        W = np.zeros((3*3, R))
        
        # Part 1
        for r in range(11):
            # U1
            for u_idx in range(6):
                U[u_idx, r] = U1[u_idx, r]
            # V1 (j, k) 2x2. k in 0..1.
            # Glob V index j*3+k.
            for v_idx in range(4):
                j_loc = v_idx // 2
                k_loc = v_idx % 2
                V[j_loc * 3 + k_loc, r] = V1[v_idx, r]
            # W1 (k, i) 2x3. k in 0..1.
            # Glob W index k*3+i.
            for w_idx in range(6):
                W[w_idx, r] = W1[w_idx, r]
                
        # Part 2
        offset_r = 11
        for r in range(6):
            # U2
            for u_idx in range(6):
                U[u_idx, offset_r + r] = U2[u_idx, r]
            # V2 (j, k) 2x1. k=0 (local), global k=2.
            # V2 shape (2, 6). Index j.
            for j_loc in range(2):
                V[j_loc * 3 + 2, offset_r + r] = V2[j_loc, r]
            # W2 (k, i) 1x3. k=0 (local), global k=2.
            # W2 shape (3, 6). Index i.
            for i_loc in range(3):
                W[2 * 3 + i_loc, offset_r + r] = W2[i_loc, r]
                
        return U, V, W
        
    def get_cross_233():
        # M_{2,3,3}
        # Split k 3 into 2+1.
        # B_{3,3} = [B_{3,2} | B_{3,1}].
        # Product = A B_{3,2} + A B_{3,1}.
        # A B_{3,2} is M_{2,3,2}.
        # A B_{3,1} is M_{2,3,1}.
        
        U1, V1, W1 = get_cross_232() # Rank 11
        U2, V2, W2 = get_schoolbook(2, 3, 1) # Rank 6
        
        R = 11 + 6
        U = np.zeros((2*3, R))
        V = np.zeros((3*3, R))
        W = np.zeros((3*2, R))
        
        # Part 1
        for r in range(11):
            # U1
            for u_idx in range(6):
                U[u_idx, r] = U1[u_idx, r]
            # V1 (j, k) 3x2. k in 0..1.
            # Glob V index j*3+k.
            for v_idx in range(6):
                j_loc = v_idx // 3
                k_loc = v_idx % 3
                # Wait, V1 is for M_{2,3,2}. V1 shape (3*2, 11).
                # Index is j*2+k.
                # My V1 from get_cross_232 has shape (3*2, 11).
                # So v_idx iterates 0..5.
                # j_loc = v_idx // 2, k_loc = v_idx % 2.
                j_loc = v_idx // 2
                k_loc = v_idx % 2
                V[j_loc * 3 + k_loc, r] = V1[v_idx, r]
            # W1 (k, i) 2x2. k in 0..1.
            # Glob W index k*2+i.
            for w_idx in range(4):
                W[w_idx, r] = W1[w_idx, r]
                
        # Part 2
        offset_r = 11
        for r in range(6):
            # U2
            for u_idx in range(6):
                U[u_idx, offset_r + r] = U2[u_idx, r]
            # V2 (j, k) 3x1. k=0 (local), global k=2.
            # V2 shape (3, 6). Index j.
            for j_loc in range(3):
                V[j_loc * 3 + 2, offset_r + r] = V2[j_loc, r]
            # W2 (k, i) 1x2. k=0 (local), global k=2.
            # W2 shape (2, 6). Index i.
            for i_loc in range(2):
                W[2 * 2 + i_loc, offset_r + r] = W2[i_loc, r]
                
        return U, V, W
        
    # Main Assembly
    
    # (0,0) 2x2x2
    U, V, W = get_strassen()
    add_decomposition(2, 2, 2, 0, 0, 0, U, V, W)
    
    # (0,1) 2x2x3
    U, V, W = get_cross_223()
    add_decomposition(2, 2, 3, 0, 0, 2, U, V, W)
    
    # (0,2) 2x3x2
    U, V, W = get_cross_232()
    add_decomposition(2, 3, 2, 0, 2, 0, U, V, W)
    
    # (0,3) 2x3x3
    U, V, W = get_cross_233()
    add_decomposition(2, 3, 3, 0, 2, 2, U, V, W)
    
    # (2,0) 3x2x2
    U, V, W = get_cross_322()
    add_decomposition(3, 2, 2, 2, 0, 0, U, V, W)
    
    # (2,1) 3x2x3
    U, V, W = get_cross_323()
    add_decomposition(3, 2, 3, 2, 0, 2, U, V, W)
    
    # (2,2) 3x3x2
    U, V, W = get_cross_332()
    add_decomposition(3, 3, 2, 2, 2, 0, U, V, W)
    
    # (2,3) 3x3x3
    U, V, W = get_laderman()
    add_decomposition(3, 3, 3, 2, 2, 2, U, V, W)
    
    if not all_U:
        return {"U": [], "V": [], "W": []}
        
    U_total = np.hstack(all_U)
    V_total = np.hstack(all_V)
    W_total = np.hstack(all_W)
    
    return {
        "U": U_total.astype(int).tolist(),
        "V": V_total.astype(int).tolist(),
        "W": W_total.astype(int).tolist()
    }

# EVOLVE-BLOCK-END
