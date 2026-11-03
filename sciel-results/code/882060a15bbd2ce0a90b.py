"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np

DIMENSIONS = [3, 4, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Global dimensions
    N, M, P = n, m, p
    
    # Initialize global matrices
    # U: (N*M, rank)
    # V: (M*P, rank)
    # W: (P*N, rank)
    # We don't know rank yet, so we'll use lists of columns or build dynamically.
    # Better: Estimate max rank or use a list of tuples.
    # Let's use a list of (U_col, V_col, W_col) tuples.
    
    terms = []
    
    def add_term(u_idx, v_idx, w_idx, coeff):
        if coeff == 0:
            return
        terms.append((u_idx, v_idx, w_idx, coeff))
        
    def add_decomp(n_l, m_l, p_l, i_off, j_off, k_off, U_l, V_l, W_l):
        # U_l: (n_l * m_l, r)
        # V_l: (m_l * p_l, r)
        # W_l: (p_l * n_l, r)
        r = U_l.shape[1]
        for c in range(r):
            u_col = U_l[:, c]
            v_col = V_l[:, c]
            w_col = W_l[:, c]
            
            # Map indices
            # u_local index: i * m_l + j
            # u_global index: (i + i_off) * M + (j + j_off)
            
            # v_local index: j * p_l + k
            # v_global index: (j + j_off) * P + (k + k_off)
            
            # w_local index: k * n_l + i
            # w_global index: (k + k_off) * N + (i + i_off)
            
            for u_idx_l in range(n_l * m_l):
                val_u = u_col[u_idx_l]
                if val_u != 0:
                    i_l = u_idx_l // m_l
                    j_l = u_idx_l % m_l
                    u_idx_g = (i_l + i_off) * M + (j_l + j_off)
                    
                    # Find corresponding v and w? No, the term is defined by the column c.
                    # We need to add the full column c to the global structure.
                    # But U, V, W are separate matrices.
                    # We need to construct the global U, V, W columns.
                    
                    # This loop structure is inefficient.
                    # Let's pre-compute the global indices for each local index.
                    
            # Pre-compute mappings
            u_map = np.zeros(n_l * m_l, dtype=int)
            for i_l in range(n_l):
                for j_l in range(m_l):
                    u_map[i_l * m_l + j_l] = (i_l + i_off) * M + (j_l + j_off)
                    
            v_map = np.zeros(m_l * p_l, dtype=int)
            for j_l in range(m_l):
                for k_l in range(p_l):
                    v_map[j_l * p_l + k_l] = (j_l + j_off) * P + (k_l + k_off)
                    
            w_map = np.zeros(p_l * n_l, dtype=int)
            for k_l in range(p_l):
                for i_l in range(n_l):
                    w_map[k_l * n_l + i_l] = (k_l + k_off) * N + (i_l + i_off)
            
            # Now add the column
            # We need to create sparse vectors or just store the non-zeros.
            # Since we are building the final U, V, W, let's just append to lists.
            
            # For this column c:
            # U_global has non-zeros at u_map where U_l[:, c] != 0
            # V_global has non-zeros at v_map where V_l[:, c] != 0
            # W_global has non-zeros at w_map where W_l[:, c] != 0
            
            # We'll store the column as a tuple of (indices, values)
            u_nz = np.nonzero(u_col)[0]
            v_nz = np.nonzero(v_col)[0]
            w_nz = np.nonzero(w_col)[0]
            
            u_idx_g = u_map[u_nz]
            u_vals = u_col[u_nz]
            
            v_idx_g = v_map[v_nz]
            v_vals = v_col[v_nz]
            
            w_idx_g = w_map[w_nz]
            w_vals = w_col[w_nz]
            
            terms.append((u_idx_g, u_vals, v_idx_g, v_vals, w_idx_g, w_vals))
            
    def get_strassen_222():
        # Returns U, V, W for 2x2x2
        # M1 = (A11+A22)(B11+B22)
        # M2 = (A21+A22)B11
        # M3 = A11(B21-B22)
        # M4 = A22(B12-B11)
        # M5 = (A11+A12)B22
        # M6 = (A21-A11)(B11+B12)
        # M7 = (A12-A22)(B21+B22)
        
        # U indices: 00, 01, 10, 11 (row, col)
        # V indices: 00, 01, 10, 11
        # W indices: 00, 01, 10, 11 (k, i) -> k is row of C, i is col of C
        
        U = np.zeros((4, 7))
        V = np.zeros((4, 7))
        W = np.zeros((4, 7))
        
        # M1
        U[0, 0] = 1; U[3, 0] = 1
        V[0, 0] = 1; V[3, 0] = 1
        W[0, 0] = 1; W[1, 0] = 1; W[2, 0] = 1; W[3, 0] = -1 # C11=M1+M4-M5+M7, C12=M3+M5, C21=M2+M4, C22=M1-M2+M3+M6
        # Wait, W is indexed by k*n + i.
        # C is 2x2.
        # C00 (k=0, i=0) -> W[0]
        # C01 (k=0, i=1) -> W[1]
        # C10 (k=1, i=0) -> W[2]
        # C11 (k=1, i=1) -> W[3]
        
        # Let's re-evaluate W coefficients for each M
        # C00 = M1 + M4 - M5 + M7
        # C01 = M3 + M5
        # C10 = M2 + M4
        # C11 = M1 - M2 + M3 + M6
        
        # M1
        U[0, 0] = 1; U[3, 0] = 1
        V[0, 0] = 1; V[3, 0] = 1
        W[0, 0] = 1; W[1, 0] = 0; W[2, 0] = 0; W[3, 0] = 1
        
        # M2
        U[2, 1] = 1; U[3, 1] = 1
        V[0, 1] = 1
        W[0, 1] = 0; W[1, 1] = 0; W[2, 1] = 1; W[3, 1] = -1
        
        # M3
        U[0, 2] = 1
        V[2, 2] = 1; V[3, 2] = -1
        W[0, 2] = 0; W[1, 2] = 1; W[2, 2] = 0; W[3, 2] = 1
        
        # M4
        U[3, 3] = 1
        V[1, 3] = 1; V[0, 3] = -1
        W[0, 3] = 1; W[1, 3] = 0; W[2, 3] = 1; W[3, 3] = 0
        
        # M5
        U[0, 4] = 1; U[1, 4] = 1
        V[3, 4] = 1
        W[0, 4] = -1; W[1, 4] = 1; W[2, 4] = 0; W[3, 4] = 0
        
        # M6
        U[2, 5] = 1; U[0, 5] = -1
        V[0, 5] = 1; V[1, 5] = 1
        W[0, 5] = 0; W[1, 5] = 0; W[2, 5] = 0; W[3, 5] = 1
        
        # M7
        U[1, 6] = 1; U[3, 6] = -1
        V[2, 6] = 1; V[3, 6] = 1
        W[0, 6] = 1; W[1, 6] = 0; W[2, 6] = 0; W[3, 6] = 0
        
        return U, V, W

    def get_schoolbook(n_l, m_l, p_l):
        r = n_l * m_l * p_l
        U = np.zeros((n_l * m_l, r))
        V = np.zeros((m_l * p_l, r))
        W = np.zeros((p_l * n_l, r))
        col = 0
        for i in range(n_l):
            for j in range(m_l):
                for k in range(p_l):
                    U[i * m_l + j, col] = 1
                    V[j * p_l + k, col] = 1
                    W[k * n_l + i, col] = 1
                    col += 1
        return U, V, W

    # Recursive builder
    
    # 3x2x2
    def build_322(i_off, j_off, k_off):
        # Strassen 2x2x2
        U, V, W = get_strassen_222()
        add_decomp(2, 2, 2, i_off, j_off, k_off, U, V, W)
        # Schoolbook 1x2x2
        U, V, W = get_schoolbook(1, 2, 2)
        add_decomp(1, 2, 2, i_off + 2, j_off, k_off, U, V, W)
        
    # 3x2x4
    def build_324(i_off, j_off, k_off):
        build_322(i_off, j_off, k_off)
        build_322(i_off, j_off, k_off + 2)
        
    # 3x2x3
    def build_323(i_off, j_off, k_off):
        build_322(i_off, j_off, k_off)
        U, V, W = get_schoolbook(3, 2, 1)
        add_decomp(3, 2, 1, i_off, j_off, k_off + 2, U, V, W)
        
    # 3x2x7
    def build_327(i_off, j_off, k_off):
        build_324(i_off, j_off, k_off)
        build_323(i_off, j_off, k_off + 4)
        
    # 3x4x7
    build_327(0, 0, 0)
    build_327(0, 2, 0)
    
    # Construct final U, V, W
    rank = len(terms)
    U = np.zeros((N * M, rank))
    V = np.zeros((M * P, rank))
    W = np.zeros((P * N, rank))
    
    for c, (u_idx, u_vals, v_idx, v_vals, w_idx, w_vals) in enumerate(terms):
        U[u_idx, c] = u_vals
        V[v_idx, c] = v_vals
        W[w_idx, c] = w_vals
        
    # Convert to lists as required by interface
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
