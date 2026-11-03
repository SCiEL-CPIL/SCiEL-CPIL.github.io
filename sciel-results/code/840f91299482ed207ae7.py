import numpy as np
import time

def construct():
    n, m, p = 3, 4, 5
    
    # Global dimensions
    dim_U = n * m # 12
    dim_V = m * p # 20
    dim_W = p * n # 15
    
    # We will build the U, V, W matrices by summing contributions from sub-problems.
    # We use a list of (U_sub, V_sub, W_sub, i_offset, j_offset, k_offset, n_sub, m_sub, p_sub)
    # Actually, it's easier to just accumulate into the global matrices.
    
    # We need to know the total rank first to allocate, or we can use lists of columns.
    # Let's use lists of columns (each column is a vector for U, V, W)
    U_cols = []
    V_cols = []
    W_cols = []
    
    def add_decomposition(U_sub, V_sub, W_sub, i_off, j_off, k_off, n_sub, m_sub, p_sub):
        """
        Adds a sub-decomposition to the global lists.
        U_sub: (n_sub*m_sub, R_sub)
        V_sub: (m_sub*p_sub, R_sub)
        W_sub: (p_sub*n_sub, R_sub)
        i_off, j_off, k_off: offsets for the global indices
        n_sub, m_sub, p_sub: dimensions of the sub-problem
        """
        R_sub = U_sub.shape[1]
        for r in range(R_sub):
            u_vec = np.zeros(dim_U)
            v_vec = np.zeros(dim_V)
            w_vec = np.zeros(dim_W)
            
            # Map U indices
            # Local index: i_loc * m_sub + j_loc
            # Global index: (i_loc + i_off) * m + (j_loc + j_off)
            for i_loc in range(n_sub):
                for j_loc in range(m_sub):
                    local_u_idx = i_loc * m_sub + j_loc
                    global_u_idx = (i_loc + i_off) * m + (j_loc + j_off)
                    u_vec[global_u_idx] = U_sub[local_u_idx, r]
            
            # Map V indices
            # Local index: j_loc * p_sub + k_loc
            # Global index: (j_loc + j_off) * p + (k_loc + k_off)
            for j_loc in range(m_sub):
                for k_loc in range(p_sub):
                    local_v_idx = j_loc * p_sub + k_loc
                    global_v_idx = (j_loc + j_off) * p + (k_loc + k_off)
                    v_vec[global_v_idx] = V_sub[local_v_idx, r]
            
            # Map W indices
            # Local index: k_loc * n_sub + i_loc
            # Global index: (k_loc + k_off) * n + (i_loc + i_off)
            for k_loc in range(p_sub):
                for i_loc in range(n_sub):
                    local_w_idx = k_loc * n_sub + i_loc
                    global_w_idx = (k_loc + k_off) * n + (i_loc + i_off)
                    w_vec[global_w_idx] = W_sub[local_w_idx, r]
            
            U_cols.append(u_vec)
            V_cols.append(v_vec)
            W_cols.append(w_vec)

    def get_strassen_222():
        """Returns U, V, W for 2x2x2 Strassen decomposition."""
        # Strassen coefficients
        # M1 = (A11 + A22)(B11 + B22)
        # M2 = (A21 + A22)B11
        # M3 = A11(B21 - B22)
        # M4 = A22(B12 - B11)
        # M5 = (A11 + A12)B22
        # M6 = (A21 - A11)(B11 + B12)
        # M7 = (A12 - A22)(B21 + B22)
        
        # U indices: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
        # V indices: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
        # W indices: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
        
        U = np.zeros((4, 7))
        V = np.zeros((4, 7))
        W = np.zeros((4, 7))
        
        # M1
        U[0, 0] = 1; U[3, 0] = 1
        V[0, 0] = 1; V[3, 0] = 1
        W[0, 0] = 1; W[1, 0] = 1; W[2, 0] = 1; W[3, 0] = 1
        
        # M2
        U[2, 1] = 1; U[3, 1] = 1
        V[0, 1] = 1
        W[2, 1] = 1; W[3, 1] = 1
        
        # M3
        U[0, 2] = 1
        V[2, 2] = 1; V[3, 2] = -1
        W[0, 2] = 1
        
        # M4
        U[3, 3] = 1
        V[1, 3] = 1; V[0, 3] = -1
        W[1, 3] = 1
        
        # M5
        U[0, 4] = 1; U[1, 4] = 1
        V[3, 4] = 1
        W[1, 4] = 1; W[3, 4] = 1
        
        # M6
        U[2, 5] = 1; U[0, 5] = -1
        V[0, 5] = 1; V[1, 5] = 1
        W[2, 5] = 1
        
        # M7
        U[1, 6] = 1; U[3, 6] = -1
        V[2, 6] = 1; V[3, 6] = 1
        W[0, 6] = 1
        
        return U, V, W

    def get_schoolbook(n_sub, m_sub, p_sub):
        """Returns U, V, W for schoolbook decomposition of n_sub x m_sub x p_sub."""
        R = n_sub * m_sub * p_sub
        U = np.zeros((n_sub * m_sub, R))
        V = np.zeros((m_sub * p_sub, R))
        W = np.zeros((p_sub * n_sub, R))
        
        col = 0
        for i in range(n_sub):
            for j in range(m_sub):
                for k in range(p_sub):
                    U[i * m_sub + j, col] = 1.0
                    V[j * p_sub + k, col] = 1.0
                    W[k * n_sub + i, col] = 1.0
                    col += 1
        return U, V, W

    def get_223_decomposition():
        """
        Returns U, V, W for 2x2x3 decomposition.
        Split k into k1 (0,1) and k2 (2).
        Part 1: 2x2x2 (Strassen, rank 7)
        Part 2: 2x2x1 (Schoolbook, rank 4)
        """
        U1, V1, W1 = get_strassen_222()
        U2, V2, W2 = get_schoolbook(2, 2, 1)
        
        # Concatenate
        U = np.hstack([U1, U2])
        V = np.hstack([V1, V2])
        # W needs to be extended for the new k index.
        # W1 is 4x7 (indices for k=0,1)
        # W2 is 2x4 (indices for k=2, mapped to local 2x1 W)
        # The local W index for 2x2x3 is k_loc * 2 + i_loc.
        # For k=0,1, indices are 0,1,2,3.
        # For k=2, indices are 4,5,6,7.
        
        W = np.zeros((8, 11)) # p_sub * n_sub = 3 * 2 = 6? No.
        # Wait, W size is p_sub * n_sub.
        # For 2x2x2: p=2, n=2. Size 4.
        # For 2x2x1: p=1, n=2. Size 2.
        # For 2x2x3: p=3, n=2. Size 6.
        
        # W1 (4x7) corresponds to k_loc in {0,1}.
        # Global W index for k_loc is k_loc * 2 + i_loc.
        # So W1 fills rows 0..3.
        W[0:4, 0:7] = W1
        
        # W2 (2x4) corresponds to k_loc=0 in the sub-sub-problem, which is k_loc=2 in the 2x2x3 problem.
        # Local W index in 2x2x1 is k_loc_sub * 2 + i_loc. k_loc_sub=0.
        # So rows 0,1 of W2 correspond to i_loc=0,1.
        # In 2x2x3, k_loc=2. Global W index is 2 * 2 + i_loc = 4 + i_loc.
        # So rows 4,5 of W should be filled with W2.
        W[4:6, 7:11] = W2
        
        return U, V, W

    # Main construction
    # j in {0,1} and j in {2,3}
    for j_off in [0, 2]:
        # i in {0,1} and i in {2}
        # k in {0,1,2} and k in {3,4}
        
        # Block 1: i in {0,1}, j in {j_off, j_off+1}, k in {0,1,2}
        # Type: 2x2x3
        U_sub, V_sub, W_sub = get_223_decomposition()
        add_decomposition(U_sub, V_sub, W_sub, i_off=0, j_off=j_off, k_off=0, n_sub=2, m_sub=2, p_sub=3)
        
        # Block 2: i in {0,1}, j in {j_off, j_off+1}, k in {3,4}
        # Type: 2x2x2
        U_sub, V_sub, W_sub = get_strassen_222()
        add_decomposition(U_sub, V_sub, W_sub, i_off=0, j_off=j_off, k_off=3, n_sub=2, m_sub=2, p_sub=2)
        
        # Block 3: i in {2}, j in {j_off, j_off+1}, k in {0,1,2}
        # Type: 1x2x3
        U_sub, V_sub, W_sub = get_schoolbook(1, 2, 3)
        add_decomposition(U_sub, V_sub, W_sub, i_off=2, j_off=j_off, k_off=0, n_sub=1, m_sub=2, p_sub=3)
        
        # Block 4: i in {2}, j in {j_off, j_off+1}, k in {3,4}
        # Type: 1x2x2
        U_sub, V_sub, W_sub = get_schoolbook(1, 2, 2)
        add_decomposition(U_sub, V_sub, W_sub, i_off=2, j_off=j_off, k_off=3, n_sub=1, m_sub=2, p_sub=2)

    # Convert lists to arrays
    if U_cols:
        U = np.column_stack(U_cols)
        V = np.column_stack(V_cols)
        W = np.column_stack(W_cols)
    else:
        U = np.zeros((dim_U, 0))
        V = np.zeros((dim_V, 0))
        W = np.zeros((dim_W, 0))
        
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
