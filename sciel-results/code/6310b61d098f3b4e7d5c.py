"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 5, 5]

# EVOLVE-BLOCK-START
import time
import sys
import numpy as np

def construct():
    n, m, p = DIMENSIONS
    
    # Split indices
    # n=3 -> [0,1], [2]
    n_splits = [[0, 1], [2]]
    # m=5 -> [0,1], [2,3], [4]
    m_splits = [[0, 1], [2, 3], [4]]
    # p=5 -> [0,1], [2,3], [4]
    p_splits = [[0, 1], [2, 3], [4]]
    
    # Accumulate U, V, W
    # U: (n*m, R), V: (m*p, R), W: (p*n, R)
    # We will build lists of columns first
    
    U_cols = []
    V_cols = []
    W_cols = []
    
    def add_decomp(n_idx, m_idx, p_idx, U_sub, V_sub, W_sub):
        """
        Adds a decomposition for a sub-block.
        n_idx, m_idx, p_idx are lists of global indices.
        U_sub, V_sub, W_sub are matrices of shapes (len(n_idx)*len(m_idx), R_sub), etc.
        """
        R_sub = U_sub.shape[1]
        for r in range(R_sub):
            u_vec = np.zeros(n * m)
            v_vec = np.zeros(m * p)
            w_vec = np.zeros(p * n)
            
            # Fill u_vec
            for i_local, i_global in enumerate(n_idx):
                for j_local, j_global in enumerate(m_idx):
                    u_val = U_sub[i_local * len(m_idx) + j_local, r]
                    u_vec[i_global * m + j_global] = u_val
                    
            # Fill v_vec
            for j_local, j_global in enumerate(m_idx):
                for k_local, k_global in enumerate(p_idx):
                    v_val = V_sub[j_local * len(p_idx) + k_local, r]
                    v_vec[j_global * p + k_global] = v_val
                    
            # Fill w_vec
            for k_local, k_global in enumerate(p_idx):
                for i_local, i_global in enumerate(n_idx):
                    w_val = W_sub[k_local * len(n_idx) + i_local, r]
                    w_vec[k_global * n + i_global] = w_val
                    
            U_cols.append(u_vec)
            V_cols.append(v_vec)
            W_cols.append(w_vec)

    def schoolbook_decomp(n_size, m_size, p_size):
        R = n_size * m_size * p_size
        U = np.zeros((n_size * m_size, R))
        V = np.zeros((m_size * p_size, R))
        W = np.zeros((p_size * n_size, R))
        col = 0
        for i in range(n_size):
            for j in range(m_size):
                for k in range(p_size):
                    U[i * m_size + j, col] = 1
                    V[j * p_size + k, col] = 1
                    W[k * n_size + i, col] = 1
                    col += 1
        return U, V, W

    def strassen_decomp():
        # 2x2x2
        R = 7
        U = np.zeros((4, R))
        V = np.zeros((4, R))
        W = np.zeros((4, R))
        
        # A indices: 00, 01, 10, 11 -> 0, 1, 2, 3
        # B indices: 00, 01, 10, 11 -> 0, 1, 2, 3
        # C indices: 00, 01, 10, 11 -> 0, 1, 2, 3
        
        # M1 = (A00+A11)(B00+B11)
        U[0, 0] = 1; U[3, 0] = 1
        V[0, 0] = 1; V[3, 0] = 1
        W[0, 0] = 1 # C00
        W[3, 0] = 1 # C11
        
        # M2 = (A10+A11)B00
        U[2, 1] = 1; U[3, 1] = 1
        V[0, 1] = 1
        W[2, 1] = 1 # C10
        W[3, 1] = -1 # C11
        
        # M3 = A00(B10-B11)
        U[0, 2] = 1
        V[2, 2] = 1; V[3, 2] = -1
        W[1, 2] = 1 # C01
        W[2, 2] = 1 # C10
        
        # M4 = A11(B01-B00)
        U[3, 3] = 1
        V[1, 3] = 1; V[0, 3] = -1
        W[0, 3] = 1 # C00
        W[2, 3] = 1 # C10
        
        # M5 = (A00+A01)B11
        U[0, 4] = 1; U[1, 4] = 1
        V[3, 4] = 1
        W[0, 4] = -1 # C00
        W[1, 4] = 1 # C01
        
        # M6 = (A10-A00)(B00+B10)
        U[2, 5] = 1; U[0, 5] = -1
        V[0, 5] = 1; V[2, 5] = 1
        W[3, 5] = 1 # C11
        
        # M7 = (A01-A11)(B11+B01)
        U[1, 6] = 1; U[3, 6] = -1
        V[3, 6] = 1; V[1, 6] = 1
        W[0, 6] = 1 # C00
        W[1, 6] = 1 # C01
        
        return U, V, W

    # Iterate over blocks
    for n_idx in n_splits:
        for m_idx in m_splits:
            for p_idx in p_splits:
                n_size = len(n_idx)
                m_size = len(m_idx)
                p_size = len(p_idx)
                
                if n_size == 2 and m_size == 2 and p_size == 2:
                    U_sub, V_sub, W_sub = strassen_decomp()
                else:
                    U_sub, V_sub, W_sub = schoolbook_decomp(n_size, m_size, p_size)
                
                add_decomp(n_idx, m_idx, p_idx, U_sub, V_sub, W_sub)
                
    # Convert to lists of lists as required
    U = np.array(U_cols).T
    V = np.array(V_cols).T
    W = np.array(W_cols).T
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
