import numpy as np

def construct():
    n, m, p = 3, 5, 5
    
    # Partitions
    n_parts = [1, 2]
    m_parts = [1, 2, 2]
    p_parts = [1, 2, 2]
    
    # Offsets
    n_offsets = [0]
    for i in range(1, len(n_parts)):
        n_offsets.append(n_offsets[-1] + n_parts[i-1])
    m_offsets = [0]
    for i in range(1, len(m_parts)):
        m_offsets.append(m_offsets[-1] + m_parts[i-1])
    p_offsets = [0]
    for i in range(1, len(p_parts)):
        p_offsets.append(p_offsets[-1] + p_parts[i-1])
        
    U_blocks = []
    V_blocks = []
    W_blocks = []
    
    for i, ni in enumerate(n_parts):
        for j, mj in enumerate(m_parts):
            for k, pk in enumerate(p_parts):
                # Get sub-decomposition for ni x mj x pk
                if ni == 2 and mj == 2 and pk == 2:
                    # Strassen
                    u_sub, v_sub, w_sub = get_strassen_222()
                else:
                    # Schoolbook
                    u_sub, v_sub, w_sub = get_schoolbook(ni, mj, pk)
                
                # We need to embed these into the full U, V, W
                # U row offset: n_offsets[i], col offset: current_rank
                # V row offset: m_offsets[j], col offset: current_rank
                # W row offset: p_offsets[k] * n + n_offsets[i], col offset: current_rank
                
                # For now, just collect blocks. We will assemble later.
                U_blocks.append((i, j, k, u_sub))
                V_blocks.append((i, j, k, v_sub))
                W_blocks.append((i, j, k, w_sub))
                
    # Calculate total rank
    total_rank = sum(u.shape[1] for _, _, _, u in U_blocks)
    
    U = np.zeros((n*m, total_rank))
    V = np.zeros((m*p, total_rank))
    W = np.zeros((p*n, total_rank))
    
    col = 0
    for idx, (i, j, k, u_sub) in enumerate(U_blocks):
        v_sub = V_blocks[idx][3]
        w_sub = W_blocks[idx][3]
        
        # U embedding
        u_rows = range(n_offsets[i], n_offsets[i] + u_sub.shape[0])
        u_cols = range(col, col + u_sub.shape[1])
        U[np.ix_(u_rows, u_cols)] = u_sub
        
        # V embedding
        v_rows = range(m_offsets[j], m_offsets[j] + v_sub.shape[0])
        v_cols = range(col, col + v_sub.shape[1])
        V[np.ix_(v_rows, v_cols)] = v_sub
        
        # W embedding
        # W index is k*n + i
        w_rows = range(p_offsets[k]*n + n_offsets[i], p_offsets[k]*n + n_offsets[i] + w_sub.shape[0])
        w_cols = range(col, col + w_sub.shape[1])
        W[np.ix_(w_rows, w_cols)] = w_sub
        
        col += u_sub.shape[1]
        
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

def get_strassen_222():
    # Strassen 2x2x2 factors
    # U: 4x7, V: 4x7, W: 4x7
    # Indices 0..3 correspond to A00, A01, A10, A11
    U = np.array([
        [1, 0, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 0]
    ])
    # Actually Strassen U, V, W are not identity.
    # Standard Strassen:
    # M1 = (A00 + A11)(B00 + B11)
    # M2 = (A10 + A11)B00
    # M3 = A00(B10 - B11)
    # M4 = A11(B01 - B00)
    # M5 = (A00 + A01)B11
    # M6 = (A10 - A00)(B00 + B01)
    # M7 = (A01 - A11)(B10 + B11)
    
    # U (A factors)
    U = np.array([
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1],
        [1, 0, 1, 0],
        [0, 1, 1, 0],
        [1, 1, 0, 0]
    ])
    # Wait, U is 4x7? No, U is (n*m, rank) = (4, 7).
    # Rows of U correspond to A elements.
    # A00, A01, A10, A11.
    # M1 uses A00, A11.
    # M2 uses A10, A11.
    # M3 uses A00.
    # M4 uses A11.
    # M5 uses A00, A01.
    # M6 uses A10, A00.
    # M7 uses A01, A11.
    
    U = np.array([
        [1, 0, 1, 0, 1, 1, 0], # A00
        [0, 0, 0, 0, 1, 0, 1], # A01
        [0, 1, 0, 0, 0, 1, 0], # A10
        [1, 1, 0, 1, 0, 0, 1]  # A11
    ])
    
    V = np.array([
        [1, 1, 0, 0, 0, 0, 0], # B00
        [0, 0, 1, -1, 0, 1, 0], # B01
        [0, 0, -1, 0, 1, 0, 1], # B10
        [1, 0, 0, 0, 1, 0, 0]  # B11
    ])
    
    # W (C factors)
    # C00 = M1 - M4 + M6
    # C01 = M5 + M7
    # C10 = M2 + M4
    # C11 = M1 - M3 + M5
    
    W = np.array([
        [1, 0, 0, -1, 0, 1, 0], # C00
        [0, 0, 0, 0, 1, 0, 1],  # C01
        [0, 1, 0, 1, 0, 0, 0],  # C10
        [1, 0, -1, 0, 1, 0, 0]  # C11
    ])
    
    return U, V, W

def get_schoolbook(ni, mj, pk):
    rank = ni * mj * pk
    U = np.zeros((ni*mj, rank))
    V = np.zeros((mj*pk, rank))
    W = np.zeros((pk*ni, rank))
    
    col = 0
    for i in range(ni):
        for j in range(mj):
            for k in range(pk):
                U[i*mj + j, col] = 1
                V[j*pk + k, col] = 1
                W[k*ni + i, col] = 1
                col += 1
                
    return U, V, W
