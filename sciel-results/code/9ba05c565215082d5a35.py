import numpy as np

DIMENSIONS = [3, 4, 6]

def construct():
    n, m, p = DIMENSIONS
    
    # We will build the decomposition block by block.
    # Partition n=3 into [2, 1]
    # Partition m=4 into [2, 2]
    # Partition p=6 into [2, 2, 2]
    
    n_parts = [2, 1]
    m_parts = [2, 2]
    p_parts = [2, 2, 2]
    
    # Total rank accumulator
    rank = 0
    U = []
    V = []
    W = []
    
    def add_block(i_start, j_start, k_start, n_block, m_block, p_block, local_U, local_V, local_W):
        nonlocal rank, U, V, W
        local_rank = local_U.shape[1]
        
        # Pad local matrices to global dimensions
        # U global shape: (n*m, rank)
        # V global shape: (m*p, rank)
        # W global shape: (p*n, rank)
        
        U_block = np.zeros((n*m, local_rank))
        V_block = np.zeros((m*p, local_rank))
        W_block = np.zeros((p*n, local_rank))
        
        # Fill U_block
        # Indices: i in [i_start, i_start+n_block), j in [j_start, j_start+m_block)
        for li in range(n_block):
            for lj in range(m_block):
                gi = i_start + li
                gj = j_start + lj
                U_block[gi*m + gj, :] = local_U[li*m_block + lj, :]
        
        # Fill V_block
        # Indices: j in [j_start, j_start+m_block), k in [k_start, k_start+p_block)
        for lj in range(m_block):
            for lk in range(p_block):
                gj = j_start + lj
                gk = k_start + lk
                V_block[gj*p + gk, :] = local_V[lj*p_block + lk, :]
        
        # Fill W_block
        # Indices: k in [k_start, k_start+p_block), i in [i_start, i_start+n_block)
        # Note: W index is k*n + i
        for lk in range(p_block):
            for li in range(n_block):
                gk = k_start + lk
                gi = i_start + li
                W_block[gk*n + gi, :] = local_W[lk*n_block + li, :]
        
        U.append(U_block)
        V.append(V_block)
        W.append(W_block)
        rank += local_rank

    # Strassen decomposition for 2x2x2
    def strassen_2x2x2():
        # U, V, W for 2x2x2
        # Standard Strassen
        U = np.array([
            [1, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0]
        ])
        # Wait, U maps A indices to intermediate.
        # A is 2x2. Indices 0,1,2,3.
        # Strassen factors:
        # M1 = (A11+A22)(B11+B22)
        # M2 = (A21+A22)B11
        # M3 = A11(B21-B22)
        # M4 = A22(B12-B11)
        # M5 = (A11+A12)B22
        # M6 = (A21-A11)(B11+B12)
        # M7 = (A12-A22)(B21+B22)
        
        # U rows correspond to A indices (0,1,2,3). Cols to M1..M7.
        U = np.array([
            [1, 0, 1, 0, 1, 0, 0], # A11 (idx 0)
            [0, 1, 0, 0, 0, -1, 0], # A21 (idx 1)
            [0, 0, 0, 1, 0, 0, -1], # A12 (idx 2)
            [1, 1, 0, 1, 0, 0, 0]  # A22 (idx 3)
        ])
        
        # V rows correspond to B indices (0,1,2,3). Cols to M1..M7.
        # B11(0), B21(1), B12(2), B22(3)
        V = np.array([
            [1, 1, 0, -1, 0, 1, 0], # B11
            [0, 0, 1, 0, 0, 0, 1], # B21
            [0, 0, 0, -1, 1, 1, 0], # B12
            [1, 0, -1, 0, 1, 0, 1]  # B22
        ])
        
        # W rows correspond to C indices (0,1,2,3). Cols to M1..M7.
        # C11 = M1 - M4 + M5 + M7
        # C12 = M3 + M5
        # C21 = M2 + M4
        # C22 = M1 - M2 + M3 + M6
        W = np.array([
            [1, 0, 0, -1, 1, 0, 1], # C11
            [0, 0, 1, 0, 1, 0, 0], # C12
            [0, 1, 0, 1, 0, 0, 0], # C21
            [1, -1, 1, 0, 0, 1, 0] # C22
        ])
        
        return U, V, W

    # Direct decomposition for 1x2x2
    # C0 = A0 B00 + A1 B10
    # C1 = A0 B01 + A1 B11
    # Rank 4
    def direct_1x2x2():
        # A indices: 0, 1 (size 2)
        # B indices: 0, 1, 2, 3 (size 2x2 -> 4)
        # C indices: 0, 1 (size 2)
        
        # Terms:
        # 1: A0 B00 -> C0
        # 2: A1 B10 -> C0
        # 3: A0 B01 -> C1
        # 4: A1 B11 -> C1
        
        U = np.array([
            [1, 0, 1, 0], # A0
            [0, 1, 0, 1]  # A1
        ])
        
        V = np.array([
            [1, 0, 0, 0], # B00
            [0, 1, 0, 0], # B10
            [0, 0, 1, 0], # B01
            [0, 0, 0, 1]  # B11
        ])
        
        W = np.array([
            [1, 1, 0, 0], # C0
            [0, 0, 1, 1]  # C1
        ])
        
        return U, V, W

    # Iterate over blocks
    i_offset = 0
    for i_idx, n_block in enumerate(n_parts):
        j_offset = 0
        for j_idx, m_block in enumerate(m_parts):
            k_offset = 0
            for k_idx, p_block in enumerate(p_parts):
                if n_block == 2 and m_block == 2 and p_block == 2:
                    lU, lV, lW = strassen_2x2x2()
                elif n_block == 1 and m_block == 2 and p_block == 2:
                    lU, lV, lW = direct_1x2x2()
                else:
                    # Fallback to schoolbook for this block if sizes don't match
                    lU = np.eye(n_block * m_block)
                    lV = np.eye(m_block * p_block)
                    lW = np.eye(p_block * n_block)
                    # This fallback is incorrect for general sizes, but our partitions are fixed.
                    # We only have 2x2x2 and 1x2x2.
                
                add_block(i_offset, j_offset, k_offset, n_block, m_block, p_block, lU, lV, lW)
                k_offset += p_block
            j_offset += m_block
        i_offset += n_block
    
    U_final = np.hstack(U) if U else np.zeros((n*m, 0))
    V_final = np.hstack(V) if V else np.zeros((m*p, 0))
    W_final = np.hstack(W) if W else np.zeros((p*n, 0))
    
    # Convert to lists for return
    U_list = U_final.tolist()
    V_list = V_final.tolist()
    W_list = W_final.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
