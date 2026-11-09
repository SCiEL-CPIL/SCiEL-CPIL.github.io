import numpy as np

DIMENSIONS = [4, 4, 5]

def construct():
    n, m, p = DIMENSIONS
    # Generate 4x4x4 decomposition
    U4, V4, W4 = strassen_decomp(4)
    rank4 = len(U4[0])
    
    # Initialize full matrices
    rank = rank4 + 16
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Copy 4x4x4 part
    # U4 shape (16, rank4), V4 shape (16, rank4), W4 shape (16, rank4)
    # Note: V4 is for 4x4 B, but our B is 4x5. The first 4 columns of B are used.
    # V index is j*p + k. For k=0..3, index is j*5 + k.
    # V4 index is j*4 + k.
    # So we need to map V4 indices to V indices.
    for i in range(n * m):
        for k in range(rank4):
            U[i][k] = U4[i][k]
            
    for j in range(4):
        for k in range(4):
            v_idx_4 = j * 4 + k
            v_idx_5 = j * 5 + k
            for r in range(rank4):
                V[v_idx_5][r] = V4[v_idx_4][r]
                
    for k in range(4):
        for i in range(4):
            w_idx_4 = k * 4 + i
            w_idx_5 = k * 4 + i # p*n index is k*n+i, same for 4x4
            for r in range(rank4):
                W[w_idx_5][r] = W4[w_idx_4][r]
                
    # Add 5th column terms
    col = rank4
    for i in range(4):
        for j in range(4):
            U[i * 4 + j][col] = 1
            V[j * 5 + 4][col] = 1
            W[4 * 4 + i][col] = 1
            col += 1
            
    return {"U": U, "V": V, "W": W}

def strassen_decomp(n):
    if n == 2:
        # Return 7-term decomposition for 2x2x2
        U = [[0]*7 for _ in range(4)]
        V = [[0]*7 for _ in range(4)]
        W = [[0]*7 for _ in range(4)]
        
        # M1 = (A00+A11)(B00+B11)
        U[0][0] = 1; U[3][0] = 1
        V[0][0] = 1; V[3][0] = 1
        W[0][0] = 1; W[3][0] = 1
        
        # M2 = (A10+A11)B00
        U[2][1] = 1; U[3][1] = 1
        V[0][1] = 1
        W[2][1] = 1; W[3][1] = 1
        
        # M3 = A00(B10-B11)
        U[0][2] = 1
        V[2][2] = 1; V[3][2] = -1
        W[1][2] = 1; W[3][2] = 1
        
        # M4 = A11(B01-B00)
        U[3][3] = 1
        V[1][3] = 1; V[0][3] = -1
        W[0][3] = 1; W[3][3] = 1
        
        # M5 = (A00+A10)B11
        U[0][4] = 1; U[2][4] = 1
        V[3][4] = 1
        W[1][4] = -1; W[2][4] = 1
        
        # M6 = (A10-A00)(B00+B01)
        U[2][5] = 1; U[0][5] = -1
        V[0][5] = 1; V[1][5] = 1
        W[1][5] = 1; W[3][5] = 1
        
        # M7 = (A01-A11)(B10+B11)
        U[1][6] = 1; U[3][6] = -1
        V[2][6] = 1; V[3][6] = 1
        W[0][6] = -1; W[1][6] = 1
        
        return U, V, W
        
    else:
        half = n // 2
        U_sub, V_sub, W_sub = strassen_decomp(half)
        rank_sub = len(U_sub[0])
        total_rank = 7 * rank_sub
        
        U = [[0] * total_rank for _ in range(n * n)]
        V = [[0] * total_rank for _ in range(n * n)]
        W = [[0] * total_rank for _ in range(n * n)]
        
        # Define the 7 combinations
        # A blocks: 00, 01, 10, 11
        # Indices in U (n*n):
        # 00: 0..half-1, 0..half-1 -> idx 0..half^2-1
        # 01: 0..half-1, half..n-1 -> idx half^2..2half^2-1
        # 10: half..n-1, 0..half-1 -> idx 2half^2..3half^2-1
        # 11: half..n-1, half..n-1 -> idx 3half^2..4half^2-1
        
        def get_block_indices(block_id):
            # block_id: 00, 01, 10, 11
            if block_id == 0: # 00
                return range(0, half*half)
            elif block_id == 1: # 01
                return range(half*half, 2*half*half)
            elif block_id == 2: # 10
                return range(2*half*half, 3*half*half)
            else: # 11
                return range(3*half*half, 4*half*half)
        
        # Strassen formulas
        # M1 = (A00+A11)(B00+B11)
        # M2 = (A10+A11)B00
        # M3 = A00(B10-B11)
        # M4 = A11(B01-B00)
        # M5 = (A00+A10)B11
        # M6 = (A10-A00)(B00+B01)
        # M7 = (A01-A11)(B10+B11)
        
        # Output blocks C00, C01, C10, C11
        # C00 = M1 + M4 - M5 - M7
        # C01 = M3 + M7
        # C10 = M2 + M5
        # C11 = M1 - M2 + M3 + M6
        
        # We need to map the sub-decomposition to the global U, V, W.
        # For each of the 7 M's, we have a sub-decomposition of rank rank_sub.
        # The sub-decomposition is for half x half x half.
        # We need to expand it to n x n x n.
        
        for m_idx in range(7):
            # Determine U and V coefficients for this M
            # U coeffs for A blocks
            u_coeffs = {}
            v_coeffs = {}
            
            if m_idx == 0: # M1
                u_coeffs = {0: 1, 3: 1}
                v_coeffs = {0: 1, 3: 1}
            elif m_idx == 1: # M2
                u_coeffs = {2: 1, 3: 1}
                v_coeffs = {0: 1}
            elif m_idx == 2: # M3
                u_coeffs = {0: 1}
                v_coeffs = {2: 1, 3: -1}
            elif m_idx == 3: # M4
                u_coeffs = {3: 1}
                v_coeffs = {1: 1, 0: -1}
            elif m_idx == 4: # M5
                u_coeffs = {0: 1, 2: 1}
                v_coeffs = {3: 1}
            elif m_idx == 5: # M6
                u_coeffs = {2: 1, 0: -1}
                v_coeffs = {0: 1, 1: 1}
            else: # M7
                u_coeffs = {1: 1, 3: -1}
                v_coeffs = {2: 1, 3: 1}
            
            # Output coefficients for W
            w_coeffs = {}
            if m_idx == 0: # M1
                w_coeffs = {0: 1, 3: 1}
            elif m_idx == 1: # M2
                w_coeffs = {2: 1, 3: 1}
            elif m_idx == 2: # M3
                w_coeffs = {1: 1, 3: 1}
            elif m_idx == 3: # M4
                w_coeffs = {0: 1, 3: 1}
            elif m_idx == 4: # M5
                w_coeffs = {1: -1, 2: 1}
            elif m_idx == 5: # M6
                w_coeffs = {1: 1, 3: 1}
            else: # M7
                w_coeffs = {0: -1, 1: 1}
                
            # Expand sub-decomposition
            base_col = m_idx * rank_sub
            for r_sub in range(rank_sub):
                col = base_col + r_sub
                
                # U
                for a_block, coeff in u_coeffs.items():
                    if coeff == 0: continue
                    for idx in get_block_indices(a_block):
                        # Map sub idx to global idx
                        # Sub idx is i_sub * half + j_sub
                        # Global idx is i * n + j
                        # i = (a_block // 2) * half + i_sub
                        # j = (a_block % 2) * half + j_sub
                        i_sub = idx // half
                        j_sub = idx % half
                        i = (a_block // 2) * half + i_sub
                        j = (a_block % 2) * half + j_sub
                        g_idx = i * n + j
                        U[g_idx][col] += coeff * U_sub[idx][r_sub]
                        
                # V
                for b_block, coeff in v_coeffs.items():
                    if coeff == 0: continue
                    for idx in get_block_indices(b_block):
                        i_sub = idx // half
                        j_sub = idx % half
                        i = (b_block // 2) * half + i_sub
                        j = (b_block % 2) * half + j_sub
                        g_idx = i * n + j
                        V[g_idx][col] += coeff * V_sub[idx][r_sub]
                        
                # W
                for c_block, coeff in w_coeffs.items():
                    if coeff == 0: continue
                    for idx in get_block_indices(c_block):
                        i_sub = idx // half
                        j_sub = idx % half
                        i = (c_block // 2) * half + i_sub
                        j = (c_block % 2) * half + j_sub
                        g_idx = i * n + j
                        W[g_idx][col] += coeff * W_sub[idx][r_sub]
                        
        return U, V, W
