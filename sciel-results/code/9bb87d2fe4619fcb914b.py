import time

DIMENSIONS = [5, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # We will use a block decomposition with blocks of size 2, 2, 1
    # This yields a rank of 117, which is better than 125.
    
    # Helper to add a decomposition
    # U, V, W are lists of lists. We will use a rank accumulator.
    # Since rank is small, we can just append columns.
    U = []
    V = []
    W = []
    
    def add_decomposition(n_sub, m_sub, p_sub, r_A, c_A, r_B, c_B, r_C, c_C):
        """
        Adds the decomposition for a sub-problem of size n_sub x m_sub x p_sub.
        A is at rows r_A..r_A+n_sub-1, cols c_A..c_A+m_sub-1.
        B is at rows r_B..r_B+m_sub-1, cols c_B..c_B+p_sub-1.
        C is at rows r_C..r_C+n_sub-1, cols c_C..c_C+p_sub-1.
        """
        nonlocal U, V, W
        
        if n_sub == 2 and m_sub == 2 and p_sub == 2:
            # Strassen
            # Local indices 0,1 for rows/cols
            # A: 2x2, B: 2x2, C: 2x2
            # We add 7 terms
            for t in range(7):
                u_row = len(U)
                u = [0] * (n_sub * m_sub)
                v = [0] * (m_sub * p_sub)
                w = [0] * (p_sub * n_sub)
                
                if t == 0:
                    # M1 = (A11+A22)(B11+B22)
                    # A11 is (0,0), A22 is (1,1)
                    u[0] = 1; u[3] = 1
                    v[0] = 1; v[3] = 1
                    # C11 += M1, C22 -= M1
                    w[0*2+0] = 1 # C11 col 0? No, C is 2x2.
                    # C index: row r, col c. W index: c*n + r.
                    # Local C: r=0, c=0 -> 0*2+0=0.
                    # r=1, c=1 -> 1*2+1=3.
                    w[0] = 1
                    w[3] = 1
                elif t == 1:
                    # M2 = (A21+A22)B11
                    u[2] = 1; u[3] = 1
                    v[0] = 1
                    # C21 += M2
                    w[0*2+1] = 1 # C21: r=1, c=0 -> 0*2+1=1.
                    w[1] = 1
                elif t == 2:
                    # M3 = A11(B12-B22)
                    u[0] = 1
                    v[1] = 1; v[3] = -1
                    # C12 += M3
                    w[1*2+0] = 1 # C12: r=0, c=1 -> 1*2+0=2.
                    w[2] = 1
                elif t == 3:
                    # M4 = A22(B21-B11)
                    u[3] = 1
                    v[2] = 1; v[0] = -1
                    # C11 += M4, C21 += M4
                    w[0] = 1
                    w[1] = 1
                elif t == 4:
                    # M5 = (A11+A12)B22
                    u[0] = 1; u[1] = 1
                    v[3] = 1
                    # C11 -= M5, C12 += M5
                    w[0] = -1
                    w[2] = 1
                elif t == 5:
                    # M6 = (A21-A11)(B11+B12)
                    u[2] = 1; u[0] = -1
                    v[0] = 1; v[1] = 1
                    # C22 += M6
                    w[3] = 1
                elif t == 6:
                    # M7 = (A12-A22)(B21+B22)
                    u[1] = 1; u[3] = -1
                    v[2] = 1; v[3] = 1
                    # C11 += M7
                    w[0] = 1

                # Map local indices to global
                # U index: (r_A + r) * m_sub + (c_A + c) ? No.
                # U is (n*m, rank). Index is i*m + j.
                # Local i, j. Global i = r_A + i, j = c_A + j.
                # Global U index: (r_A + i) * 5 + (c_A + j).
                # We need to expand u to size 25.
                u_global = [0] * (n * m)
                for i in range(n_sub):
                    for j in range(m_sub):
                        u_global[(r_A + i) * m + (c_A + j)] = u[i * m_sub + j]
                
                v_global = [0] * (m * p)
                for i in range(m_sub):
                    for j in range(p_sub):
                        v_global[(r_B + i) * p + (c_B + j)] = v[i * p_sub + j]
                
                w_global = [0] * (p * n)
                for i in range(p_sub):
                    for j in range(n_sub):
                        # W index: k*n + i. Local k, i.
                        # Global k = c_C + k, i = r_C + i.
                        w_global[(c_C + i) * n + (r_C + j)] = w[i * n_sub + j]

                U.append(u_global)
                V.append(v_global)
                W.append(w_global)
        
        else:
            # Schoolbook
            for i in range(n_sub):
                for j in range(m_sub):
                    for k in range(p_sub):
                        u = [0] * (n * m)
                        v = [0] * (m * p)
                        w = [0] * (p * n)
                        
                        u[(r_A + i) * m + (c_A + j)] = 1
                        v[(r_B + j) * p + (c_B + k)] = 1
                        w[(c_C + k) * n + (r_C + i)] = 1
                        
                        U.append(u)
                        V.append(v)
                        W.append(w)

    # Block definitions
    # Rows: 0,1 (block 0), 2,3 (block 1), 4 (block 2)
    # Cols: 0,1 (block 0), 2,3 (block 1), 4 (block 2)
    row_offsets = [0, 2, 4]
    row_sizes = [2, 2, 1]
    col_offsets = [0, 2, 4]
    col_sizes = [2, 2, 1]

    # Iterate over output blocks i, j and sum over k
    for i in range(3):
        for j in range(3):
            for k in range(3):
                n_sub = row_sizes[i]
                m_sub = col_sizes[k]
                p_sub = col_sizes[j]
                
                r_A = row_offsets[i]
                c_A = col_offsets[k]
                r_B = row_offsets[k]
                c_B = col_offsets[j]
                r_C = row_offsets[i]
                c_C = col_offsets[j]
                
                add_decomposition(n_sub, m_sub, p_sub, r_A, c_A, r_B, c_B, r_C, c_C)

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
