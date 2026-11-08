import numpy as np

DIMENSIONS = [3, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    # Rank calculation
    # A_L (3x2) * B_T (2x8)
    # A_L1 (2x2) * B_T (2x8) -> 4 blocks of 2x2x2 -> 4*7 = 28
    # A_L2 (1x2) * B_T (2x8) -> 1x2x8 -> 16
    # Total A_L B_T = 44
    # Total A_R B_B = 44
    # Total rank = 88
    
    rank = 88
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Helper to add Strassen 2x2x2 factors
    def add_strassen(u_offset, v_offset, w_offset, scale=1):
        # u_offset: (i, j) base index in U
        # v_offset: (j, k) base index in V
        # w_offset: (k, i) base index in W
        
        # Strassen 7 multiplications
        # M1 = (A11+A22)(B11+B22)
        # M2 = (A21+A22)B11
        # M3 = A11(B12-B22)
        # M4 = A22(B21-B11)
        # M5 = (A11+A12)B22
        # M6 = (A21-A11)(B11+B12)
        # M7 = (A12-A22)(B21+B22)
        
        def set_u(i, j, val):
            U[u_offset[0] + i][u_offset[1] + j][col] = val
        def set_v(i, j, val):
            V[v_offset[0] + i][v_offset[1] + j][col] = val
        def set_w(i, j, val):
            # W index is k*n + i
            # w_offset is (k_base, i_base)
            W[w_offset[0] + j][w_offset[1] + i][col] = val # Wait, W shape is (p*n, rank).
            # W[k*n + i, r].
            # So row index is k*n + i.
            # w_offset[0] is k_base, w_offset[1] is i_base.
            # So row = (w_offset[0] + j)*n + (w_offset[1] + i)
            pass

        # We need to be careful with indices.
        # U is (n*m, rank). Index i*m + j.
        # V is (m*p, rank). Index j*p + k.
        # W is (p*n, rank). Index k*n + i.
        
        # Let's rewrite set functions to take absolute indices
        def set_u_abs(u_idx, val):
            U[u_idx][col] = val
        def set_v_abs(v_idx, val):
            V[v_idx][col] = val
        def set_w_abs(w_idx, val):
            W[w_idx][col] = val

        # M1
        set_u_abs(u_offset[0]*m + u_offset[1], scale) # A11
        set_u_abs((u_offset[0]+1)*m + (u_offset[1]+1), scale) # A22
        set_v_abs(v_offset[0]*p + v_offset[1], scale) # B11
        set_v_abs((v_offset[0]+1)*p + (v_offset[1]+1), scale) # B22
        set_w_abs(w_offset[0]*n + w_offset[1], scale) # C11
        set_w_abs((w_offset[0]+1)*n + (w_offset[1]+1), scale) # C22
        # C11 += M1, C22 += M1
        # Wait, Strassen:
        # C11 = M1 + M4 - M5 + M7
        # C12 = M3 + M5
        # C21 = M2 + M4
        # C22 = M1 - M2 + M3 + M6
        
        # We are building the tensor.
        # For each multiplication M_r, we add the contribution to U, V, W.
        # The contribution is U * V * W.
        # So for M1:
        # U has A11 and A22. V has B11 and B22.
        # But Strassen multiplies sums.
        # So U should have 1 for A11 and 1 for A22? No.
        # The term is (A11+A22)(B11+B22).
        # This expands to A11B11 + A11B22 + A22B11 + A22B22.
        # In the tensor decomposition, we just assign the coefficients.
        # U[A11] += 1, U[A22] += 1
        # V[B11] += 1, V[B22] += 1
        # W[C11] += 1, W[C22] += 1
        # This is correct for the term (A11+A22)(B11+B22) contributing to C11 and C22?
        # No, C11 and C22 are different.
        # The term (A11+A22)(B11+B22) is added to BOTH C11 and C22?
        # Yes, in Strassen:
        # C11 = ... + M1 ...
        # C22 = ... + M1 ...
        # So W[C11] += 1 and W[C22] += 1.
        
        # Let's implement a generic add_term(u_indices, v_indices, w_indices, coeff)
        # where u_indices is list of (idx, coeff), etc.
        
        nonlocal col
        
        def add_term(u_terms, v_terms, w_terms):
            for u_idx, u_c in u_terms:
                U[u_idx][col] += u_c
            for v_idx, v_c in v_terms:
                V[v_idx][col] += v_c
            for w_idx, w_c in w_terms:
                W[w_idx][col] += w_c
            col += 1

        # Define indices relative to offsets
        # u_offset: (i0, j0)
        # v_offset: (j0, k0)
        # w_offset: (k0, i0)
        
        i0, j0 = u_offset
        k0 = v_offset[1] # j0 must match
        i0_w = w_offset[1] # i0 must match
        
        # M1
        add_term(
            [(i0*m+j0, 1), ((i0+1)*m+(j0+1), 1)],
            [(j0*p+k0, 1), ((j0+1)*p+(k0+1), 1)],
            [(k0*n+i0, 1), ((k0+1)*n+(i0+1), 1)]
        )
        
        # M2: (A21+A22)B11
        # C21 += M2
        add_term(
            [((i0+1)*m+j0, 1), ((i0+1)*m+(j0+1), 1)],
            [(j0*p+k0, 1)],
            [((k0+1)*n+i0, 1)] # C21 is i0+1, k0? No.
            # C indices:
            # C11: i0, k0
            # C12: i0, k0+1
            # C21: i0+1, k0
            # C22: i0+1, k0+1
            # W index for C21: k0*n + (i0+1)
        )
        
        # M3: A11(B12-B22)
        # C12 += M3
        add_term(
            [(i0*m+j0, 1)],
            [(j0*p+(k0+1), 1), ((j0+1)*p+(k0+1), -1)],
            [(k0+1)*n+i0, 1]
        )
        
        # M4: A22(B21-B11)
        # C11 += M4, C21 += M4
        add_term(
            [((i0+1)*m+(j0+1), 1)],
            [((j0+1)*p+k0, 1), (j0*p+k0, -1)],
            [(k0*n+i0, 1), ((k0+1)*n+(i0+1), 1)]
        )
        
        # M5: (A11+A12)B22
        # C11 -= M5, C12 += M5
        add_term(
            [(i0*m+j0, 1), (i0*m+(j0+1), 1)],
            [((j0+1)*p+(k0+1), 1)],
            [(k0*n+i0, -1), ((k0+1)*n+i0, 1)]
        )
        
        # M6: (A21-A11)(B11+B12)
        # C22 += M6
        add_term(
            [((i0+1)*m+j0, 1), (i0*m+j0, -1)],
            [(j0*p+k0, 1), (j0*p+(k0+1), 1)],
            [((k0+1)*n+(i0+1), 1)]
        )
        
        # M7: (A12-A22)(B21+B22)
        # C11 += M7
        add_term(
            [(i0*m+(j0+1), 1), ((i0+1)*m+(j0+1), -1)],
            [((j0+1)*p+k0, 1), ((j0+1)*p+(k0+1), 1)],
            [(k0*n+i0, 1)]
        )
        
        return col

    # Helper for 1xm x mxp (dot product)
    def add_dot_product(u_offset, v_offset, w_offset, m_size, p_size, scale=1):
        # u_offset: (i0, j0)
        # v_offset: (j0, k0)
        # w_offset: (k0, i0)
        # m_size: size of dot product
        # p_size: number of outputs
        nonlocal col
        for k in range(p_size):
            for j in range(m_size):
                u_idx = u_offset[0]*m + (u_offset[1] + j)
                v_idx = (v_offset[0] + j)*p + (v_offset[1] + k)
                w_idx = (w_offset[0] + k)*n + w_offset[1]
                
                U[u_idx][col] = scale
                V[v_idx][col] = scale
                W[w_idx][col] = scale
                col += 1
        return col

    # 1. A_L B_T
    # A_L is rows 0-2, cols 0-1 of A.
    # B_T is rows 0-1, cols 0-7 of B.
    # C is rows 0-2, cols 0-7.
    
    # A_L1: rows 0-1, cols 0-1.
    # B_T split into 4 blocks of 2x2.
    # B_T1: rows 0-1, cols 0-1.
    # B_T2: rows 0-1, cols 2-3.
    # B_T3: rows 0-1, cols 4-5.
    # B_T4: rows 0-1, cols 6-7.
    
    # M1: A_L1 * B_T1
    col = add_strassen((0, 0), (0, 0), (0, 0))
    # M2: A_L1 * B_T2
    col = add_strassen((0, 0), (0, 2), (2, 0))
    # M3: A_L1 * B_T3
    col = add_strassen((0, 0), (0, 4), (4, 0))
    # M4: A_L1 * B_T4
    col = add_strassen((0, 0), (0, 6), (6, 0))
    
    # A_L2: row 2, cols 0-1.
    # B_T: rows 0-1, cols 0-7.
    # Dot product of size 2, 8 outputs.
    col = add_dot_product((2, 0), (0, 0), (0, 2), 2, 8)
    
    # 2. A_R B_B
    # A_R is rows 0-2, cols 2-3 of A.
    # B_B is rows 2-3, cols 0-7 of B.
    
    # A_R1: rows 0-1, cols 2-3.
    # B_B split into 4 blocks of 2x2.
    # B_B1: rows 2-3, cols 0-1.
    # B_B2: rows 2-3, cols 2-3.
    # B_B3: rows 2-3, cols 4-5.
    # B_B4: rows 2-3, cols 6-7.
    
    # M5: A_R1 * B_B1
    col = add_strassen((0, 2), (2, 0), (0, 0))
    # M6: A_R1 * B_B2
    col = add_strassen((0, 2), (2, 2), (2, 0))
    # M7: A_R1 * B_B3
    col = add_strassen((0, 2), (2, 4), (4, 0))
    # M8: A_R1 * B_B4
    col = add_strassen((0, 2), (2, 6), (6, 0))
    
    # A_R2: row 2, cols 2-3.
    # B_B: rows 2-3, cols 0-7.
    col = add_dot_product((2, 2), (2, 0), (0, 2), 2, 8)
    
    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
