import time
import numpy as np

def construct():
    n, m, p = 3, 4, 4
    nm = n * m  # 12
    mp = m * p  # 16
    pn = p * n  # 12
    
    # Rank 32 decomposition
    # Part 1: i=0 (1x4x4), Rank 4
    # Part 2: i=1,2 (2x4x4), Rank 28
    
    rank = 32
    U = np.zeros((nm, rank))
    V = np.zeros((mp, rank))
    W = np.zeros((pn, rank))
    
    col = 0
    
    # Part 1: i=0
    # C_{0,k} = sum_{j=0}^3 A_{0,j} B_{j,k}
    # Terms r=0..3 correspond to j=r
    for r in range(4):
        # U: a = i*m + j = 0*4 + j = j
        U[r, col] = 1.0  # a=r, r=r
        
        # V: b = j*p + k = r*4 + k
        for k in range(4):
            V[r * 4 + k, col] = 1.0
            
        # W: c = k*n + i = k*3 + 0 = 3k
        for k in range(4):
            W[3 * k, col] = 1.0
        
        col += 1
        
    # Part 2: i=1,2
    # A_2 is 2x4. Rows i=1,2.
    # Split A_2 into A_L (cols 0,1) and A_R (cols 2,3).
    # A_L is 2x2, A_R is 2x2.
    # B is 4x4. Split into B_TL, B_TR, B_BL, B_BR (2x2 each).
    # C_L (cols 0,1) = A_L B_TL + A_R B_BL
    # C_R (cols 2,3) = A_L B_TR + A_R B_BR
    
    # We need to decompose 4 products of 2x2 matrices.
    # Each 2x2x2x2 product has rank 7 (Strassen).
    # Total 4 * 7 = 28 terms.
    
    # Strassen terms for 2x2 matrix multiplication C = A B
    # A = [[a0, a1], [a2, a3]], B = [[b0, b1], [b2, b3]]
    # M1 = (a0+a3)(b0+b3)
    # M2 = (a2+a3)b0
    # M3 = a0(b1-b3)
    # M4 = a3(b2-b0)
    # M5 = (a0+a1)b3
    # M6 = (a2-a0)(b0+b1)
    # M7 = (a1-a3)(b2+b3)
    # C00 = M1 - M4 + M5 - M7
    # C01 = M3 + M7
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # Indices for 2x2 block:
    # Local indices 0,1 for rows/cols within block.
    # Global indices:
    # A_L: i in {1,2}, j in {0,1}. a = 4i + j.
    # A_R: i in {1,2}, j in {2,3}. a = 4i + j.
    # B_TL: j in {0,1}, k in {0,1}. b = 4j + k.
    # B_TR: j in {0,1}, k in {2,3}. b = 4j + k.
    # B_BL: j in {2,3}, k in {0,1}. b = 4j + k.
    # B_BR: j in {2,3}, k in {2,3}. b = 4j + k.
    # C_L: i in {1,2}, k in {0,1}. c = 3k + i.
    # C_R: i in {1,2}, k in {2,3}. c = 3k + i.
    
    def add_strassen_term(A_indices, B_indices, C_indices, sign, col_idx):
        # A_indices: list of (a, coeff)
        # B_indices: list of (b, coeff)
        # C_indices: list of (c, coeff)
        for a, ca in A_indices:
            U[a, col_idx] += ca
        for b, cb in B_indices:
            V[b, col_idx] += cb
        for c, cc in C_indices:
            W[c, col_idx] += cc * sign # Sign applied to W? No, sign is part of the combination.
            # Actually, the term is sum of products.
            # The Strassen formula is a linear combination.
            # We can absorb the sign into U, V, or W.
            # Let's just set the values directly.
            # But we are adding to existing arrays.
            # So we should not multiply by sign here if we want to accumulate.
            # Wait, the Strassen terms are M1..M7.
            # C00 = M1 - M4 + M5 - M7.
            # So we add M1, subtract M4, etc.
            # So we should add the term with the appropriate sign.
            # But U, V, W are additive.
            # So for M1, we add U_M1, V_M1, W_M1.
            # For M4, we add -U_M4, -V_M4, -W_M4.
            # So the sign should be applied to all three.
            pass

    # Let's re-implement add_strassen_term properly
    def add_strassen_block(A_rows, A_cols, B_rows, B_cols, C_rows, C_cols, signs, start_col):
        # A_rows, A_cols: local indices 0,1
        # Maps to global a = 4*i + j
        # i = A_rows[0] + local_i
        # j = A_cols[0] + local_j
        
        # Strassen coefficients for A:
        # M1: a0+a3 -> (0,1), (1,1)
        # M2: a2+a3 -> (1,0), (1,1)
        # M3: a0 -> (0,0)
        # M4: a3 -> (1,1)
        # M5: a0+a1 -> (0,0), (0,1)
        # M6: a2-a0 -> (1,0), (0,0)
        # M7: a1-a3 -> (0,1), (1,1)
        
        A_coeffs = [
            [(0, 1), (1, 1)], # M1
            [(1, 0), (1, 1)], # M2
            [(0, 0)],         # M3
            [(1, 1)],         # M4
            [(0, 0), (0, 1)], # M5
            [(1, 0), (0, -1)],# M6
            [(0, 1), (1, -1)] # M7
        ]
        
        # Strassen coefficients for B:
        # M1: b0+b3 -> (0,0), (1,1)
        # M2: b0 -> (0,0)
        # M3: b1-b3 -> (0,1), (1,1)
        # M4: b2-b0 -> (1,0), (0,0)
        # M5: b3 -> (1,1)
        # M6: b0+b1 -> (0,0), (0,1)
        # M7: b2+b3 -> (1,0), (1,1)
        
        B_coeffs = [
            [(0, 0), (1, 1)], # M1
            [(0, 0)],         # M2
            [(0, 1), (1, -1)],# M3
            [(1, 0), (0, -1)],# M4
            [(1, 1)],         # M5
            [(0, 0), (0, 1)], # M6
            [(1, 0), (1, 1)]  # M7
        ]
        
        # Strassen coefficients for C:
        # C00: M1 - M4 + M5 - M7
        # C01: M3 + M7
        # C10: M2 + M4
        # C11: M1 - M2 + M3 + M6
        
        C_coeffs = [
            [(0, 0, 1), (1, 0, -1), (1, 1, 1), (0, 1, -1)], # M1, M4, M5, M7
            [(0, 1, 1), (1, 1, 1)],                          # M3, M7
            [(1, 0, 1), (1, 1, 1)],                          # M2, M4
            [(0, 0, 1), (1, 0, -1), (0, 1, 1), (1, 1, 1)]   # M1, M2, M3, M6
        ]
        # Wait, the C_coeffs list is indexed by M1..M7?
        # No, it's a mapping from M to C entries.
        # It's easier to iterate M1..M7 and add to C.
        
        for m_idx in range(7):
            col_idx = start_col + m_idx
            sign = 1.0 # We handle signs in C_coeffs? No, signs are in the combination.
            
            # Add A part
            for (li, lj), coeff in A_coeffs[m_idx]:
                i = A_rows[0] + li
                j = A_cols[0] + lj
                a = 4 * i + j
                U[a, col_idx] += coeff
                
            # Add B part
            for (lj, lk), coeff in B_coeffs[m_idx]:
                j = B_rows[0] + lj
                k = B_cols[0] + lk
                b = 4 * j + k
                V[b, col_idx] += coeff
                
            # Add C part
            # We need to know which C entries this M contributes to.
            # M1: C00 (+1), C11 (+1)
            # M2: C10 (+1), C11 (-1)
            # M3: C01 (+1), C11 (+1)
            # M4: C00 (-1), C10 (+1)
            # M5: C00 (+1)
            # M6: C11 (+1)
            # M7: C00 (-1), C01 (+1)
            
            C_contribs = [
                [(0, 0, 1), (1, 1, 1)], # M1
                [(1, 0, 1), (1, 1, -1)],# M2
                [(0, 1, 1), (1, 1, 1)],# M3
                [(0, 0, -1), (1, 0, 1)],# M4
                [(0, 0, 1)],           # M5
                [(1, 1, 1)],           # M6
                [(0, 0, -1), (0, 1, 1)] # M7
            ]
            
            for (li, lk, coeff) in C_contribs[m_idx]:
                i = C_rows[0] + li
                k = C_cols[0] + lk
                c = 3 * k + i
                W[c, col_idx] += coeff
                
    # Product 1: A_L B_TL -> C_L
    # A_L: rows 1,2; cols 0,1
    # B_TL: rows 0,1; cols 0,1
    # C_L: rows 1,2; cols 0,1
    add_strassen_block([1, 3], [0, 2], [0, 2], [0, 2], [1, 3], [0, 2], None, col)
    col += 7
    
    # Product 2: A_R B_BL -> C_L
    # A_R: rows 1,2; cols 2,3
    # B_BL: rows 2,3; cols 0,1
    # C_L: rows 1,2; cols 0,1
    add_strassen_block([1, 3], [2, 4], [2, 4], [0, 2], [1, 3], [0, 2], None, col)
    col += 7
    
    # Product 3: A_L B_TR -> C_R
    # A_L: rows 1,2; cols 0,1
    # B_TR: rows 0,1; cols 2,3
    # C_R: rows 1,2; cols 2,3
    add_strassen_block([1, 3], [0, 2], [0, 2], [2, 4], [1, 3], [2, 4], None, col)
    col += 7
    
    # Product 4: A_R B_BR -> C_R
    # A_R: rows 1,2; cols 2,3
    # B_BR: rows 2,3; cols 2,3
    # C_R: rows 1,2; cols 2,3
    add_strassen_block([1, 3], [2, 4], [2, 4], [2, 4], [1, 3], [2, 4], None, col)
    col += 7
    
    # Verify
    T = np.zeros((nm, mp, pn))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1.0
                
    T_recon = np.einsum('ak,bk,ck->abc', U, V, W, optimize=True)
    if not np.allclose(T_recon, T, atol=1e-6, rtol=0):
        # Fallback to schoolbook if verification fails
        rank = 48
        U = np.zeros((nm, rank))
        V = np.zeros((mp, rank))
        W = np.zeros((pn, rank))
        c_idx = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i * m + j
                    b = j * p + k
                    c = k * n + i
                    U[a, c_idx] = 1.0
                    V[b, c_idx] = 1.0
                    W[c, c_idx] = 1.0
                    c_idx += 1

    return {
        'U': U.tolist(),
        'V': V.tolist(),
        'W': W.tolist()
    }
