import numpy as np

DIMENSIONS = [3, 4, 6]

def construct():
    n, m, p = DIMENSIONS
    # Total rank estimate: 22 + 22 + 24 = 68
    # We will build it block by block.
    
    U = [[0] * 68 for _ in range(n * m)]
    V = [[0] * 68 for _ in range(m * p)]
    W = [[0] * 68 for _ in range(p * n)]
    
    rank_offset = 0

    # Helper to add 2x2x3 decomposition
    def add_2x2x3(i_off, m_off_A, m_off_B, k_off):
        nonlocal rank_offset
        # 11 terms for 2x2x3
        # A is 2x2, B is 2x3, C is 2x3
        # C = A B
        # Using Strassen for first 2 cols (2x2x2) + 4 terms for last col (2x2x1)
        
        # Strassen 2x2x2
        # M1 = A11(B11+B22) -> Wait, B is 2x3.
        # Let's just use the explicit 11 terms derived from literature or derive.
        # Actually, 7 terms for 2x2x2 part, 4 terms for 2x2x1 part.
        
        # 2x2x2 part (cols 0,1 of B, cols 0,1 of C)
        # B_part is 2x2: B[0:2, k_off:k_off+2]
        # C_part is 2x2: C[i_off:i_off+2, k_off:k_off+2]
        
        # Term 1: A11 * (B11 + B22) ? No, Strassen terms are specific.
        # Let's use the standard 7 terms for A(2x2) * B(2x2)
        # U1 = A11, U2 = A11+A21, U3 = A22, U4 = A22-A12, U5 = A11-A12, U6 = A12+A22, U7 = A21+A22
        # V1 = B11+B22, V2 = B11, V3 = B21+B22, V4 = B11-B21, V5 = B22-B11, V6 = B11+B21, V7 = B12+B22
        # Wait, standard Strassen:
        # M1 = A11(B11+B22)
        # M2 = (A21+A22)B11
        # M3 = A22(B21+B22)
        # M4 = A11(B11-B21)
        # M5 = (A11+A21)(B22-B11)
        # M6 = (A12-A11)(B11+B21)
        # M7 = (A22-A12)(B12+B22)
        
        # C11 = M1 + M4 - M5 + M7
        # C12 = M6 + M5
        # C21 = M2 + M3
        # C22 = M1 - M2 + M3 + M4
        
        # We need to add these to U, V, W.
        # U rows: i_off, i_off+1
        # V rows: m_off_B, m_off_B+1 (for B rows)
        # W rows: k_off, k_off+1 (for C cols)
        
        # Note: B in our problem is 2x3.
        # The 2x2x2 part uses B cols 0,1.
        # The 2x2x1 part uses B col 2.
        
        # Term 1: M1 = A11 * (B11 + B22)
        # U[i_off, r] += 1 (A11)
        # V[m_off_B, r] += 1 (B11)
        # V[m_off_B+1, r] += 1 (B22)
        # W[k_off, r] += 1 (C11 part)
        # W[k_off+1, r] += 1 (C22 part? No, M1 contributes to C11 and C22)
        # C11 += M1, C22 += M1.
        # So W[k_off, r] += 1, W[k_off+1, r] += 1.
        
        # This is getting complex to hardcode.
        # Let's generate the matrices for the sub-problem and add them.
        
        pass

    # This approach is too slow to write correctly in thought.
    # I will use a known decomposition or derive carefully.
    
    # Alternative: Use the 7-term Strassen for 2x2x2 and 4-term for 2x2x1.
    
    # Let's refine the 2x2x3 decomposition.
    # A = [[a11, a12], [a21, a22]]
    # B = [[b11, b12, b13], [b21, b22, b23]]
    # C = [[c11, c12, c13], [c21, c22, c23]]
    
    # 7 terms for C11, C12, C21, C22 (using b11, b12, b21, b22)
    # t1 = a11 * (b11 + b22)
    # t2 = (a21 + a22) * b11
    # t3 = a22 * (b21 + b22)
    # t4 = a11 * (b11 - b21)
    # t5 = (a11 + a21) * (b22 - b11)
    # t6 = (a12 - a11) * (b11 + b21)
    # t7 = (a22 - a12) * (b12 + b22)
    
    # c11 = t1 + t4 - t5 + t7
    # c12 = t6 + t5
    # c21 = t2 + t3
    # c22 = t1 - t2 + t3 + t4
    
    # 4 terms for C13, C23 (using b13, b23)
    # t8 = a11 * b13
    # t9 = a12 * b23
    # t10 = a21 * b13
    # t11 = a22 * b23
    
    # c13 = t8 + t9
    # c23 = t10 + t11
    
    # Total 11 terms.
    
    # Now map to global indices.
    
    def add_2x2x3(i_off, m_off_A, m_off_B, k_off):
        nonlocal rank_offset
        # Term 1
        U[i_off + 0][rank_offset] += 1
        V[m_off_B + 0][rank_offset] += 1
        V[m_off_B + 1][rank_offset] += 1
        W[k_off + 0][rank_offset] += 1
        W[k_off + 1][rank_offset] += 1
        rank_offset += 1
        
        # Term 2
        U[i_off + 1][rank_offset] += 1
        U[i_off + 1][rank_offset] += 1 # Wait, (a21+a22) means U[i_off+1] and U[i_off+1]? No.
        # a21 is row 1 col 0. a22 is row 1 col 1.
        # U index is i*m + j.
        # A is 2x2.
        # a11: i=0, j=0. Global: (i_off)*m + (m_off_A+0)
        # a12: i=0, j=1. Global: (i_off)*m + (m_off_A+1)
        # a21: i=1, j=0. Global: (i_off+1)*m + (m_off_A+0)
        # a22: i=1, j=1. Global: (i_off+1)*m + (m_off_A+1)
        
        # My previous mapping was wrong.
        # U row index is `i_global * m + j_global`.
        
        # Let's restart the mapping logic inside the function.
        pass

    # I will implement this carefully in the final code.
    
    # For 1x2x6:
    # A is 1x2: [a1, a2]
    # B is 2x6: b11..b16, b21..b26
    # C is 1x6: c1..c6
    # cj = a1*b1j + a2*b2j
    # 6 terms? No, 12 terms?
    # t1 = a1*b11, t2 = a2*b21 -> c1 = t1+t2
    # ...
    # 12 terms.
    
    return {"U": U, "V": V, "W": W}
