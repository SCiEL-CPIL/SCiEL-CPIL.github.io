import time

DIMENSIONS = [2, 5, 6]

def construct():
    n, m, p = DIMENSIONS
    # Rank estimate: 12 (2x1x6) + 42 (2x4x6) = 54
    rank = 54
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Part 1: 2x1x6 (m=0)
    # A is 2x1, B is 1x6
    # C[i,k] = A[i,0] * B[0,k]
    for i in range(2):
        for k in range(6):
            U[i * 5 + 0][col] = 1
            V[0 * 6 + k][col] = 1
            W[k * 2 + i][col] = 1
            col += 1
            
    # Part 2: 2x4x6 (m=1..4)
    # Split into two 2x2x6 blocks
    # Block 1: m in {1, 2}
    # Block 2: m in {3, 4}
    
    def add_2x2x6_block(m_start):
        nonlocal col
        # m_start is 1 or 3
        # A indices: i in {0,1}, m in {m_start, m_start+1}
        # B indices: m in {m_start, m_start+1}, k in {0..5}
        # Split k into 3 blocks of 2
        for k_start in [0, 2, 4]:
            # 2x2x2 block
            # A: i in {0,1}, m in {m_start, m_start+1}
            # B: m in {m_start, m_start+1}, k in {k_start, k_start+1}
            
            # Strassen terms
            # M1 = (A00+A11)(B00+B11)
            # M2 = (A10+A11)B00
            # M3 = A00(B11-B01)  # Note: B indices are (m,k). B01 is B[m_start, k_start+1]
            # M4 = A11(B10-B00)
            # M5 = (A00+A01)B11
            # M6 = (A10-A00)(B00+B01)
            # M7 = (A01-A11)(B10+B11)
            
            # Indices in local 2x2x2:
            # A00: i=0, m=m_start
            # A01: i=0, m=m_start+1
            # A10: i=1, m=m_start
            # A11: i=1, m=m_start+1
            
            # B00: m=m_start, k=k_start
            # B01: m=m_start, k=k_start+1
            # B10: m=m_start+1, k=k_start
            # B11: m=m_start+1, k=k_start+1
            
            # C00: i=0, k=k_start
            # C01: i=0, k=k_start+1
            # C10: i=1, k=k_start
            # C11: i=1, k=k_start+1
            
            # Helper to add a term
            def add_term(a_coeffs, b_coeffs, c_coeffs):
                nonlocal col
                for (i, m_idx, coeff_a) in a_coeffs:
                    if coeff_a != 0:
                        U[i * 5 + m_idx][col] += coeff_a
                for (m_idx, k, coeff_b) in b_coeffs:
                    if coeff_b != 0:
                        V[m_idx * 6 + k][col] += coeff_b
                for (i, k, coeff_c) in c_coeffs:
                    if coeff_c != 0:
                        W[k * 2 + i][col] += coeff_c
                col += 1
            
            # M1
            a_coeffs = [(0, m_start, 1), (1, m_start+1, 1)]
            b_coeffs = [(m_start, k_start, 1), (m_start+1, k_start+1, 1)]
            c_coeffs = [(0, k_start, 1), (0, k_start+1, -1), (1, k_start, 1), (1, k_start+1, -1)]
            add_term(a_coeffs, b_coeffs, c_coeffs)
            
            # M2
            a_coeffs = [(1, m_start, 1), (1, m_start+1, 1)]
            b_coeffs = [(m_start, k_start, 1)]
            c_coeffs = [(1, k_start, 1), (1, k_start+1, 1)]
            add_term(a_coeffs, b_coeffs, c_coeffs)
            
            # M3
            a_coeffs = [(0, m_start, 1)]
            b_coeffs = [(m_start+1, k_start+1, 1), (m_start, k_start+1, -1)]
            c_coeffs = [(0, k_start+1, 1), (1, k_start+1, 1)]
            add_term(a_coeffs, b_coeffs, c_coeffs)
            
            # M4
            a_coeffs = [(1, m_start+1, 1)]
            b_coeffs = [(m_start+1, k_start, 1), (m_start, k_start, -1)]
            c_coeffs = [(0, k_start, -1), (0, k_start+1, 1), (1, k_start, 1), (1, k_start+1, -1)]
            # Wait, C11 = M1 - M2 + M3 + M6. C00 = M1 - M4 + M5 + M7.
            # C01 = M3 + M4. C10 = M2 + M5.
            # Let's re-verify Strassen formulas.
            # C00 = M1 - M4 + M5 + M7
            # C01 = M3 + M4
            # C10 = M2 + M5
            # C11 = M1 - M2 + M3 + M6
            
            # So for M4:
            # C00 gets -M4
            # C01 gets +M4
            c_coeffs = [(0, k_start, -1), (0, k_start+1, 1)]
            add_term(a_coeffs, b_coeffs, c_coeffs)
            
            # M5
            a_coeffs = [(0, m_start, 1), (0, m_start+1, 1)]
            b_coeffs = [(m_start+1, k_start+1, 1)]
            c_coeffs = [(0, k_start, 1), (1, k_start, 1)]
            add_term(a_coeffs, b_coeffs, c_coeffs)
            
            # M6
            a_coeffs = [(1, m_start, 1), (0, m_start, -1)]
            b_coeffs = [(m_start, k_start, 1), (m_start, k_start+1, 1)]
            c_coeffs = [(1, k_start+1, 1)]
            add_term(a_coeffs, b_coeffs, c_coeffs)
            
            # M7
            a_coeffs = [(0, m_start+1, 1), (1, m_start+1, -1)]
            b_coeffs = [(m_start+1, k_start, 1), (m_start+1, k_start+1, 1)]
            c_coeffs = [(0, k_start, 1)]
            add_term(a_coeffs, b_coeffs, c_coeffs)

    add_2x2x6_block(1)
    add_2x2x6_block(3)
    
    return {"U": U, "V": V, "W": W}
