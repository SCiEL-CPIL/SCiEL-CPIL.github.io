import numpy as np

DIMENSIONS = [2, 3, 3]

def construct():
    n, m, p = DIMENSIONS
    # Rank 17 decomposition
    rank = 17
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Part 1: Strassen for 2x2x2 core (i in 0,1; j in 0,1; k in 0,1)
    # Strassen products M1..M7
    # We need to map Strassen terms to U, V, W
    # Strassen computes C = A B for 2x2 matrices.
    # A indices: 00, 01, 10, 11
    # B indices: 00, 01, 10, 11
    # C indices: 00, 01, 10, 11
    
    # M1 = (A00 + A11) * (B00 + B11)
    # M2 = (A10 + A11) * B00
    # M3 = A00 * (B01 - B11)
    # M4 = A11 * (B10 - B00)
    # M5 = (A00 + A01) * B11
    # M6 = (A01 - A10) * (B10 + B11)
    # M7 = (A10 - A00) * (B00 + B01)
    
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # We will assign columns 0..6 to these 7 products.
    # For each product, we define U, V, W contributions.
    
    # Helper to add a rank-1 term
    def add_term(col, u_idx, v_idx, w_idx, coeff=1):
        U[u_idx][col] += coeff
        V[v_idx][col] += coeff
        W[w_idx][col] += coeff
        # Wait, Strassen involves sums/diffs in U and V.
        # The decomposition is linear in A and B.
        # So U and V can have multiple non-zero entries.
        # W has one non-zero entry per term? No, W is for C.
        # The term is sum_r U_r V_r W_r.
        # For a specific C_ik, we sum over r.
        # So for each r, U_r, V_r are vectors, W_r is a vector.
        # In Strassen, each M_k is a product of a linear form in A and a linear form in B.
        # So for each M_k (column r), U_r has non-zeros at A indices, V_r at B indices.
        # W_r has non-zeros at C indices where M_k contributes.
        
    # Let's define the Strassen terms explicitly.
    # A indices in U: i*m + j. m=3.
    # A00 -> 0, A01 -> 1, A10 -> 3, A11 -> 4. (A02, A12 are 5, 6)
    # B indices in V: j*p + k. p=3.
    # B00 -> 0, B01 -> 1, B10 -> 3, B11 -> 4. (B02, B12, B20, B21, B22 are 2, 5, 6, 7, 8)
    # C indices in W: k*n + i. n=2.
    # C00 -> 0, C01 -> 2, C10 -> 1, C11 -> 3. (C02 -> 4, C12 -> 5)
    
    # M1: (A00 + A11) * (B00 + B11)
    # U: A00(0)=1, A11(4)=1
    # V: B00(0)=1, B11(4)=1
    # W: C00(0)=1, C11(3)=1
    add_term(0, 0, 0, 0) # A00, B00, C00
    add_term(0, 0, 4, 0) # A00, B11, C00
    add_term(0, 4, 0, 3) # A11, B00, C11
    add_term(0, 4, 4, 3) # A11, B11, C11
    # This adds 4 rank-1 tensors? No, this is wrong.
    # A rank-1 tensor is a SINGLE product of a vector in U, vector in V, vector in W.
    # Strassen M1 is ONE multiplication.
    # So M1 corresponds to ONE column r.
    # U_r is the vector for (A00+A11).
    # V_r is the vector for (B00+B11).
    # W_r is the vector for the contribution to C.
    # But W_r must be such that sum_r U_r V_r W_r = C.
    # For M1, it contributes to C00 and C11.
    # So W_r should have 1 at C00 and 1 at C11?
    # If W_r = e_C00 + e_C11, then the term is (A00+A11)(B00+B11) * (e_C00 + e_C11).
    # This adds (A00+A11)(B00+B11) to both C00 and C11.
    # This is correct for M1.
    
    # So, for each Strassen product M_k, we create ONE column r.
    # U_r contains the coefficients of A.
    # V_r contains the coefficients of B.
    # W_r contains the coefficients of C.
    
    # Let's redo.
    # Clear U, V, W
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    def set_strassen_term(col, u_coeffs, v_coeffs, w_coeffs):
        # u_coeffs: dict {index: coeff}
        # v_coeffs: dict {index: coeff}
        # w_coeffs: dict {index: coeff}
        for idx, c in u_coeffs.items():
            U[idx][col] = c
        for idx, c in v_coeffs.items():
            V[idx][col] = c
        for idx, c in w_coeffs.items():
            W[idx][col] = c
            
    # M1: (A00 + A11)(B00 + B11) -> C00 + C11
    set_strassen_term(0, {0: 1, 4: 1}, {0: 1, 4: 1}, {0: 1, 3: 1})
    
    # M2: (A10 + A11)B00 -> C10
    set_strassen_term(1, {3: 1, 4: 1}, {0: 1}, {1: 1})
    
    # M3: A00(B01 - B11) -> C01
    set_strassen_term(2, {0: 1}, {1: 1, 4: -1}, {2: 1})
    
    # M4: A11(B10 - B00) -> C00 + C10
    set_strassen_term(3, {4: 1}, {3: 1, 0: -1}, {0: 1, 1: 1})
    
    # M5: (A00 + A01)B11 -> C00 + C01
    set_strassen_term(4, {0: 1, 1: 1}, {4: 1}, {0: -1, 2: 1})
    # Wait, C00 = M1 + M4 - M5 + M7. So M5 coeff for C00 is -1.
    # C01 = M3 + M5. So M5 coeff for C01 is 1.
    
    # M6: (A01 - A10)(B10 + B11) -> C11
    set_strassen_term(5, {1: 1, 3: -1}, {3: 1, 4: 1}, {3: 1})
    
    # M7: (A10 - A00)(B00 + B01) -> C00 + C11
    set_strassen_term(6, {3: 1, 0: -1}, {0: 1, 1: 1}, {0: 1, 3: -1})
    # C00 = ... + M7. Coeff 1.
    # C11 = ... + M7. Coeff -1?
    # C11 = M1 - M2 + M3 + M6. M7 is NOT in C11 formula?
    # Let's check Strassen formulas again.
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    # So M7 only contributes to C00.
    # Correction:
    set_strassen_term(6, {3: 1, 0: -1}, {0: 1, 1: 1}, {0: 1})
    
    # Part 2: Remaining terms (Rank 10)
    # j=2 terms: A_i2 B_2k
    # i in 0,1; k in 0,1,2
    col = 7
    for i in range(2):
        for k in range(3):
            u_idx = i * 3 + 2
            v_idx = 2 * 3 + k
            w_idx = k * 2 + i
            U[u_idx][col] = 1
            V[v_idx][col] = 1
            W[w_idx][col] = 1
            col += 1
            
    # k=2 terms (j in 0,1): A_ij B_j2
    # i in 0,1; j in 0,1
    for i in range(2):
        for j in range(2):
            u_idx = i * 3 + j
            v_idx = j * 3 + 2
            w_idx = 2 * 2 + i
            U[u_idx][col] = 1
            V[v_idx][col] = 1
            W[w_idx][col] = 1
            col += 1
            
    return {"U": U, "V": V, "W": W}
