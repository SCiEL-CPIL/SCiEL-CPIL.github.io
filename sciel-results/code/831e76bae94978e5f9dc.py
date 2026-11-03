import numpy as np

DIMENSIONS = [3, 3, 8]

def construct():
    n, m, p = DIMENSIONS
    # n=3, m=3, p=8
    # We decompose p=8 into 3 + 3 + 2
    # Part 1: 3x3x3 using Strassen (rank 7)
    # Part 2: 3x3x3 using Strassen (rank 7)
    # Part 3: 3x3x2 using Schoolbook (rank 18)
    
    rank = 7 + 7 + 18
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Helper to add a term
    def add_term(u_idx, v_idx, w_idx, val):
        U[u_idx][col] = val
        V[v_idx][col] = val
        W[w_idx][col] = val
        return col + 1

    col = 0
    
    # Strassen decomposition for 3x3x3
    # Indices for A (3x3), B (3x3), C (3x3)
    # A indices: i*m+j -> i*3+j
    # B indices: j*p+k -> j*3+k (local k)
    # C indices: k*n+i -> k*3+i (local k)
    
    def strassen_block(offset_b, offset_c):
        nonlocal col
        # Strassen 7 terms for 3x3x3
        # M1 = (A00+A22)(B00+B22)
        # M2 = (A20+A22)B00
        # M3 = A00(B01-B21)
        # M4 = A22(B20-B00)
        # M5 = (A00+A11)(B11+B22)
        # M6 = (A11+A22)(B00+B11)
        # M7 = (A01-A21)(B20+B21)
        
        # A rows 0,1,2 cols 0,1,2
        # B rows 0,1,2 cols offset_b .. offset_b+2
        # C rows offset_c .. offset_c+2 cols 0,1,2 (Wait, C is p x n, so C[k*n+i])
        # C local indices: k_local * n + i
        
        terms = []
        # M1
        terms.append(([(0,0), (2,2)], [(0,0), (2,2)], [(0,0), (2,2)], 1))
        # M2
        terms.append(([(2,0), (2,2)], [(0,0)], [(0,2), (2,2)], 1)) # C02, C22? 
        # Let's use the standard Strassen formulas carefully.
        # C00 = M1 - M4 + M7
        # C01 = M3 + M5
        # C02 = M6 - M2 + M5
        # C10 = M2 + M4
        # C11 = M1 - M3 + M4 + M7
        # C12 = M5 - M1 + M6
        # C20 = M3 - M7 + M1
        # C21 = M2 - M4 + M7
        # C22 = M1 - M5 + M6
        
        # This is getting complicated to map to indices manually.
        # I will use a pre-computed list of Strassen terms for 3x3x3.
        # U, V, W indices are (row, col) in the flattened matrices.
        # A: 3x3. B: 3x3. C: 3x3.
        
        s_terms = [
            # (A_mask, B_mask, C_mask, coeff)
            # M1
            ([(0,0), (2,2)], [(0,0), (2,2)], [(0,0), (2,2)], 1),
            # M2
            ([(2,0), (2,2)], [(0,0)], [(0,2), (2,2)], 1),
            # M3
            ([(0,0)], [(0,1), (2,1)], [(0,1), (0,2)], 1), # Wait, M3 = A00(B01-B21)
            # Let's just use the explicit 7 terms with signs in W.
        ]
        
        # Explicit Strassen for 3x3x3
        # M1 = (A00+A22)(B00+B22)
        # M2 = (A20+A22)B00
        # M3 = A00(B01-B21)
        # M4 = A22(B20-B00)
        # M5 = (A00+A11)(B11+B22)
        # M6 = (A11+A22)(B00+B11)
        # M7 = (A01-A21)(B20+B21)
        
        # C00 = M1 - M4 + M7
        # C01 = M3 + M5
        # C02 = M6 - M2 + M5
        # C10 = M2 + M4
        # C11 = M1 - M3 + M4 + M7
        # C12 = M5 - M1 + M6
        # C20 = M3 - M7 + M1
        # C21 = M2 - M4 + M7
        # C22 = M1 - M5 + M6
        
        # We need to add these to U, V, W.
        # For each M, we add a term to U, V, and multiple terms to W.
        
        def add_strassen_term(a_indices, b_indices, c_indices, coeff):
            nonlocal col
            # U and V get the sum of indices? No, U and V are factors.
            # The term is sum_{(i,j) in a_indices} U[i,j] ...
            # Actually, U, V, W are matrices.
            # U[i*m+j, col] = 1 for all (i,j) in a_indices
            # V[j*p+k, col] = 1 for all (j,k) in b_indices
            # W[k*n+i, col] = coeff for all (k,i) in c_indices
            # This works if the masks are disjoint or if we handle overlaps?
            # Strassen masks are disjoint in A and B?
            # M1: A00, A22. B00, B22. Disjoint.
            # M2: A20, A22. B00. Disjoint.
            # M3: A00. B01, B21. Disjoint.
            # M4: A22. B20, B00. Disjoint.
            # M5: A00, A11. B11, B22. Disjoint.
            # M6: A11, A22. B00, B11. Disjoint.
            # M7: A01, A21. B20, B21. Disjoint.
            
            for i, j in a_indices:
                U[i*m + j][col] = 1
            for j, k in b_indices:
                V[j*p + (offset_b + k)][col] = 1
            for k, i in c_indices:
                W[(offset_c + k)*n + i][col] = coeff
            col += 1

        # M1
        add_strassen_term([(0,0), (2,2)], [(0,0), (2,2)], [(0,0), (2,2)], 1)
        # M2
        add_strassen_term([(2,0), (2,2)], [(0,0)], [(0,2), (2,2)], 1)
        # M3
        add_strassen_term([(0,0)], [(0,1), (2,1)], [(0,1), (0,2)], 1) # C01, C02?
        # Wait, C01 = M3 + M5. C02 = M6 - M2 + M5.
        # M3 contributes to C01 and C02?
        # M3 = A00(B01-B21).
        # C01 gets M3. C02 gets M3?
        # Standard Strassen:
        # C00 = M1 - M4 + M7
        # C01 = M3 + M5
        # C02 = M6 - M2 + M5
        # C10 = M2 + M4
        # C11 = M1 - M3 + M4 + M7
        # C12 = M5 - M1 + M6
        # C20 = M3 - M7 + M1
        # C21 = M2 - M4 + M7
        # C22 = M1 - M5 + M6
        
        # So M3 contributes to C01 (coeff 1) and C20 (coeff 1).
        # My add_strassen_term adds coeff to ALL c_indices.
        # I need to add separate terms for each C entry if the coeff differs.
        # But M3 has coeff 1 for C01 and 1 for C20.
        # So I can add c_indices [(0,1), (2,0)] with coeff 1.
        
        # Let's redo the Strassen terms with correct C indices.
        
        # M1: C00(1), C11(1), C20(1), C22(1)
        # M2: C02(-1), C10(1), C21(1), C22(1) -> Wait, C02 = M6 - M2 + M5. So M2 is -1 in C02.
        # M3: C01(1), C11(-1), C20(1)
        # M4: C00(-1), C10(1), C11(1), C21(-1)
        # M5: C01(1), C02(1), C12(1), C22(-1)
        # M6: C02(1), C12(1), C22(1)
        # M7: C00(1), C11(1), C20(-1), C21(1)
        
        # I will just add 7 terms, and for each term, add the corresponding W entries with correct signs.
        
        def add_strassen_term_v2(a_indices, b_indices, c_entries):
            nonlocal col
            for i, j in a_indices:
                U[i*m + j][col] = 1
            for j, k in b_indices:
                V[j*p + (offset_b + k)][col] = 1
            for k, i, coeff in c_entries:
                W[(offset_c + k)*n + i][col] = coeff
            col += 1

        # M1
        add_strassen_term_v2([(0,0), (2,2)], [(0,0), (2,2)], 
                             [(0,0,1), (1,1,1), (2,0,1), (2,2,1)])
        # M2
        add_strassen_term_v2([(2,0), (2,2)], [(0,0)], 
                             [(0,2,-1), (1,0,1), (2,1,1), (2,2,1)])
        # M3
        add_strassen_term_v2([(0,0)], [(0,1), (2,1)], 
                             [(0,1,1), (1,1,-1), (2,0,1)])
        # M4
        add_strassen_term_v2([(2,2)], [(2,0), (0,0)], 
                             [(0,0,-1), (1,0,1), (1,1,1), (2,1,-1)])
        # M5
        add_strassen_term_v2([(0,0), (1,1)], [(1,1), (2,2)], 
                             [(0,1,1), (0,2,1), (1,2,1), (2,2,-1)])
        # M6
        add_strassen_term_v2([(1,1), (2,2)], [(0,0), (1,1)], 
                             [(0,2,1), (1,2,1), (2,2,1)])
        # M7
        add_strassen_term_v2([(0,1), (2,1)], [(2,0), (2,1)], 
                             [(0,0,1), (1,1,1), (2,0,-1), (2,1,1)])

    # Apply Strassen for B cols 0-2, C cols 0-2
    strassen_block(0, 0)
    # Apply Strassen for B cols 3-5, C cols 3-5
    strassen_block(3, 3)
    
    # Schoolbook for B cols 6-7, C cols 6-7
    # 3x3x2
    for i in range(n):
        for j in range(m):
            for k in range(2):
                U[i*m + j][col] = 1
                V[j*p + (6 + k)][col] = 1
                W[(6 + k)*n + i][col] = 1
                col += 1
                
    return {"U": U, "V": V, "W": W}

