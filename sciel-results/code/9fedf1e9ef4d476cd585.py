import numpy as np

DIMENSIONS = [6, 6, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # 6x6 matrix multiplication
    # Using 3x3 block decomposition with 2x2 Strassen for each block multiplication
    # Total rank: 9 blocks * 3 terms per block * 7 Strassen rank = 189
    
    rank = 189
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Strassen 2x2 decomposition
    # A = [[a, b], [c, d]], B = [[e, f], [g, h]]
    # M1 = (a+d)(e+h)
    # M2 = (c+d)e
    # M3 = a(f-h)
    # M4 = d(g-e)
    # M5 = (a+b)h
    # M6 = (c-a)(e+f)
    # M7 = (b-d)(g+h)
    
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    # Indices for 2x2 block:
    # A: (0,0)->a, (0,1)->b, (1,0)->c, (1,1)->d
    # B: (0,0)->e, (0,1)->f, (1,0)->g, (1,1)->h
    # C: (0,0)->C11, (0,1)->C12, (1,0)->C21, (1,1)->C22
    
    strassen_terms = [
        # (U_coeffs, V_coeffs, W_coeffs)
        # U_coeffs: dict {index: coeff}, V_coeffs: dict {index: coeff}, W_coeffs: dict {index: coeff}
        # Indices 0..3 for 2x2
        # M1
        ({0: 1, 3: 1}, {0: 1, 3: 1}, {0: 1, 3: 1}),
        # M2
        ({2: 1, 3: 1}, {0: 1}, {2: 1, 3: 1}), # Wait, W for M2 is C21 and C22?
        # Let's re-verify Strassen output mapping.
        # C11 = M1 + M4 - M5 + M7
        # C12 = M3 + M5
        # C21 = M2 + M4
        # C22 = M1 - M2 + M3 + M6
        
        # So W coefficients for each M:
        # M1: C11(+1), C22(+1)
        # M2: C21(+1), C22(-1)
        # M3: C12(+1), C22(+1)
        # M4: C11(+1), C21(+1)
        # M5: C11(-1), C12(+1)
        # M6: C22(+1)
        # M7: C11(+1)
    ]
    
    # Corrected Strassen terms
    terms = [
        # M1: (a+d)(e+h)
        ({0: 1, 3: 1}, {0: 1, 3: 1}, {0: 1, 3: 1}),
        # M2: (c+d)e
        ({2: 1, 3: 1}, {0: 1}, {2: 1, 3: -1}),
        # M3: a(f-h)
        ({0: 1}, {1: 1, 3: -1}, {1: 1, 3: 1}),
        # M4: d(g-e)
        ({3: 1}, {2: 1, 0: -1}, {0: 1, 2: 1}),
        # M5: (a+b)h
        ({0: 1, 1: 1}, {3: 1}, {0: -1, 1: 1}),
        # M6: (c-a)(e+f)
        ({2: 1, 0: -1}, {0: 1, 1: 1}, {3: 1}),
        # M7: (b-d)(g+h)
        ({1: 1, 3: -1}, {2: 1, 3: 1}, {0: 1}),
    ]
    
    # Iterate over 3x3 block grid
    for i in range(3):
        for j in range(3):
            for k in range(3):
                # A block (i, k)
                # B block (k, j)
                # C block (i, j)
                
                # Local indices in 2x2
                # A: (0,0) -> global (2i, 2k)
                # A: (0,1) -> global (2i, 2k+1)
                # A: (1,0) -> global (2i+1, 2k)
                # A: (1,1) -> global (2i+1, 2k+1)
                
                # B: (0,0) -> global (2k, 2j)
                # B: (0,1) -> global (2k, 2j+1)
                # B: (1,0) -> global (2k+1, 2j)
                # B: (1,1) -> global (2k+1, 2j+1)
                
                # C: (0,0) -> global (2i, 2j)
                # C: (0,1) -> global (2i, 2j+1)
                # C: (1,0) -> global (2i+1, 2j)
                # C: (1,1) -> global (2i+1, 2j+1)
                
                for u_coeffs, v_coeffs, w_coeffs in terms:
                    for local_idx, coeff in u_coeffs.items():
                        r, c = divmod(local_idx, 2)
                        global_r = 2 * i + r
                        global_c = 2 * k + c
                        U[global_r * m + global_c][col] += coeff
                    
                    for local_idx, coeff in v_coeffs.items():
                        r, c = divmod(local_idx, 2)
                        global_r = 2 * k + r
                        global_c = 2 * j + c
                        V[global_r * p + global_c][col] += coeff
                    
                    for local_idx, coeff in w_coeffs.items():
                        r, c = divmod(local_idx, 2)
                        global_r = 2 * i + r
                        global_c = 2 * j + c
                        # W convention: W[k*n+i, r]
                        # Here C is p x n (6x6).
                        # C index (r, c) corresponds to W[c*n + r]?
                        # Task says: W[k*n+i, r]
                        # Usually C[i, j] = sum_k A[i, k] B[k, j]
                        # Tensor T_{ijk} = delta_{ik} delta_{kj}? No.
                        # Standard: C_{ij} = sum_k A_{ik} B_{kj}.
                        # U_{i, k, t}, V_{k, j, t}, W_{j, i, t}?
                        # Task: U (n*m, rank), V (m*p, rank), W (p*n, rank).
                        # n=6, m=6, p=6.
                        # U index: i*m + k? No, U is n*m.
                        # Usually U_{i, k} for A_{i, k}.
                        # V_{k, j} for B_{k, j}.
                        # W_{j, i} for C_{i, j}.
                        # Task says W[k*n+i, r].
                        # If C is p x n (6x6), and standard C_{i, j} with i in 0..5, j in 0..5.
                        # Then W index is j*n + i.
                        # My global_r is i (row of C), global_c is j (col of C).
                        # So W index = global_c * n + global_r.
                        
                        W[global_c * n + global_r][col] += coeff
                    
                    col += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
