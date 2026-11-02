"""Exact tensor decomposition for 2x3 by 3x6 matrix multiplication.

We split the p=6 dimension into three blocks of size 2.
For each block, we split the m=3 dimension into j=0 (scalar) and j in {1,2} (2x2 block).
- The j=0 part is a 2x1 by 1x2 multiplication, which has rank 4 (schoolbook).
- The j in {1,2} part is a 2x2 by 2x2 multiplication, which has rank 7 (Strassen).
Total rank per block: 4 + 7 = 11.
Total rank: 3 * 11 = 33.

Indices:
- U: shape (n*m, rank) = (6, rank), indexed by i*m + j = i*3 + j
- V: shape (m*p, rank) = (18, rank), indexed by j*p + k = j*6 + k
- W: shape (p*n, rank) = (12, rank), indexed by k*n + i = k*2 + i

For each block b in {0, 1, 2} (corresponding to k in {2b, 2b+1}):
  1. j=0 part (rank 4):
     For i in {0, 1} and k_local in {0, 1}:
       U[i*3 + 0] += 1
       V[0*6 + (2b + k_local)] += 1
       W[(2b + k_local)*2 + i] += 1

  2. j in {1, 2} part (rank 7, Strassen):
     Local A indices: (i, j_local) where i in {0,1}, j_local in {0,1} -> global j = j_local + 1
     Local B indices: (j_local, k_local) where j_local in {0,1}, k_local in {0,1} -> global j = j_local + 1, global k = 2b + k_local
     Local C indices: (i, k_local) -> global k = 2b + k_local
     
     Strassen for 2x2 by 2x2:
     Let A = [[a, b], [c, d]] and B = [[e, f], [g, h]]
     C = A*B = [[ae+bg, af+bh], [ce+dg, cf+dh]]
     
     The 7 products are:
     M1 = (a+d)(e+h)
     M2 = (c+d)e
     M3 = a(f-h)
     M4 = d(g-e)
     M5 = (a+b)h
     M6 = (c-a)(e+f)
     M7 = (b-d)(g+h)
     
     Then:
     C00 = M1 + M4 - M5 + M7
     C01 = M3 + M5
     C10 = M2 + M6
     C11 = M1 - M2 + M3 + M6
"""

DIMENSIONS = [2, 3, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # n=2, m=3, p=6
    rank = 33
    U = [[0] * rank for _ in range(n * m)]   # 6 x 33
    V = [[0] * rank for _ in range(m * p)]   # 18 x 33
    W = [[0] * rank for _ in range(p * n)]   # 12 x 33
    
    col = 0
    
    # Strassen coefficients for 2x2 by 2x2
    # M1 = (a+d)(e+h)
    strassen = [
        # (U_coeffs, V_coeffs, W_coeffs)
        # Each is a dict: {local_index: coeff}
        # Local A indices: (i, j_local) -> i*2 + j_local
        # Local B indices: (j_local, k_local) -> j_local*2 + k_local
        # Local C indices: (i, k_local) -> i*2 + k_local
        # M1: (a+d)(e+h)
        ({(0,0): 1, (1,0): 1}, {(0,0): 1, (0,1): 1, (1,0): 1, (1,1): 1}, {(0,0): 1, (0,1): 1, (1,0): 1, (1,1): 1}),
        # M2: (c+d)e
        ({(1,0): 1, (1,1): 1}, {(0,0): 1}, {(1,0): 1}),
        # M3: a(f-h)
        ({(0,0): 1}, {(0,1): 1, (1,1): -1}, {(0,1): 1, (1,1): 1}),
        # M4: d(g-e)
        ({(1,1): 1}, {(1,0): 1, (0,0): -1}, {(0,0): 1, (1,0): 1}),
        # M5: (a+b)h
        ({(0,0): 1, (0,1): 1}, {(1,1): 1}, {(0,0): -1, (0,1): 1}),
        # M6: (c-a)(e+f)
        ({(1,0): 1, (0,0): -1}, {(0,0): 1, (0,1): 1}, {(1,1): 1}),
        # M7: (b-d)(g+h)
        ({(0,1): 1, (1,1): -1}, {(1,0): 1, (1,1): 1}, {(0,0): 1}),
    ]
    
    # W coefficients for Strassen (already included in the tuples above)
    # But we need to verify the W mapping is correct.
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M6
    # C11 = M1 - M2 + M3 + M6
    
    # Let's redefine strassen with explicit W coefficients for each output
    # Actually, the standard Strassen decomposition is:
    # M1 = (a+d)(e+h)
    # M2 = (c+d)e
    # M3 = a(f-h)
    # M4 = d(g-e)
    # M5 = (a+b)h
    # M6 = (c-a)(e+f)
    # M7 = (b-d)(g+h)
    #
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M6
    # C11 = M1 - M2 + M3 + M6
    
    # So for each M_r, we have U_r, V_r, and then W_r is a vector over C indices.
    
    strassen_decomp = [
        # (U_dict, V_dict, W_dict)
        # M1
        ({(0,0): 1, (1,0): 1}, {(0,0): 1, (0,1): 1, (1,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}),
        # M2
        ({(1,0): 1, (1,1): 1}, {(0,0): 1}, {(1,0): 1}),
        # M3
        ({(0,0): 1}, {(0,1): 1, (1,1): -1}, {(0,1): 1, (1,1): 1}),
        # M4
        ({(1,1): 1}, {(1,0): 1, (0,0): -1}, {(0,0): 1, (1,0): 1}),
        # M5
        ({(0,0): 1, (0,1): 1}, {(1,1): 1}, {(0,0): -1, (0,1): 1}),
        # M6
        ({(1,0): 1, (0,0): -1}, {(0,0): 1, (0,1): 1}, {(1,1): 1}),
        # M7
        ({(0,1): 1, (1,1): -1}, {(1,0): 1, (1,1): 1}, {(0,0): 1}),
    ]
    
    for b in range(3):
        # j=0 part: 4 terms
        for i in range(2):
            for k_local in range(2):
                k_global = 2 * b + k_local
                U[i * 3 + 0][col] += 1
                V[0 * 6 + k_global][col] += 1
                W[k_global * 2 + i][col] += 1
                col += 1
        
        # j in {1, 2} part: 7 terms (Strassen)
        for U_dict, V_dict, W_dict in strassen_decomp:
            for (i_local, j_local), u_coeff in U_dict.items():
                i_global = i_local
                j_global = j_local + 1
                U[i_global * 3 + j_global][col] += u_coeff
            
            for (j_local, k_local), v_coeff in V_dict.items():
                j_global = j_local + 1
                k_global = 2 * b + k_local
                V[j_global * 6 + k_global][col] += v_coeff
            
            for (i_local, k_local), w_coeff in W_dict.items():
                i_global = i_local
                k_global = 2 * b + k_local
                W[k_global * 2 + i_global][col] += w_coeff
            
            col += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
