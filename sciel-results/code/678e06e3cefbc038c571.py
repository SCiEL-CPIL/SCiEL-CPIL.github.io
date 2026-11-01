"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 2]

# EVOLVE-BLOCK-START
def construct():
    """
    Rank-7 exact tensor decomposition for 2x2 matrix multiplication (Strassen).
    
    Indexing convention:
      U[i*m + j, r]  — coefficient for A[i,j] in product r
      V[j*p + k, r]  — coefficient for B[j,k] in product r
      W[k*n + i, r]  — coefficient for C[i,k] in product r  (trace convention)
    
    Tensor identity: T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    """
    n, m, p = DIMENSIONS
    rank = 7
    
    # Strassen's 7 intermediate products.
    # Each entry: (U_coeffs, V_coeffs, W_coeffs)
    #   U_coeffs: {A_flat_index: coeff}  where A_flat_index = i*m + j
    #   V_coeffs: {B_flat_index: coeff}  where B_flat_index = j*p + k
    #   W_coeffs: {C_flat_index: coeff}  where C_flat_index = k*n + i
    #
    # A flat indices: A00=0, A01=1, A10=2, A11=3
    # B flat indices: B00=0, B01=1, B10=2, B11=3
    # C flat indices (W[k*n+i]): C00=0, C10=1, C01=2, C11=3
    
    strassen_products = [
        # M0 = (A00 + A11)(B00 + B11)
        # C00 += M0, C11 += M0
        ({0: 1, 3: 1}, {0: 1, 3: 1}, {0: 1, 3: 1}),
        
        # M1 = (A10 + A11) * B00
        # C10 += M1, C11 -= M1
        ({2: 1, 3: 1}, {0: 1}, {1: 1, 3: -1}),
        
        # M2 = A00 * (B01 - B11)
        # C01 += M2, C11 += M2
        ({0: 1}, {1: 1, 3: -1}, {2: 1, 3: 1}),
        
        # M3 = A11 * (B10 - B00)
        # C00 += M3, C10 += M3
        ({3: 1}, {2: 1, 0: -1}, {0: 1, 1: 1}),
        
        # M4 = (A00 + A01) * B11
        # C00 -= M4, C01 += M4
        ({0: 1, 1: 1}, {3: 1}, {0: -1, 2: 1}),
        
        # M5 = (A10 - A00) * (B00 + B01)
        # C11 += M5
        ({2: 1, 0: -1}, {0: 1, 1: 1}, {3: 1}),
        
        # M6 = (A01 - A11) * (B10 + B11)
        # C00 += M6
        ({1: 1, 3: -1}, {2: 1, 3: 1}, {0: 1}),
    ]
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for r, (u_c, v_c, w_c) in enumerate(strassen_products):
        for idx, val in u_c.items():
            U[idx][r] = val
        for idx, val in v_c.items():
            V[idx][r] = val
        for idx, val in w_c.items():
            W[idx][r] = val
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
