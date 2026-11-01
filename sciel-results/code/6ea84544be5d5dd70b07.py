"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 2]

import time

# EVOLVE-BLOCK-START
def construct():
    """
    Strassen's algorithm provides a rank-7 decomposition for 2x2 matrix multiplication.
    
    The multiplication tensor T_{i,j,k} (for C = A*B, C_{ik} = sum_j A_{ij} B_{jk})
    has T_{i,j,k} = 1 for all i,j,k in {0,1}.
    
    Strassen's 7 products:
    M1 = (A00 + A11)(B00 + B11)
    M2 = (A10 + A11) B00
    M3 = A00 (B10 - B11)
    M4 = A11 (B01 - B00)
    M5 = (A00 + A01) B11
    M6 = (A10 - A00)(B00 + B01)
    M7 = (A01 - A11)(B10 + B11)
    
    C00 = M1 + M4 - M5 + M7
    C01 = M3 + M5
    C10 = M2 + M4
    C11 = M1 - M2 + M3 + M6
    
    We need U, V, W such that:
    sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = T_{i,j,k} = 1
    
    With n=m=p=2:
    U has shape (4, 7), indexed by (i,j) -> row i*2+j
    V has shape (4, 7), indexed by (j,k) -> row j*2+k
    W has shape (4, 7), indexed by (k,i) -> row k*2+i
    
    For each of the 7 Strassen products r, we define:
    - U[i,j,r] = coefficient of A_{ij} in the r-th product
    - V[j,k,r] = coefficient of B_{jk} in the r-th product
    - W[k,i,r] = coefficient of C_{ki} that includes the r-th product
    """
    start_time = time.time()
    
    n, m, p = DIMENSIONS
    rank = 7
    
    # U[row = i*m+j, col = r] = coefficient of A[i,j] in product r
    # V[row = j*p+k, col = r] = coefficient of B[j,k] in product r
    # W[row = k*n+i, col = r] = coefficient in C[k,i] from product r
    
    # Product 0: M1 = (A00 + A11)(B00 + B11)
    # A coefficients: A00: 1, A11: 1, others 0
    # B coefficients: B00: 1, B11: 1, others 0
    # C contributions: C00: +1, C11: +1
    
    # Product 1: M2 = (A10 + A11) B00
    # A coefficients: A10: 1, A11: 1, others 0
    # B coefficients: B00: 1, others 0
    # C contributions: C10: +1, C11: -1
    
    # Product 2: M3 = A00 (B10 - B11)
    # A coefficients: A00: 1, others 0
    # B coefficients: B10: 1, B11: -1, others 0
    # C contributions: C01: +1, C11: +1
    
    # Product 3: M4 = A11 (B01 - B00)
    # A coefficients: A11: 1, others 0
    # B coefficients: B01: 1, B00: -1, others 0
    # C contributions: C00: +1, C10: +1
    
    # Product 4: M5 = (A00 + A01) B11
    # A coefficients: A00: 1, A01: 1, others 0
    # B coefficients: B11: 1, others 0
    # C contributions: C00: -1, C01: +1
    
    # Product 5: M6 = (A10 - A00)(B00 + B01)
    # A coefficients: A10: 1, A00: -1, others 0
    # B coefficients: B00: 1, B01: 1, others 0
    # C contributions: C11: +1
    
    # Product 6: M7 = (A01 - A11)(B10 + B11)
    # A coefficients: A01: 1, A11: -1, others 0
    # B coefficients: B10: 1, B11: 1, others 0
    # C contributions: C00: +1
    
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]
    
    # Define each product
    # (U_coeffs, V_coeffs, W_coeffs) where:
    # U_coeffs: dict from (i,j) to coeff
    # V_coeffs: dict from (j,k) to coeff
    # W_coeffs: dict from (k,i) to coeff
    
    products = [
        # M1 = (A00 + A11)(B00 + B11)
        ({(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}),
        # M2 = (A10 + A11) B00
        ({(1,0): 1, (1,1): 1}, {(0,0): 1}, {(1,0): 1, (1,1): -1}),
        # M3 = A00 (B10 - B11)
        ({(0,0): 1}, {(1,0): 1, (1,1): -1}, {(0,1): 1, (1,1): 1}),
        # M4 = A11 (B01 - B00)
        ({(1,1): 1}, {(0,1): 1, (0,0): -1}, {(0,0): 1, (1,0): 1}),
        # M5 = (A00 + A01) B11
        ({(0,0): 1, (0,1): 1}, {(1,1): 1}, {(0,0): -1, (0,1): 1}),
        # M6 = (A10 - A00)(B00 + B01)
        ({(1,0): 1, (0,0): -1}, {(0,0): 1, (0,1): 1}, {(1,1): 1}),
        # M7 = (A01 - A11)(B10 + B11)
        ({(0,1): 1, (1,1): -1}, {(1,0): 1, (1,1): 1}, {(0,0): 1}),
    ]
    
    for r, (u_coeff, v_coeff, w_coeff) in enumerate(products):
        for (i, j), val in u_coeff.items():
            U[i * m + j][r] = val
        for (j, k), val in v_coeff.items():
            V[j * p + k][r] = val
        for (k, i), val in w_coeff.items():
            W[k * n + i][r] = val
    
    # Verify the decomposition
    # T[i,j,k] should equal sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                total = 0
                for r in range(rank):
                    total += U[i * m + j][r] * V[j * p + k][r] * W[k * n + i][r]
                if total != 1:
                    # Fallback to schoolbook if verification fails
                    rank_fallback = n * m * p
                    U_fb = [[0] * rank_fallback for _ in range(n * m)]
                    V_fb = [[0] * rank_fallback for _ in range(m * p)]
                    W_fb = [[0] * rank_fallback for _ in range(p * n)]
                    col = 0
                    for i2 in range(n):
                        for j2 in range(m):
                            for k2 in range(p):
                                U_fb[i2 * m + j2][col] = 1
                                V_fb[j2 * p + k2][col] = 1
                                W_fb[k2 * n + i2][col] = 1
                                col += 1
                    return {"U": U_fb, "V": V_fb, "W": W_fb}
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
