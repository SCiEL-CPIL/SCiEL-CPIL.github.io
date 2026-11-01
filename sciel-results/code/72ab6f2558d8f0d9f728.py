import time

DIMENSIONS = [2, 2, 2]

# EVOLVE-BLOCK-START
def construct():
    """
    Returns a rank-7 tensor decomposition for 2x2 matrix multiplication.
    This is the minimal rank for 2x2 matrix multiplication.
    """
    n, m, p = DIMENSIONS
    rank = 7
    
    # Initialize U, V, W with zeros
    # U: (n*m, rank) = (4, 7)
    # V: (m*p, rank) = (4, 7)
    # W: (p*n, rank) = (4, 7)
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen's 7 intermediate products coefficients
    # Format: (U_coeffs, V_coeffs, W_coeffs)
    # U indices: i*m + j (A[i,j]) -> 0:A00, 1:A01, 2:A10, 3:A11
    # V indices: j*p + k (B[j,k]) -> 0:B00, 1:B01, 2:B10, 3:B11
    # W indices: k*n + i (C[i,k]) -> 0:C00, 1:C10, 2:C01, 3:C11
    
    # M1 = (A00 + A11)(B00 + B11)
    # M2 = (A10 + A11) * B00
    # M3 = A00 * (B01 - B11)
    # M4 = A11 * (B10 - B00)
    # M5 = (A00 + A01) * B11
    # M6 = (A10 - A00) * (B00 + B01)
    # M7 = (A01 - A11) * (B10 + B11)
    
    # Output:
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    strassen_terms = [
        # r=0: M1
        ({0: 1, 3: 1}, {0: 1, 3: 1}, {0: 1, 3: 1}),
        # r=1: M2
        ({2: 1, 3: 1}, {0: 1}, {1: 1, 3: -1}),
        # r=2: M3
        ({0: 1}, {1: 1, 3: -1}, {2: 1, 3: 1}),
        # r=3: M4
        ({3: 1}, {2: 1, 0: -1}, {0: 1, 1: 1}),
        # r=4: M5
        ({0: 1, 1: 1}, {3: 1}, {0: -1, 2: 1}),
        # r=5: M6
        ({2: 1, 0: -1}, {0: 1, 1: 1}, {3: 1}),
        # r=6: M7
        ({1: 1, 3: -1}, {2: 1, 3: 1}, {0: 1}),
    ]
    
    for r, (u_c, v_c, w_c) in enumerate(strassen_terms):
        for idx, val in u_c.items():
            U[idx][r] = val
        for idx, val in v_c.items():
            V[idx][r] = val
        for idx, val in w_c.items():
            W[idx][r] = val
            
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
