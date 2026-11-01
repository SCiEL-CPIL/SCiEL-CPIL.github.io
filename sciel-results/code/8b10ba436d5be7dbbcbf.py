import time

DIMENSIONS = [2, 2, 2]

# EVOLVE-BLOCK-START
def construct():
    """
    Returns a rank-7 tensor decomposition for 2x2 matrix multiplication.
    This is the minimal rank for 2x2 matrix multiplication (Strassen's algorithm).
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
    
    # Strassen's 7 intermediate products
    # A indices: 0:A00, 1:A01, 2:A10, 3:A11
    # B indices: 0:B00, 1:B01, 2:B10, 3:B11
    # C indices (W): 0:C00, 1:C01, 2:C10, 3:C11 (Note: W[k*n+i] => W[0]=C00, W[1]=C01, W[2]=C10, W[3]=C11)
    # Wait, the prompt says W[k*n+i, r].
    # If k=0, i=0 -> index 0 (C00)
    # If k=1, i=0 -> index 1 (C01)
    # If k=0, i=1 -> index 2 (C10)
    # If k=1, i=1 -> index 3 (C11)
    
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
    
    # Define coefficients for each rank dimension r
    # Format: {U_index: coeff, ...}, {V_index: coeff, ...}, {W_index: coeff, ...}
    
    strassen_terms = [
        # r=0: M1
        ({0: 1, 3: 1}, {0: 1, 3: 1}, {0: 1, 3: 1}),
        # r=1: M2
        ({2: 1, 3: 1}, {0: 1}, {2: 1, 3: -1}),
        # r=2: M3
        ({0: 1}, {1: 1, 3: -1}, {1: 1, 3: 1}),
        # r=3: M4
        ({3: 1}, {2: 1, 0: -1}, {0: 1, 2: 1}),
        # r=4: M5
        ({0: 1, 1: 1}, {3: 1}, {0: -1, 1: 1}),
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
