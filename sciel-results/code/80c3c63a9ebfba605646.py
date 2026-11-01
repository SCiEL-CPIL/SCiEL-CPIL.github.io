import time

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    
    # The minimal rank for 2x2 matrix multiplication is 7.
    # We construct the Strassen decomposition.
    rank = 7
    n = 2
    m = 2
    p = 2
    
    # Initialize matrices
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Define the 7 Strassen products.
    # Each product r is defined by:
    # U_r: coefficients for A (flattened i*m + j)
    # V_r: coefficients for B (flattened j*p + k)
    # W_r: coefficients for C (flattened k*n + i)
    
    # We define them as lists of (index, coeff) pairs to be concise and distinct in structure.
    
    # P1: (A00 + A11)(B00 + B11)
    # A: 00 (0), 11 (3)
    # B: 00 (0), 11 (3)
    # C: 
    # Standard Strassen:
    # C00 += P1
    # C11 += P1
    # C01 -= P4, etc.
    
    # Let's use a verified Strassen table for the convention W[k*n+i].
    # Based on the previous valid program, the mapping was:
    
    # Term 1: (A00+A11)(B00+B11)
    u1 = [(0, 1), (3, 1)]
    v1 = [(0, 1), (3, 1)]
    w1 = [(0, 1), (3, 1)] # Indices 0 (k=0,i=0 -> C00?) and 3 (k=1,i=1 -> C11?)
    
    # Term 2: (A10+A11)B00
    u2 = [(2, 1), (3, 1)]
    v2 = [(0, 1)]
    w2 = [(1, 1), (3, -1)] # Indices 1 (k=0,i=1 -> C10) and 3 (k=1,i=1 -> C11)
    
    # Term 3: A00(B01-B11)
    u3 = [(0, 1)]
    v3 = [(1, 1), (3, -1)]
    w3 = [(2, 1), (3, 1)] # Indices 2 (k=1,i=0 -> C01) and 3 (k=1,i=1 -> C11)
    
    # Term 4: A11(B10-B00)
    u4 = [(3, 1)]
    v4 = [(2, 1), (0, -1)]
    w4 = [(0, 1), (1, 1)] # Indices 0 (C00) and 1 (C10)
    
    # Term 5: (A00+A01)B11
    u5 = [(0, 1), (1, 1)]
    v5 = [(3, 1)]
    w5 = [(0, -1), (2, 1)] # Indices 0 (C00) and 2 (C01)
    
    # Term 6: (A10-A00)(B00+B01)
    u6 = [(2, 1), (0, -1)]
    v6 = [(0, 1), (1, 1)]
    w6 = [(3, 1)] # Index 3 (C11)
    
    # Term 7: (A01-A11)(B10+B11)
    u7 = [(1, 1), (3, -1)]
    v7 = [(2, 1), (3, 1)]
    w7 = [(0, 1)] # Index 0 (C00)
    
    terms = [
        (u1, v1, w1),
        (u2, v2, w2),
        (u3, v3, w3),
        (u4, v4, w4),
        (u5, v5, w5),
        (u6, v6, w6),
        (u7, v7, w7)
    ]
    
    for r, (u_terms, v_terms, w_terms) in enumerate(terms):
        for idx, val in u_terms:
            U[idx][r] = val
        for idx, val in v_terms:
            V[idx][r] = val
        for idx, val in w_terms:
            W[idx][r] = val
            
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
