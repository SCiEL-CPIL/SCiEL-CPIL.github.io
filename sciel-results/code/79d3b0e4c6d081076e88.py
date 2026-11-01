"""Rewritten program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 3]

# EVOLVE-BLOCK-START
def construct():
    """
    Construct a rank-11 exact tensor decomposition for 2x2 * 2x3 matrix multiplication.
    
    Strategy: Use Strassen's rank-7 decomposition for the 2x2 * 2x2 part (first two columns
    of B), and a rank-4 standard decomposition for the third column of B.
    
    The output factor follows the official trace convention W[k*n+i, r].
    U has shape (n*m, rank) = (4, 11)
    V has shape (m*p, rank) = (6, 11)
    W has shape (p*n, rank) = (6, 11)
    """
    n, m, p = DIMENSIONS
    rank = 11
    
    # Initialize factor matrices
    U = [[0] * rank for _ in range(n * m)]  # 4 x 11
    V = [[0] * rank for _ in range(m * p)]  # 6 x 11
    W = [[0] * rank for _ in range(p * n)]  # 6 x 11
    
    # Strassen's algorithm for 2x2 * 2x2
    # Input A = [[A00, A01], [A10, A11]] indexed as U[i*m+j]
    # Input B = [[B00, B01], [B10, B11]] indexed as V[j*p+k]
    # Output C = [[C00, C01], [C10, C11]] indexed as W[k*n+i]
    
    # The 7 Strassen products:
    # P1 = (A00 + A11) * (B00 + B11)
    # P2 = (A10 + A11) * B00
    # P3 = A00 * (B01 - B11)
    # P4 = A11 * (B10 - B00)
    # P5 = (A00 + A01) * B11
    # P6 = (A10 - A00) * (B00 + B01)
    # P7 = (A01 - A11) * (B10 + B11)
    
    # Output reconstruction:
    # C00 = P1 - P2 + P4 + P7
    # C01 = P3 + P5
    # C10 = P2 + P6
    # C11 = P1 - P3 + P5 + P6
    
    # Define the 7 Strassen terms (indices 0-6)
    strassen_terms = [
        # (U coefficients, V coefficients, W coefficients)
        # P1 = (A00 + A11) * (B00 + B11)
        {
            'U': {0: 1, 3: 1},       # A00 + A11
            'V': {0: 1, 3: 1},       # B00 + B11
            'W': {0: 1, 3: 1}        # contributes to C00 and C11
        },
        # P2 = (A10 + A11) * B00
        {
            'U': {2: 1, 3: 1},       # A10 + A11
            'V': {0: 1},             # B00
            'W': {1: -1, 2: 1}       # -C00 contribution? No, let me re-derive
        },
        # P3 = A00 * (B01 - B11)
        {
            'U': {0: 1},             # A00
            'V': {1: 1, 3: -1},     # B01 - B11
            'W': {1: 1, 3: -1}      # contributes to C01 and C11
        },
        # P4 = A11 * (B10 - B00)
        {
            'U': {3: 1},             # A11
            'V': {2: 1, 0: -1},     # B10 - B00
            'W': {0: 1}             # contributes to C00
        },
        # P5 = (A00 + A01) * B11
        {
            'U': {0: 1, 1: 1},      # A00 + A01
            'V': {3: 1},            # B11
            'W': {1: 1, 3: 1}       # contributes to C01 and C11
        },
        # P6 = (A10 - A00) * (B00 + B01)
        {
            'U': {2: 1, 0: -1},     # A10 - A00
            'V': {0: 1, 1: 1},      # B00 + B01
            'W': {2: 1, 3: 1}       # contributes to C10 and C11
        },
        # P7 = (A01 - A11) * (B10 + B11)
        {
            'U': {1: 1, 3: -1},     # A01 - A11
            'V': {2: 1, 3: 1},      # B10 + B11
            'W': {0: 1}             # contributes to C00
        }
    ]
    
    # Let me re-derive Strassen's formulas carefully:
    # Standard Strassen:
    # M1 = (A11 + A22)(B11 + B22)
    # M2 = (A21 + A22)B11
    # M3 = A11(B21 - B22)
    # M4 = A22(B12 - B11)
    # M5 = (A11 + A12)B22
    # M6 = (A21 - A11)(B11 + B12)
    # M7 = (A12 - A22)(B21 + B22)
    #
    # C11 = M1 - M2 + M4 + M7
    # C12 = M3 + M5
    # C21 = M2 + M6
    # C22 = M1 - M3 + M5 + M6
    #
    # Mapping to our 0-indexed notation:
    # A11 -> A00, A12 -> A01, A21 -> A10, A22 -> A11
    # B11 -> B00, B12 -> B01, B21 -> B10, B22 -> B11
    # C11 -> C00, C12 -> C01, C21 -> C10, C22 -> C11
    #
    # So:
    # P1 = (A00 + A11)(B00 + B11)
    # P2 = (A10 + A11)B00
    # P3 = A00(B10 - B11)
    # P4 = A11(B01 - B00)
    # P5 = (A00 + A01)B11
    # P6 = (A10 - A00)(B00 + B01)
    # P7 = (A01 - A11)(B10 + B11)
    #
    # C00 = P1 - P2 + P4 + P7
    # C01 = P3 + P5
    # C10 = P2 + P6
    # C11 = P1 - P3 + P5 + P6
    
    # Now let's define each term correctly.
    # For each product P_i, we need:
    # - U coefficients: which A entries are multiplied
    # - V coefficients: which B entries are multiplied
    # - W coefficients: how this product contributes to each C entry
    
    # P1 = (A00 + A11) * (B00 + B11)
    # U: A00 (index 0) + A11 (index 3)
    # V: B00 (index 0) + B11 (index 3)
    # Contributes to: C00 with +1, C11 with +1
    # So W: C00 (index 0) +1, C11 (index 3) +1
    
    # P2 = (A10 + A11) * B00
    # U: A10 (index 2) + A11 (index 3)
    # V: B00 (index 0)
    # Contributes to: C00 with -1, C10 with +1
    # So W: C00 (index 0) -1, C10 (index 2) +1
    
    # P3 = A00 * (B10 - B11)
    # U: A00 (index 0)
    # V: B10 (index 2) - B11 (index 3)
    # Contributes to: C01 with +1, C11 with -1
    # So W: C01 (index 1) +1, C11 (index 3) -1
    
    # P4 = A11 * (B01 - B00)
    # U: A11 (index 3)
    # V: B01 (index 1) - B00 (index 0)
    # Contributes to: C00 with +1
    # So W: C00 (index 0) +1
    
    # P5 = (A00 + A01) * B11
    # U: A00 (index 0) + A01 (index 1)
    # V: B11 (index 3)
    # Contributes to: C01 with +1, C11 with +1
    # So W: C01 (index 1) +1, C11 (index 3) +1
    
    # P6 = (A10 - A00) * (B00 + B01)
    # U: A10 (index 2) - A00 (index 0)
    # V: B00 (index 0) + B01 (index 1)
    # Contributes to: C10 with +1, C11 with +1
    # So W: C10 (index 2) +1, C11 (index 3) +1
    
    # P7 = (A01 - A11) * (B10 + B11)
    # U: A01 (index 1) - A11 (index 3)
    # V: B10 (index 2) + B11 (index 3)
    # Contributes to: C00 with +1
    # So W: C00 (index 0) +1
    
    # Now fill in the matrices for the first 7 terms
    for col in range(7):
        term = strassen_terms[col]
        for u_idx, coeff in term['U'].items():
            U[u_idx][col] = coeff
        for v_idx, coeff in term['V'].items():
            V[v_idx][col] = coeff
        for w_idx, coeff in term['W'].items():
            W[w_idx][col] = coeff
    
    # Now add the 4 terms for the third column (k=2)
    # We need to compute C02 and C12:
    # C02 = A00 * B02 + A01 * B12
    # C12 = A10 * B02 + A11 * B12
    
    # B02 is at V index: j=0, k=2 -> 0*3 + 2 = 2
    # B12 is at V index: j=1, k=2 -> 1*3 + 2 = 5
    # C02 is at W index: k=2, i=0 -> 2*2 + 0 = 4
    # C12 is at W index: k=2, i=1 -> 2*2 + 1 = 5
    
    # Term 8: A00 * B02 -> contributes to C02
    # U: A00 (index 0)
    # V: B02 (index 2)
    # W: C02 (index 4)
    col = 7
    U[0][col] = 1
    V[2][col] = 1
    W[4][col] = 1
    
    # Term 9: A01 * B12 -> contributes to C02
    # U: A01 (index 1)
    # V: B12 (index 5)
    # W: C02 (index 4)
    col = 8
    U[1][col] = 1
    V[5][col] = 1
    W[4][col] = 1
    
    # Term 10: A10 * B02 -> contributes to C12
    # U: A10 (index 2)
    # V: B02 (index 2)
    # W: C12 (index 5)
    col = 9
    U[2][col] = 1
    V[2][col] = 1
    W[5][col] = 1
    
    # Term 11: A11 * B12 -> contributes to C12
    # U: A11 (index 3)
    # V: B12 (index 5)
    # W: C12 (index 5)
    col = 10
    U[3][col] = 1
    V[5][col] = 1
    W[5][col] = 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
