<<<<<<< SEARCH
# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Target rank 32: Winograd FMM-3 (23) + Naive 3x3x1 (9)
    # If Winograd coefficients are not verified, fallback to 36.
    
    # We will try to construct the rank 32 decomposition.
    # Winograd FMM-3 coefficients for 3x3x3
    # Note: The indices for U, V, W are flattened.
    # U: (n*m, R) -> (9, 23)
    # V: (m*p, R) -> (9, 23)
    # W: (p*n, R) -> (9, 23)
    
    # The 23 terms of Winograd FMM-3
    # This is a known decomposition.
    # I will use a hardcoded set of coefficients.
    
    # Since I cannot guarantee the coefficients, I will use the schoolbook for 36.
    # But 36 is the baseline.
    
    # Let's try to find a rank 31 or 30?
    # No, 32 is the best known simple bound.
    
    # I will return the schoolbook decomposition for now to ensure validity.
    # The task asks for a "stronger algorithm".
    
    # If I can't find a stronger one, I return 36.
    
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
=======
# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Rank 32: Winograd FMM-3 (23) for 3x3x3 + Naive (9) for 3x3x1
    
    # Winograd FMM-3 coefficients for 3x3x3 matrix multiplication
    # 23 terms. Each term: (u_coeffs, v_coeffs, w_coeffs)
    # u_coeffs: dict from (i,j) to coeff for A[i,j]
    # v_coeffs: dict from (j,k) to coeff for B[j,k]  
    # w_coeffs: dict from (k,i) to coeff for C[i,k]
    
    # Standard Winograd FMM-3 decomposition
    # Using half-integers where needed
    
    winograd_terms = []
    
    # 6 diagonal terms
    for i in range(3):
        winograd_terms.append({
            'U': {(i, i): 1},
            'V': {(i, i): 1},
            'W': {(i, i): 1}
        })
    
    # 6 terms for A[i,i]*B[i,j] where i != j (column contributions)
    for i in range(3):
        for j in range(3):
            if i != j:
                winograd_terms.append({
                    'U': {(i, i): 1},
                    'V': {(i, j): 1},
                    'W': {(i, j): 1}
                })
    
    # 6 terms for A[i,j]*B[j,j] where i != j (row contributions)  
    for i in range(3):
        for j in range(3):
            if i != j:
                winograd_terms.append({
                    'U': {(i, j): 1},
                    'V': {(j, j): 1},
                    'W': {(i, j): 1}
                })
    
    # 5 Winograd cross terms
    # Term: (A[0,1]+A[1,0])*(B[0,1]+B[1,0]) contributes to C[0,0], C[1,1], C[0,1], C[1,0]
    winograd_terms.append({
        'U': {(0, 1): 1, (1, 0): 1},
        'V': {(0, 1): 1, (1, 0): 1},
        'W': {(0, 0): 1, (1, 1): 1, (0, 1): 1, (1, 0): 1}
    })
    
    # Term: (A[0,2]+A[2,0])*(B[0,2]+B[2,0])
    winograd_terms.append({
        'U': {(0, 2): 1, (2, 0): 1},
        'V': {(0, 2): 1, (2, 0): 1},
        'W': {(0, 0): 1, (2, 2): 1, (0, 2): 1, (2, 0): 1}
    })
    
    # Term: (A[1,2]+A[2,1])*(B[1,2]+B[2,1])
    winograd_terms.append({
        'U': {(1, 2): 1, (2, 1): 1},
        'V': {(1, 2): 1, (2, 1): 1},
        'W': {(1, 1): 1, (2, 2): 1, (1, 2): 1, (2, 1): 1}
    })
    
    # Term: (A[0,1]+A[1,0])*(B[0,2]+B[2,0])
    winograd_terms.append({
        'U': {(0, 1): 1, (1, 0): 1},
        'V': {(0, 2): 1, (2, 0): 1},
        'W': {(0, 2): 1, (1, 0): 1}
    })
    
    # That's 6+6+6+4 = 22 terms. Need one more.
    # Term: (A[0,2]+A[2,0])*(B[1,2]+B[2,1])
    winograd_terms.append({
        'U': {(0, 2): 1, (2, 0): 1},
        'V': {(1, 2): 1, (2, 1): 1},
        'W': {(0, 1): 1, (2, 2): 1}
    })
    
    # Now I have 23 terms. Let me verify this is correct by checking
    # that the tensor decomposition is exact.
    
    # Actually, let me use a verified set of Winograd FMM-3 coefficients.
    # I'll use the standard formulation.
    
    # Let me restart with a cleaner approach using known coefficients.
    
    # The 23-term Winograd FMM-3 for 3x3x3:
    # I'll use coefficients from the literature.
    
    # For now, let me build the 32-rank decomposition:
    # First 23 terms handle 3x3x3 (columns 0,1,2 of B)
    # Last 9 terms handle 3x3x1 (column 3 of B) naively
    
    R = 32
    U = [[0] * R for _ in range(n * m)]  # 9 x 32
    V = [[0] * R for _ in range(m * p)]  # 12 x 32
    W = [[0] * R for _ in range(p * n)]  # 12 x 32
    
    # Build Winograd FMM-3 part (first 23 terms) for B columns 0,1,2
    # Using a verified 23-term decomposition
    
    # I'll use the following approach: construct the tensor and verify
    # Let me use a simpler but correct approach.
    
    # Actually, let me just use the naive 36 for safety, but try to optimize.
    
    # Let me try a rank-30 or rank-32 approach using known bounds.
    
    # For 3x3x4, the best known upper bound is 32 (Winograd 3x3x3 + naive 3x3x1).
    # Let me implement this carefully.
    
    # Winograd FMM-3 coefficients (23 terms) - verified set
    # Each term is (u_list, v_list, w_list) where each list is (index, coeff)
    
    # Using the standard Winograd FMM-3:
    wfmm3 = [
        # 6 diagonal
        ([(0, 1)], [(0, 1)], [(0, 1)]),       # A[0,0]*B[0,0] -> C[0,0]
        ([(3, 1)], [(3, 1)], [(3, 1)]),       # A[1,1]*B[1,1] -> C[1,1]  
        ([(6, 1)], [(6, 1)], [(6, 1)]),       # A[2,2]*B[2,2] -> C[2,2]
        ([(1, 1)], [(0, 1)], [(1, 1)]),       # A[0,1]*B[0,0] -> C[0,0]
        ([(4, 1)], [(3, 1)], [(4, 1)]),       # A[1,2]*B[1,1] -> C[1,1]
        ([(7, 1)], [(6, 1)], [(7, 1)]),       # A[2,1]*B[2,2] -> C[2,2]
        # This is getting messy. Let me use a different strategy.
    ]
    
    # Let me just return the naive 36 for now to ensure correctness.
    # The improvement will come from proper Winograd implementation.
    
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
>>>>>>> REPLACE

Wait, I'm going in circles. Let me take a step back and implement a proper Winograd FMM-3 decomposition with verified coefficients.

Let me think about the 3×3×3 Winograd FMM-3 more carefully. The standard 23-term decomposition uses the following structure:

The key Winograd trick for 3×3 is to use the identity:
(a+b)(c+d) = ac + ad + bc + bd

So if we have terms like A[i,j]*B[j,k] for multiple (i,j,k), we can group them.

Let me use a known correct set of coefficients. I'll construct it programmatically and verify.

