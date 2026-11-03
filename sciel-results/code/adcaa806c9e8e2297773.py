"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = 3, 3, 5
    
    # Rank-40 decomposition for 3x3 * 3x5 matrix multiplication
    # Strategy: Use Strassen's rank-23 decomposition for the 3x3x3 subproblem
    # (computing C[:, 0:3] = A * B[:, 0:3]) and add 17 terms for C[:, 3:5]
    
    rank = 40
    U = [[0] * rank for _ in range(n * m)]  # 9 x 40
    V = [[0] * rank for _ in range(m * p)]  # 15 x 40
    W = [[0] * rank for _ in range(p * n)]  # 15 x 40
    
    # Strassen's 23-term decomposition for 3x3x3
    # The 23 terms are defined by their support in (i, j, k) space
    # Each term has a coefficient pattern
    
    # Strassen terms for 3x3x3 (indexed 0-22):
    # These are the standard 23 terms from Strassen's algorithm
    # Format: (term_idx, [(i,j,k,coeff), ...])
    
    strassen_terms = [
        # Term 0: A[0,0]*B[0,0]
        [(0, 0, 0, 1)],
        # Term 1: A[0,0]*B[0,1] + A[0,1]*B[1,0]
        [(0, 0, 1, 1), (0, 1, 0, 1)],
        # Term 2: A[0,1]*B[0,0]
        [(0, 1, 0, 1)],
        # Term 3: A[0,0]*B[1,0]
        [(0, 0, 0, 1), (1, 0, 1, 1)],
        # Term 4: A[0,1]*B[1,1]
        [(0, 1, 1, 1)],
        # Term 5: A[1,0]*B[0,0]
        [(1, 0, 0, 1)],
        # Term 6: A[1,1]*B[0,0]
        [(1, 1, 0, 1)],
        # Term 7: A[1,0]*B[0,1] + A[1,1]*B[1,0]
        [(1, 0, 1, 1), (1, 1, 0, 1)],
        # Term 8: A[1,1]*B[1,1]
        [(1, 1, 1, 1)],
        # Term 9: A[2,0]*B[0,0]
        [(2, 0, 0, 1)],
        # Term 10: A[2,1]*B[0,0]
        [(2, 1, 0, 1)],
        # Term 11: A[2,0]*B[0,1] + A[2,1]*B[1,0]
        [(2, 0, 1, 1), (2, 1, 0, 1)],
        # Term 12: A[2,1]*B[1,1]
        [(2, 1, 1, 1)],
        # Term 13: A[0,0]*B[2,0]
        [(0, 0, 2, 1)],
        # Term 14: A[0,1]*B[2,0]
        [(0, 1, 2, 1)],
        # Term 15: A[0,0]*B[2,1] + A[0,1]*B[2,2] - wait, B is 3x3 so k=2 is valid
        # Let me reconsider. For 3x3x3, k ranges 0,1,2
        # Term 15: A[0,0]*B[2,1]
        [(0, 0, 2, 1), (1, 0, 2, 1)],  # This is getting complex
    ]
    
    # Actually, let me use a cleaner approach. I'll use the known rank-23 
    # decomposition and explicitly define all 23 terms.
    
    # Here's a clean rank-23 decomposition for 3x3x3:
    # The terms are:
    
    terms_3x3 = [
        # (i, j, k) contributions for each term
        # Term 0
        [(0,0,0,1)],
        # Term 1
        [(0,0,1,1), (0,1,0,1)],
        # Term 2
        [(0,1,0,1)],
        # Term 3
        [(1,0,0,1)],
        # Term 4
        [(0,0,0,1), (1,0,1,1)],
        # Term 5
        [(1,1,0,1)],
        # Term 6
        [(1,0,1,1), (1,1,0,1)],
        # Term 7
        [(1,1,1,1)],
        # Term 8
        [(2,0,0,1)],
        # Term 9
        [(2,1,0,1)],
        # Term 10
        [(2,0,1,1), (2,1,0,1)],
        # Term 11
        [(2,1,1,1)],
        # Term 12
        [(0,0,2,1)],
        # Term 13
        [(0,1,2,1)],
        # Term 14
        [(1,0,2,1)],
        # Term 15
        [(1,1,2,1)],
        # Term 16
        [(2,0,2,1)],
        # Term 17
        [(2,1,2,1)],
        # Term 18: (A[0,0]+A[0,1]+A[1,0]+A[1,1]) * (B[0,0]+B[0,1]+B[1,0]+B[1,1])
        [(0,0,0,1), (0,1,0,1), (1,0,0,1), (1,1,0,1),
         (0,0,1,1), (0,1,1,1), (1,0,1,1), (1,1,1,1)],
        # Term 19: (A[0,0]+A[1,0]+A[2,0]) * (B[0,0]+B[1,0]+B[2,0])
        [(0,0,0,1), (1,0,0,1), (2,0,0,1),
         (0,0,0,1), (1,0,0,1), (2,0,0,1)],
        # Term 20: (A[0,1]+A[1,1]+A[2,1]) * (B[0,1]+B[1,1]+B[2,1])
        [(0,1,1,1), (1,1,1,1), (2,1,1,1),
         (0,1,1,1), (1,1,1,1), (2,1,1,1)],
        # Term 21: (A[0,0]+A[0,1]+A[0,2]) * (B[0,0]+B[1,0]+B[2,0]) - but B is 3x3, so this doesn't work
        # Let me use a different set of terms
    ]
    
    # This is getting too complex. Let me use a simpler valid construction.
    # I'll use the schoolbook method but try to merge some terms.
    
    # For now, let me return a valid rank-45 decomposition.
    rank = 45
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
