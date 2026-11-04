"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = 3, 3, 6
    # We decompose 3x6 as three 3x2 blocks, each of rank 7
    # Total rank = 21
    
    rank = 21
    U = [[0] * rank for _ in range(n * m)]  # 9 x 21
    V = [[0] * rank for _ in range(m * p)]  # 18 x 21
    W = [[0] * rank for _ in range(p * n)]  # 18 x 21
    
    # Rank-7 decomposition for 3x2 matrix multiplication
    # For a 3x3 matrix A and 3x2 matrix B, output C is 3x2
    # C[i,k] = sum_j A[i,j] * B[j,k] for i=0,1,2; j=0,1,2; k=0,1
    # 
    # The 7 rank-1 terms:
    # Term 0: A[0,0]*B[0,0] -> C[0,0]
    # Term 1: A[0,1]*B[1,0] -> C[0,0]
    # Term 2: A[0,2]*B[2,0] -> C[0,0]
    # Term 3: A[1,0]*B[0,1] -> C[1,1]
    # Term 4: A[1,1]*B[1,1] -> C[1,1]
    # Term 5: A[1,2]*B[2,1] -> C[1,1]
    # Term 6: Handles remaining terms C[0,1], C[1,0], C[2,0], C[2,1]
    
    # Standard rank-7 for 3x2:
    # Using the decomposition from the literature:
    # t0: (i=0,j=0) x (j=0,k=0) -> (k=0,i=0)
    # t1: (i=0,j=1) x (j=1,k=0) -> (k=0,i=0)
    # t2: (i=0,j=2) x (j=2,k=0) -> (k=0,i=0)
    # t3: (i=1,j=0) x (j=0,k=1) -> (k=1,i=1)
    # t4: (i=1,j=1) x (j=1,k=1) -> (k=1,i=1)
    # t5: (i=1,j=2) x (j=2,k=1) -> (k=1,i=1)
    # t6: (i=2,j=0) + corrections x (j=0,k=0) + corrections -> (k=0,i=2) + corrections
    
    # Actually, let me use a cleaner construction. The rank-7 decomposition
    # for 3x2 matrix multiplication can be written as:
    #
    # For each of the 7 terms, we specify (u_indices, v_indices, w_indices, coefficients)
    
    # Define the 7 terms for a single 3x2 block
    # Each term: (A_entries, B_entries, C_entries) as list of (index, coeff)
    
    terms_3x2 = [
        # Term 0: C[0,0] part 1
        {'A': [(0*3+0, 1)], 'B': [(0*2+0, 1)], 'C': [(0*3+0, 1)]},  # A[0,0], B[0,0], C[0,0]
        # Term 1: C[0,0] part 2
        {'A': [(0*3+1, 1)], 'B': [(1*2+0, 1)], 'C': [(0*3+0, 1)]},  # A[0,1], B[1,0], C[0,0]
        # Term 2: C[0,0] part 3
        {'A': [(0*3+2, 1)], 'B': [(2*2+0, 1)], 'C': [(0*3+0, 1)]},  # A[0,2], B[2,0], C[0,0]
        # Term 3: C[1,1] part 1
        {'A': [(1*3+0, 1)], 'B': [(0*2+1, 1)], 'C': [(1*3+1, 1)]},  # A[1,0], B[0,1], C[1,1]
        # Term 4: C[1,1] part 2
        {'A': [(1*3+1, 1)], 'B': [(1*2+1, 1)], 'C': [(1*3+1, 1)]},  # A[1,1], B[1,1], C[1,1]
        # Term 5: C[1,1] part 3
        {'A': [(1*3+2, 1)], 'B': [(2*2+1, 1)], 'C': [(1*3+1, 1)]},  # A[1,2], B[2,1], C[1,1]
        # Term 6: Handles C[2,0], C[0,1], C[1,0], C[2,1]
        {'A': [(2*3+0, 1), (2*3+1, 1), (2*3+2, 1), (0*3+0, 1), (0*3+1, 1), (0*3+2, 1), (1*3+0, 1), (1*3+1, 1), (1*3+2, 1)],
         'B': [(0*2+0, 1), (1*2+0, 1), (2*2+0, 1), (0*2+1, 1), (1*2+1, 1), (2*2+1, 1)],
         'C': [(2*3+0, 1), (2*3+1, 1), (0*3+1, -1), (1*3+0, -1), (0*3+0, -1), (1*3+1, -1)]},
    ]
    
    # This last term is not quite right. Let me use a different approach.
    # 
    # The correct rank-7 decomposition for 3x2 matrix multiplication:
    # Based on the standard construction, the 7th term should handle
    # the "remaining" products in a clever way.
    
    # Let me restart with a cleaner, verified construction.
    
    # For 3x2: A is 3x3, B is 3x2, C is 3x2
    # C[i,k] = sum_{j=0}^{2} A[i,j] * B[j,k]
    # 
    # Rank 7 terms (using half-integer coefficients where needed):
    # I'll use the decomposition where:
    
    terms_3x2_clean = []
    
    # Terms for C[0,0]: A[0,0]B[0,0] + A[0,1]B[1,0] + A[0,2]B[2,0]
    # Terms for C[1,1]: A[1,0]B[0,1] + A[1,1]B[1,1] + A[1,2]B[2,1]
    # Remaining: C[0,1] = A[0,0]B[0,1] + A[0,1]B[1,1] + A[0,2]B[2,1]
    #            C[1,0] = A[1,0]B[0,0] + A[1,1]B[1,0] + A[1,2]B[2,0]
    #            C[2,0] = A[2,0]B[0,0] + A[2,1]B[1,0] + A[2,2]B[2,0]
    #            C[2,1] = A[2,0]B[0,1] + A[2,1]B[1,1] + A[2,2]B[2,1]
    
    # That's 6 terms for the "diagonal" blocks + 4 more output entries
    # Total would be 10 if done naively, but rank 7 exploits structure.
    
    # The key insight: use combinations to handle multiple outputs at once.
    
    # Standard rank-7 for 3x2 (from Lader's work):
    # t1: (A00, B00) -> C00
    # t2: (A01, B10) -> C00
    # t3: (A02, B20) -> C00
    # t4: (A10, B01) -> C11
    # t5: (A11, B11) -> C11
    # t6: (A12, B21) -> C11
    # t7: (A20+A10-A00, B00+B10-B20) -> (C20+C10-C00) ... this is getting complicated
    
    # Let me just use the schoolbook for 3x2 (rank 6) for each block, giving total 18.
    # That's already better than 54. But the task asks for rank-minimal.
    
    # Actually, for 3x2, the schoolbook rank is 3*3*2 = 18? No, it's 9*6=54 for the full problem.
    # For 3x2 alone: n=3, m=3, p=2, schoolbook rank = 3*3*2 = 18.
    # Three blocks: 3*18 = 54. That's the same as before!
    
    # Wait, I'm confusing myself. Let me reconsider.
    # The full problem: A is 3x3, B is 3x6.
    # Schoolbook: for each i in 0..2, j in 0..2, k in 0..5: one term. Total = 3*3*6 = 54.
    
    # If I decompose B into three 3x2 blocks, each 3x2 subproblem has:
    # A: 3x3, B_block: 3x2, C_block: 3x2
    # Schoolbook for 3x2: 3*3*2 = 18 terms per block, total 54.
    
    # To get rank 21, I need each 3x2 block to have rank 7.
    # 3 * 7 = 21.
    
    # The rank-7 decomposition for 3x2 matrix multiplication is known.
    # Let me look up the explicit construction.
    
    # Based on the standard result, the 7 terms for 3x2 are:
    # Using the notation where A[i,j], B[j,k], C[i,k]:
    
    def get_3x2_terms():
        """Return 7 rank-1 terms for 3x2 matrix multiplication.
        Each term is a tuple (A_entries, B_entries, C_entries) where each
        is a list of (linear_index, coefficient).
        A: 3x3 (indices 0..8), B: 3x2 (indices 0..5), C: 3x2 (indices 0..5)
        """
        # The rank-7 decomposition
        terms = []
        
        # t0: A[0,0]*B[0,0] -> C[0,0]
        terms.append(((0, 1), ((0, 1),), ((0, 1),)))
        # t1: A[0,1]*B[1,0] -> C[0,0]
        terms.append(((1, 1), ((2, 1),), ((0, 1),)))
        # t2: A[0,2]*B[2,0] -> C[0,0]
        terms.append(((2, 1), ((4, 1),), ((0, 1),)))
        # t3: A[1,0]*B[0,1] -> C[1,1]
        terms.append(((3, 1), ((1, 1),), ((1, 1),)))
        # t4: A[1,1]*B[1,1] -> C[1,1]
        terms.append(((4, 1), ((3, 1),), ((1, 1),)))
        # t5: A[1,2]*B[2,1] -> C[1,1]
        terms.append(((5, 1), ((5, 1),), ((1, 1),)))
        # t6: This term handles C[2,0], C[2,1], C[0,1], C[1,0]
        # A part: A[2,0] + A[2,1] + A[2,2] + A[0,0] + A[0,1] + A[0,2] + A[1,0] + A[1,1] + A[1,2]
        # B part: B[0,0] + B[1,0] + B[2,0] + B[0,1] + B[1,1] + B[2,1]
        # C part: C[2,0] + C[2,1] - C[0,1] - C[1,0] - C[0,0] - C[1,1]
        
        # This isn't quite right. Let me use a different approach.
        
        return None
    
    # I'll take a different approach: construct the decomposition and verify it numerically.
    
    # For now, let me use the schoolbook decomposition but optimized for the rectangular structure.
    # Actually, let me just return the schoolbook for safety, then optimize.
    
    # Schoolbook: rank = 54
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
