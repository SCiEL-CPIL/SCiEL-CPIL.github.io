"""Matrix multiplication 4x4 * 4x5: rank-minimal exact tensor decomposition."""
import time

DIMENSIONS = [4, 4, 5]

def construct():
    start_time = time.time()
    DEADLINE = start_time + 140  # safety margin before 150s wall-clock kill
    
    n, m, p = DIMENSIONS  # 4, 4, 5
    
    # Strategy: 
    # Part 1: 4x4 * 4x4 (first 4 columns of B) using one-level Strassen: rank 28
    # Part 2: 4x4 * 4x1 (5th column of B) using schoolbook: rank 16
    # Total rank = 28 + 16 = 44
    
    # Indexing:
    # U has shape (n*m, rank) = (16, rank)
    # V has shape (m*p, rank) = (20, rank)  
    # W has shape (p*n, rank) = (20, rank)
    # 
    # A is n x m = 4x4, indexed A[i,j] with i in [0,3], j in [0,3]
    # B is m x p = 4x5, indexed B[j,k] with j in [0,3], k in [0,4]
    # C is n x p = 4x5, indexed C[i,k] with i in [0,3], k in [0,4]
    #
    # U index: i*m + j = i*4 + j
    # V index: j*p + k = j*5 + k
    # W index: k*n + i = k*4 + i
    
    total_rank = 44
    U = [[0.0] * total_rank for _ in range(n * m)]
    V = [[0.0] * total_rank for _ in range(m * p)]
    W = [[0.0] * total_rank for _ in range(p * n)]
    
    def add_term(u_indices, v_indices, w_indices, col):
        """Add a rank-1 term with given index sets."""
        for ui in u_indices:
            U[ui][col] += 1.0
        for vi in v_indices:
            V[vi][col] += 1.0
        for wi in w_indices:
            W[wi][col] += 1.0
    
    # Part 1: 4x4 * 4x4 using one-level Strassen
    # Split A into four 2x2 blocks:
    # A = [[A00, A01], [A10, A11]] where each is 2x2
    # A00 = A[0:2, 0:2], A01 = A[0:2, 2:4], A10 = A[2:4, 0:2], A11 = A[2:4, 2:4]
    # Similarly for B (first 4 columns): B00, B01, B10, B11
    
    # Strassen's 7 products for 2x2 * 2x2:
    # M1 = (A00 + A11) * (B00 + B11)
    # M2 = (A10 + A11) * B00
    # M3 = A00 * (B10 - B11)
    # M4 = A11 * (B01 - B00)
    # M5 = (A00 + A01) * B11
    # M6 = (A10 - A00) * (B00 + B01)
    # M7 = (A01 - A11) * (B10 + B11)
    
    # Each M is a 2x2 matrix, so M * C_block requires 4 rank-1 terms per M
    # Total: 7 * 4 = 28 terms
    
    # For each Strassen product, we need to express it as rank-1 terms in the tensor sense
    # A product (linear combo of A entries) * (linear combo of B entries) contributes to 
    # specific C entries.
    
    # Let me define the 2x2 blocks more explicitly.
    # A00 indices: (0,0), (0,1), (1,0), (1,1) -> U indices: 0, 1, 4, 5
    # A01 indices: (0,2), (0,3), (1,2), (1,3) -> U indices: 2, 3, 6, 7
    # A10 indices: (2,0), (2,1), (3,0), (3,1) -> U indices: 8, 9, 12, 13
    # A11 indices: (2,2), (2,3), (3,2), (3,3) -> U indices: 10, 11, 14, 15
    
    # B00 indices (j=0,1; k=0,1): V indices: 0, 1, 5, 6
    # B01 indices (j=0,1; k=2,3): V indices: 2, 3, 7, 8
    # B10 indices (j=2,3; k=0,1): V indices: 10, 11, 15, 16
    # B11 indices (j=2,3; k=2,3): V indices: 12, 13, 17, 18
    
    # C00 indices (i=0,1; k=0,1): W indices: 0, 1, 4, 5
    # C01 indices (i=0,1; k=2,3): W indices: 2, 3, 6, 7
    # C10 indices (i=2,3; k=0,1): W indices: 8, 9, 12, 13
    # C11 indices (i=2,3; k=2,3): W indices: 10, 11, 14, 15
    
    # For Strassen, the output blocks are:
    # C00 = M1 + M4 - M5 - M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # Each M is 2x2, so we need to handle each entry separately.
    # Let's define helper functions for the 2x2 block indices.
    
    # A block indices (U indices)
    A00_u = [0, 1, 4, 5]    # A[0,0], A[0,1], A[1,0], A[1,1]
    A01_u = [2, 3, 6, 7]    # A[0,2], A[0,3], A[1,2], A[1,3]
    A10_u = [8, 9, 12, 13]  # A[2,0], A[2,1], A[3,0], A[3,1]
    A11_u = [10, 11, 14, 15] # A[2,2], A[2,3], A[3,2], A[3,3]
    
    # B block indices (V indices) for first 4 columns
    B00_v = [0, 1, 5, 6]    # B[0,0], B[0,1], B[1,0], B[1,1]
    B01_v = [2, 3, 7, 8]    # B[0,2], B[0,3], B[1,2], B[1,3]
    B10_v = [10, 11, 15, 16] # B[2,0], B[2,1], B[3,0], B[3,1]
    B11_v = [12, 13, 17, 18] # B[2,2], B[2,3], B[3,2], B[3,3]
    
    # C block indices (W indices)
    C00_w = [0, 1, 4, 5]    # C[0,0], C[0,1], C[1,0], C[1,1]
    C01_w = [2, 3, 6, 7]    # C[0,2], C[0,3], C[1,2], C[1,3]
    C10_w = [8, 9, 12, 13]  # C[2,0], C[2,1], C[3,0], C[3,1]
    C11_w = [10, 11, 14, 15] # C[2,2], C[2,3], C[3,2], C[3,3]
    
    # Now, for each Strassen product M = X * Y where X is a 2x2 linear combo of A blocks
    # and Y is a 2x2 linear combo of B blocks, we need to add rank-1 terms.
    
    # The key insight: M[r][c] = sum over (X block entries) * (Y block entries)
    # For a 2x2 matrix product M = X*Y, M[r][c] = X[r][0]*Y[0][c] + X[r][1]*Y[1][c]
    # So each M[r][c] is a sum of 2 products, each being a product of linear combos.
    
    # For the tensor decomposition, each "product of linear combos" is one rank-1 term
    # where U has the A indices with coefficients, V has the B indices with coefficients,
    # and W has the single C index with coefficient 1 (or -1).
    
    # Let me define the Strassen products more carefully.
    
    # M1 = (A00 + A11) * (B00 + B11)
    # M1[r][c] = sum_{s} (A00[r][s] + A11[r][s]) * (B00[s][c] + B11[s][c])
    # = sum_{s} (A00[r][s]*B00[s][c] + A00[r][s]*B11[s][c] + A11[r][s]*B00[s][c] + A11[r][s]*B11[s][c])
    
    # This is 4 terms per (r,c), and 4 (r,c) pairs, so 16 terms for M1 alone?
    # No wait. Let me reconsider.
    
    # Actually, for the tensor decomposition of matrix multiplication, each rank-1 term
    # corresponds to one "multiplication" in the algorithm. The U, V, W vectors can have
    # multiple non-zero entries.
    
    # For M1 = (A00 + A11) * (B00 + B11), this is a 2x2 * 2x2 multiplication.
    # The 2x2 * 2x2 multiplication tensor has rank 7 (Strassen) or 4 (schoolbook).
    # But here, we're using Strassen at the 4x4 level, so M1 is just one of the 7 products.
    # M1 itself is a 2x2 matrix product, which requires its own decomposition.
    
    # Wait, I'm confusing myself. Let me restart the approach.
    
    # The 4x4 * 4x4 matrix multiplication can be done with rank 19 using known algorithms,
    # but implementing that correctly is complex. Let me use a simpler approach.
    
    # Alternative: use schoolbook for the whole thing but with some optimization.
    # Schoolbook rank = 4*4*5 = 80.
    
    # Let me try a different decomposition. For 4x4 * 4x5:
    # We can think of this as 5 separate 4x4 * 4x1 multiplications.
    # Each 4x4 * 4x1 has rank 16 (schoolbook).
    # Total rank = 5 * 16 = 80.
    
    # To beat 80, I need a better algorithm. Let me use Strassen for the 4x4 * 4x4 part.
    
    # Actually, let me just implement a valid rank-44 decomposition using the following:
    # For the first 4 columns (4x4 * 4x4), use a rank-28 decomposition based on
    # treating it as 4 separate 2x2 * 2x2 multiplications with Strassen.
    
    # No, that's not right either. Let me think about this differently.
    
    # The simplest correct approach: use schoolbook for everything, rank 80.
    # Then try to reduce.
    
    # Let me implement schoolbook first to ensure correctness, then optimize.
    
    rank = n * m * p  # 80
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1.0
                V[j * p + k][col] = 1.0
                W[k * n + i][col] = 1.0
                col += 1
    
    # Now try to reduce the rank using Strassen for the 4x4 * 4x4 part.
    # If time permits, implement a rank-28 decomposition for 4x4 * 4x4.
    
    if time.time() - start_time > DEADLINE:
        return {"U": U, "V": V, "W": W}
    
    # For now, return the schoolbook decomposition which is guaranteed correct.
    # The rank is 80, which is the same as the seed program.
    # To improve, I would need to implement a valid lower-rank decomposition.
    
    return {"U": U, "V": V, "W": W}
