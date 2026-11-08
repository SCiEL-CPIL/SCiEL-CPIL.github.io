"""Program for 5x5 matrix multiplication tensor decomposition."""
import time

DIMENSIONS = [5, 5, 5]
DEADLINE = time.time() + 140  # Safety margin before 150s kill

# EVOLVE-BLOCK-START
def construct():
    n, m, p = 5, 5, 5
    # We will use a block decomposition:
    # - Top-left 4x4 block: use a rank-56 decomposition (8 independent 2x2 Strassen products)
    # - Cross terms (involving row/col 4): use schoolbook
    # Total rank: 56 + 8*5 = 56 + 40 = 96
    # Wait, let me recount cross terms.
    # C = A*B where A, B are 5x5.
    # Partition into 4x4 block (indices 0-3) and last row/col (index 4).
    # C[0:4, 0:4] = A[0:4, 0:4] * B[0:4, 0:4]  (rank 56)
    # C[0:4, 4]   = A[0:4, 4] * B[4, 4]        (8 terms, rank 8? No, it's outer product)
    # Actually C[0:4, 4] is a 4x1 vector. A[0:4, 4] is 4x1, B[4, 4] is scalar.
    # So C[0:4, 4] = A[0:4, 4] * B[4, 4]. This is 4 multiplications. Rank 4.
    # Similarly C[4, 0:4] = A[4, 0:4] * B[0:4, 4]. 4x1 times 1x4. Rank 4? No.
    # C[4, j] = sum_k A[4, k] B[k, j]. For j=0..3, k=0..3. 
    # This is A[4, 0:4] * B[0:4, 0:4]. A[4, 0:4] is 1x4, B[0:4, 0:4] is 4x4.
    # So C[4, 0:4] is 1x4 vector. It involves 4*4 = 16 scalar multiplications.
    # But these are not independent of the 4x4 block.
    # Let's just do full schoolbook for all terms involving index 4 in A or B.
    # Terms:
    # C[i, j] = sum_k A[i, k] B[k, j]
    # If i < 4 and j < 4: k can be 0..3 (in 4x4 block) or k=4 (cross term).
    #   4x4 block handles k=0..3.
    #   Cross term for k=4: A[i, 4] * B[4, j]. There are 4*4=16 such terms.
    # If i < 4 and j = 4: k can be 0..3 or 4.
    #   k=0..3: A[i, k] * B[k, 4]. 4*4=16 terms.
    #   k=4: A[i, 4] * B[4, 4]. 4 terms.
    # If i = 4 and j < 4: k can be 0..3 or 4.
    #   k=0..3: A[4, k] * B[k, j]. 4*4=16 terms.
    #   k=4: A[4, 4] * B[4, j]. 4 terms.
    # If i = 4 and j = 4: k can be 0..3 or 4.
    #   k=0..3: A[4, k] * B[k, 4]. 4 terms.
    #   k=4: A[4, 4] * B[4, 4]. 1 term.
    # Total cross terms: 16 + 16 + 4 + 16 + 4 + 4 + 1 = 61? No, this is double counting.
    # Let's just list all (i, k, j) where i=4 or k=4 or j=4.
    # Total terms = 125. 4x4x4 = 64 terms are in the 4x4 block.
    # Remaining terms = 125 - 64 = 61.
    # So rank = 56 (for 4x4x4) + 61 (schoolbook for rest) = 117.
    # This is worse than 125. Bad idea.
    
    # Let's try a different approach.
    # Use Strassen for the whole 5x5? No, Strassen is for 2x2.
    # What about using the known rank-17 decomposition for 4x4? 
    # Actually, the rank of 4x4 matrix multiplication is known to be <= 49 (schoolbook) 
    # and better bounds exist. But implementing a rank-49 or lower decomposition is complex.
    
    # Let's stick to a simpler, correct approach.
    # The current seed is rank 125.
    # Can we do better with a simple block decomposition?
    # Split into 2x2 blocks? 5 = 2+2+1.
    # This gets complicated.
    
    # Alternative: Use the rank-25 decomposition for 3x3? No, 3x3 rank is 27.
    
    # Let's try to use Strassen-like ideas.
    # Actually, the simplest improvement is to use a known low-rank decomposition for 5x5.
    # The rank of 5x5 matrix multiplication is known to be 103 (or less).
    # But finding the exact decomposition is hard.
    
    # Let's just implement the schoolbook correctly.
    # Wait, the seed already does schoolbook with rank 125.
    # The score is 0.0000, which means it's invalid or not better.
    # Let's check the convention: W[k*n+i, r].
    # The seed uses W[k * n + i][column] = 1.
    # This matches W[k*n+i, r].
    
    # Why is the score 0.0000? Maybe the format is wrong.
    # "integer/half-integer entries" - the seed uses 0 and 1, which are integers.
    # "exact decomposition" - the seed should be exact.
    # Maybe the issue is the return type? "{'U', 'V', 'W'}: factor matrices of shapes (n*m, rank), (m*p, rank), (p*n, rank)"
    # The seed returns lists of lists. Maybe it needs to be numpy arrays?
    # Let's try returning numpy arrays.
    
    import numpy as np
    
    n, m, p = 5, 5, 5
    rank = n * m * p
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
