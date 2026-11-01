import numpy as np

DIMENSIONS = [2, 3, 7]

def construct():
    n, m, p = DIMENSIONS
    
    # We model the matrix multiplication tensor for 2x3 * 3x7
    # The tensor T[a, b, c] = 1 if a = i*m + j, b = j*p + k, c = k*n + i
    # where i in [0, n), j in [0, m), k in [0, p)
    
    # We want to find a rank-minimal decomposition:
    # T[a, b, c] = sum_r U[a, r] * V[b, r] * W[c, r]
    
    # For small dimensions, we can try to find a lower rank than schoolbook (n*m*p = 42)
    # by exploiting structure.
    
    # Strategy: Since m=3 is small, we can group operations.
    # Let's try to find a decomposition with rank < 42 using a systematic approach.
    
    # First, let's try a simple block decomposition approach.
    # We can think of the 2x3 matrix as two 1x3 rows and the 3x7 matrix as three 1x7 rows.
    
    # For each pair of rows (one from A, one from B), we compute their dot product.
    # A has 2 rows, B has 3 rows, so we have 2*3 = 6 dot products of length 3.
    
    # Each dot product of length 3 can be done in rank 3 (schoolbook) or potentially less.
    # But we need to combine all 6 dot products into the final 2x7 matrix.
    
    # Let's try a direct approach: enumerate all possible low-rank decompositions
    # using integer/half-integer coefficients and check for exactness.
    
    # Given the complexity, let's start with a known lower bound and try to find
    # a decomposition with rank close to that.
    
    # For 2x3 * 3x7, the schoolbook rank is 42.
    # Let's try to find a decomposition with rank 36, 33, 30, etc.
    
    # We'll use a greedy approach: try to combine terms where possible.
    
    # Actually, let's try a more systematic approach using the fact that
    # we can use Strassen-like techniques for small blocks.
    
    # For now, let's return the schoolbook decomposition as a baseline,
    # but try to find improvements.
    
    # Let's try rank = 36 first by combining some terms.
    
    # The key insight: for each (i, j, k) triple, we have a term.
    # We can potentially combine terms that share the same a or b or c index.
    
    # Let's try a different approach: use the fact that the tensor has a special structure.
    
    # For 2x3 * 3x7, we can write:
    # C[i, k] = sum_j A[i, j] * B[j, k]
    
    # This is a sum of 3 terms for each (i, k) pair.
    # Total terms: 2 * 7 * 3 = 42 (schoolbook)
    
    # Can we do better? Let's try to use a rank-2 decomposition for the 3-term sum.
    # sum_j A[i,j] * B[j,k] = (A[i,0] + A[i,1]) * (B[0,k] + B[1,k]) - A[i,0]*B[1,k] - A[i,1]*B[0,k]
    # That's 4 terms instead of 3, which is worse.
    
    # Let's try: sum_j A[i,j] * B[j,k] = A[i,0]*B[0,k] + A[i,1]*B[1,k] + A[i,2]*B[2,k]
    # No obvious improvement.
    
    # Given the time constraint and the need for exactness, let's return the schoolbook
    # decomposition. We can try to improve this in future iterations.
    
    rank_schoolbook = n * m * p  # 42
    
    U = np.zeros((n * m, rank_schoolbook), dtype=float)
    V = np.zeros((m * p, rank_schoolbook), dtype=float)
    W = np.zeros((p * n, rank_schoolbook), dtype=float)
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                U[a, col] = 1
                V[b, col] = 1
                W[c, col] = 1
                col += 1
    
    # Convert to lists
    U_list = [list(map(int, row)) for row in U]
    V_list = [list(map(int, row)) for row in V]
    W_list = [list(map(int, row)) for row in W]
    
    return {"U": U_list, "V": V_list, "W": W_list}
