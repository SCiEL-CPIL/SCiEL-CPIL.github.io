"""Program for rank-minimal exact tensor decomposition of 3x5 * 5x7 matrix multiplication."""
DIMENSIONS = [3, 5, 7]

# EVOLVE-BLOCK-START
def _mul3x3(A, B):
    """Multiply two 3x3 matrices, returning the 3x3 result."""
    n = len(A)
    m = len(A[0])
    p = len(B[0])
    C = [[0] * p for _ in range(n)]
    for i in range(n):
        for j in range(m):
            aij = A[i][j]
            if aij:
                for k in range(p):
                    C[i][k] += aij * B[j][k]
    return C

def construct():
    n, m, p = DIMENSIONS  # 3, 5, 7

    # We'll build U, V, W as dense lists of lists
    # U has shape (n*m, rank), V has shape (m*p, rank), W has shape (p*n, rank)
    # We start with rank = n*m*p = 105 and use Strassen for the 3x3 block to reduce it.
    
    # Actually, let's use a different approach.
    # The schoolbook rank is 105.
    # Let's try to see if we can reduce it.
    
    # Let's implement the Strassen-like reduction for the 3x3 part.
    # Actually, the simplest way to get a valid decomposition is the schoolbook one.
    # But let's try to be smarter.
    
    # Let's use the block structure.
    # A is 3x5, B is 5x7.
    # Split A into [A1, A2] where A1 is 3x3 and A2 is 3x2.
    # Split B into [B1; B2] where B1 is 3x7 and B2 is 2x7.
    # C = A1*B1 + A2*B2.
    
    # A1*B1 is 3x3 * 3x7. Rank 63.
    # A2*B2 is 3x2 * 2x7. Rank 42.
    # Total rank 105.
    
    # Can we do better?
    # What if we split A into [A1, A2] where A1 is 3x2 and A2 is 3x3?
    # And B into [B1; B2] where B1 is 2x7 and B2 is 3x7?
    # C = A1*B1 + A2*B2.
    # A1*B1 is 3x2 * 2x7. Rank 42.
    # A2*B2 is 3x3 * 3x7. Rank 63.
    # Total rank 105.
    
    # What if we use Strassen for the 3x3 part?
    # A1*B1 where A1 is 3x3 and B1 is 3x3 (first 3 cols of B).
    # Strassen rank for 3x3 is 23 (instead of 27).
    # So A1*B1 rank 23 * 7 = 161? No, that's not right.
    # Strassen decomposes the 3x3x3 tensor.
    # The 3x3x7 tensor can be decomposed into 7 3x3x3 tensors.
    # So rank 7 * 23 = 161? No, that's worse.
    
    # Let's stick with the schoolbook decomposition for now.
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
