"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Rank 16 decomposition
    rank = 16
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # The 4 blocks of 2x2x2 multiplication
    # Block (bi, bj) corresponds to A block i and B block j
    # A is 2x4, so A blocks are 2x2.
    # B is 4x4, so B blocks are 2x2.
    # Output is 2x4.
    
    # We map the 4 products of each 2x2x2 block to the global indices.
    # For a 2x2x2 block, the 4 rank-1 terms are:
    # 1. A11 * B11 -> C11
    # 2. A12 * B22 -> C12 ? No.
    # Let's use the standard rank-4 decomposition for 2x2x2.
    # The terms are:
    # T1: A11, B11, C11
    # T2: A12, B22, C12 ? No.
    # Actually, the rank-4 decomposition of 2x2x2 is:
    # C11 = A11B11 + A12B21
    # C12 = A11B12 + A12B22
    # C21 = A21B11 + A22B21
    # C22 = A21B12 + A22B22
    # This uses 8 products.
    # However, the rank is 4.
    # The 4 products are:
    # M1 = A11 * B11
    # M2 = A12 * B22
    # M3 = A21 * B12
    # M4 = A22 * B21
    # And then:
    # C11 = M1 + (A12*B21) ? No.
    # I will use the 8-product decomposition for each block, which gives rank 32.
    # But the task asks for rank minimal.
    # If I can't find rank 16, I'll provide rank 32.
    # But rank 16 is the lower bound.
    # Let's try to construct rank 16.
    
    # The 4 blocks are:
    # Block 0: A11 (cols 0-1), B11 (rows 0-1, cols 0-1) -> C cols 0-1
    # Block 1: A11 (cols 0-1), B12 (rows 0-1, cols 2-3) -> C cols 2-3
    # Block 2: A21 (cols 2-3), B21 (rows 2-3, cols 0-1) -> C cols 0-1
    # Block 3: A21 (cols 2-3), B22 (rows 2-3, cols 2-3) -> C cols 2-3
    
    # Each block is a 2x2x2 multiplication.
    # Rank of 2x2x2 is 4.
    # Total rank 16.
    
    # The 4 terms for a 2x2x2 block (a, b, c):
    # a = [[a11, a12], [a21, a22]]
    # b = [[b11, b12], [b21, b22]]
    # c = [[c11, c12], [c21, c22]]
    # c11 = a11b11 + a12b21
    # c12 = a11b12 + a12b22
    # c21 = a21b11 + a22b21
    # c22 = a21b12 + a22b22
    
    # The rank-4 decomposition:
    # Term 1: a11, b11, c11
    # Term 2: a12, b22, c12 ? No.
    # I will use the 8-term decomposition to be safe, which is rank 32.
    # But I want to try rank 16.
    
    # Let's assume the rank-4 decomposition is:
    # 1. a11 * b11 -> c11
    # 2. a12 * b21 -> c11
    # 3. a11 * b12 -> c12
    # 4. a12 * b22 -> c12
    # 5. a21 * b11 -> c21
    # 6. a22 * b21 -> c21
    # 7. a21 * b12 -> c22
    # 8. a22 * b22 -> c22
    # This is 8 terms.
    
    # I will output rank 32 for now.
    rank = 32
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    for i in range(2): # A block row (0 or 1)
        for j in range(2): # B block row (0 or 1)
            for k in range(2): # B block col (0 or 1)
                # This is getting complicated.
                pass
    
    # Simple schoolbook
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
