import numpy as np

DIMENSIONS = [3, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    # We use the decomposition:
    # A (3x4) split into A1 (3x3) and A2 (3x1)
    # B (4x8) split into B1 (3x8) and B2 (1x8)
    # C = A1 B1 + A2 B2
    
    # Rank of A2 B2 (3x1 * 1x8) is 8.
    # Rank of A1 B1 (3x3 * 3x8) is computed recursively.
    # B1 (3x8) split into B11 (3x4) and B12 (3x4)
    # A1 B1 = A1 B11 + A1 B12
    # Rank of A1 B11 (3x3 * 3x4) is computed recursively.
    # B11 (3x4) split into B111 (3x2) and B112 (3x2)
    # A1 B11 = A1 B111 + A1 B112
    # Rank of A1 B111 (3x3 * 3x2) is computed recursively.
    # B111 (3x2) split into B1111 (3x1) and B1112 (3x1)
    # A1 B111 = A1 B1111 + A1 B1112
    # Rank of A1 B1111 (3x3 * 3x1) is 9.
    # Rank of A1 B1112 (3x3 * 3x1) is 9.
    # Total for A1 B111 is 18.
    
    # Wait, earlier I said R(3, 3, 2) = 14.
    # Let's re-evaluate R(3, 3, 2).
    # A (3x3), B (3x2).
    # Split B into B1 (3x1) and B2 (3x1).
    # C = A B1 + A B2.
    # R(3, 3, 1) = 9.
    # Total 18.
    # My previous 14 was based on R(3, 2, 2)=12 and R(3, 1, 2)=2.
    # But A is 3x3. Split A into A1 (3x2) and A2 (3x1).
    # B is 3x2. Split B into B1 (2x2) and B2 (1x2).
    # C = A1 B1 + A2 B2.
    # R(3, 2, 2) = 12.
    # R(3, 1, 2) = 2.
    # Total 14.
    # So R(3, 3, 2) = 14.
    
    # So R(3, 3, 4) = 2 * 14 = 28.
    # R(3, 3, 8) = 2 * 28 = 56.
    # R(3, 1, 8) = 8.
    # Total 64.
    
    # I will construct the matrices for this rank 64 decomposition.
    rank = 64
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Helper to add a rank-1 term
    def add_term(i, j, k, col):
        U[i * m + j][col] = 1
        V[j * p + k][col] = 1
        W[k * n + i][col] = 1
        
    col = 0
    # A2 B2 part: A2 is rows 0-2, cols 3. B2 is rows 3, cols 0-7.
    # C_ik = A_i3 B_3k
    # i in 0..2, k in 0..7
    for i in range(3):
        for k in range(8):
            add_term(i, 3, k, col)
            col += 1
    # col is now 8.
    
    # A1 B1 part: A1 is rows 0-2, cols 0-2. B1 is rows 0-2, cols 0-7.
    # Recursively decompose A1 B1.
    # B1 split into B11 (cols 0-3) and B12 (cols 4-7).
    # A1 B11 and A1 B12.
    # Each is 3x3 * 3x4.
    # R(3, 3, 4) = 28.
    
    # Let's implement the recursion for R(3, 3, 4).
    # It splits into two R(3, 3, 2).
    # R(3, 3, 2) splits into R(3, 2, 2) + R(3, 1, 2).
    
    # This is complex to implement generically.
    # I will hardcode the structure.
    
    # For now, return the schoolbook solution if the recursive construction is too complex.
    # But 64 is a significant improvement.
    
    # Let's just return schoolbook for safety if I can't write the recursive builder quickly.
    # But I can write a simple recursive builder.
    
    def build(n, m, p, offset_U, offset_V, offset_W, start_col):
        # Returns end_col
        if m == 1:
            # R(n, 1, p) = n * p
            for i in range(n):
                for k in range(p):
                    U[(offset_U + i) * m + offset_V][start_col] = 1 # Wait, indices are global
                    # This is getting messy with offsets.
                    pass
        return start_col

    # Given the complexity, I will stick to schoolbook to ensure correctness.
    # But I know 64 is possible.
    # I will try to implement the 64 rank solution.
    
    # Re-initialize for rank 64
    rank = 64
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    # Part 1: A2 B2 (3x1 * 1x8)
    for i in range(3):
        for k in range(8):
            U[i * m + 3][col] = 1
            V[3 * p + k][col] = 1
            W[k * n + i][col] = 1
            col += 1
            
    # Part 2: A1 B1 (3x3 * 3x8)
    # Split B1 into B11 (3x4) and B12 (3x4)
    # A1 B11 (3x3 * 3x4)
    # Split B11 into B111 (3x2) and B112 (3x2)
    # A1 B111 (3x3 * 3x2)
    # Split A1 into A11 (3x2) and A12 (3x1)
    # Split B111 into B1111 (2x2) and B1112 (1x2)
    # A11 B1111 (3x2 * 2x2) -> Rank 12
    # A12 B1112 (3x1 * 1x2) -> Rank 2
    # Total 14.
    
    # This is too detailed to hardcode without errors.
    # I will return the schoolbook solution.
    
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
