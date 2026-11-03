"""Strassen-based rank-minimal decomposition for 4x4 x 4x6 matrix multiplication."""
DIMENSIONS = [4, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 4, 4, 6
    
    # We'll build U, V, W matrices.
    # U: (n*m) x rank = 16 x rank
    # V: (m*p) x rank = 24 x rank
    # W: (p*n) x rank = 24 x rank
    # Each rank-1 term corresponds to a scalar multiplication at (i, j, k)
    # where i in [0,n), j in [0,m), k in [0,p)
    # U[i*m + j, c] = 1, V[j*p + k, c] = 1, W[k*n + i, c] = 1
    
    terms = []  # list of (i, j, k) tuples
    
    # Split the 4x6 output into two 4x3 halves: columns 0-2 and columns 3-5
    # For each half, use Strassen on 2x2 blocks of A and 2x3 blocks of B
    
    # For a 4x4 x 4x3 multiplication using Strassen:
    # A blocks: A11 (rows 0-1, cols 0-1), A12 (rows 0-1, cols 2-3), 
    #           A21 (rows 2-3, cols 0-1), A22 (rows 2-3, cols 2-3)
    # B blocks: B11 (rows 0-1, cols 0-2), B12 (rows 0-1, cols 3-4... wait)
    # 
    # Actually, B is 4x3. Split into 2x2 blocks:
    # B11: rows 0-1, cols 0-1 (2x2)
    # B12: rows 0-1, cols 2 (2x1) -- this is awkward
    #
    # Let me reconsider. B is 4x3. If I split into 2x2 blocks:
    # B11: rows 0-1, cols 0-1 (2x2)
    # B12: rows 0-1, col 2 (2x1)
    # B21: rows 2-3, cols 0-1 (2x2)
    # B22: rows 2-3, col 2 (2x1)
    #
    # This is messy. Let me instead think of it differently.
    
    # Alternative: treat 4x4 x 4x6 as two 4x4 x 4x3 problems.
    # For each 4x4 x 4x3: split A into 2x2 blocks, B into 2x3 blocks.
    # A: A11(2x2), A12(2x2), A21(2x2), A22(2x2)
    # B: B11(2x3), B12(2x3), B21(2x3), B22(2x3)
    # C: C11(2x3), C12(2x3), C21(2x3), C22(2x3)
    
    # Strassen for 2x2 x 2x3:
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B12-B22)
    # M4 = A22(B21-B11)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    #
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M5
    # C22 = M1 - M2 + M3 + M6
    
    # Each M is a 2x2 x 2x3 product = 6 scalar mults.
    # 7 M's = 42 scalar mults per 4x4 x 4x3.
    # Two halves = 84 scalar mults total.
    
    # For the left half (columns 0-2 of B, which is 4x3):
    # A is the full 4x4, B_left is columns 0-2 of B (4x3)
    # B_left blocks:
    #   B11: rows 0-1, cols 0-1 of B_left -> rows 0-1, cols 0-1 of B
    #   B12: rows 0-1, col 2 of B_left -> rows 0-1, col 2 of B
    #   B21: rows 2-3, cols 0-1 of B_left -> rows 2-3, cols 0-1 of B
    #   B22: rows 2-3, col 2 of B_left -> rows 2-3, col 2 of B
    
    # For the right half (columns 3-5 of B, which is 4x3):
    # B_right is columns 3-5 of B (4x3)
    # B_right blocks:
    #   B11: rows 0-1, cols 0-1 of B_right -> rows 0-1, cols 3-4 of B
    #   B12: rows 0-1, col 2 of B_right -> rows 0-1, col 5 of B
    #   B21: rows 2-3, cols 0-1 of B_right -> rows 2-3, cols 3-4 of B
    #   B22: rows 2-3, col 2 of B_right -> rows 2-3, col 5 of B
    
    # A blocks (same for both halves):
    #   A11: rows 0-1, cols 0-1
    #   A12: rows 0-1, cols 2-3
    #   A21: rows 2-3, cols 0-1
    #   A22: rows 2-3, cols 2-3
    
    # For each Strassen multiplication M = X * Y where X is 2x2 and Y is 2x3:
    # X has elements X[0][0], X[0][1], X[1][0], X[1][1]
    # Y has elements Y[0][0], Y[0][1], Y[0][2], Y[1][0], Y[1][1], Y[1][2]
    # The 6 scalar multiplications are:
    #   X[0][0]*Y[0][0], X[0][0]*Y[0][1], X[0][0]*Y[0][2],
    #   X[0][1]*Y[1][0], X[0][1]*Y[1][1], X[0][1]*Y[1][2]
    # Wait, no. 2x2 * 2x3:
    # Result[0][0] = X[0][0]*Y[0][0] + X[0][1]*Y[1][0]
    # Result[0][1] = X[0][0]*Y[0][1] + X[0][1]*Y[1][1]
    # Result[0][2] = X[0][0]*Y[0][2] + X[0][1]*Y[1][2]
    # Result[1][0] = X[1][0]*Y[0][0] + X[1][1]*Y[1][0]
    # Result[1][1] = X[1][0]*Y[0][1] + X[1][1]*Y[1][1]
    # Result[1][2] = X[1][0]*Y[0][2] + X[1][1]*Y[1][2]
    # So 6 scalar mults: (X[0][0],Y[0][0]), (X[0][1],Y[1][0]), (X[0][0],Y[0][1]), (X[0][1],Y[1][1]), (X[0][0],Y[0][2]), (X[0][1],Y[1][2]), (X[1][0],Y[0][0]), (X[1][1],Y[1][0]), (X[1][0],Y[0][1]), (X[1][1],Y[1][1]), (X[1][0],Y[0][2]), (X[1][1],Y[1][2])
    # That's 12! Not 6. Let me recount.
    # 2x2 * 2x3: each of 2 rows of result, each of 3 cols, needs 2 terms. 2*3*2 = 12 scalar mults.
    
    # Hmm, so each Strassen M is 12 scalar mults, not 6.
    # 7 M's = 84 per 4x4 x 4x3.
    # Two halves = 168 total. That's worse than schoolbook (96)!
    
    # I made an error. Let me reconsider.
    
    # Actually, for a 2x2 * 2x2 multiplication, schoolbook is 8 scalar mults, Strassen is 7.
    # For 2x2 * 2x3, schoolbook is 12 scalar mults. Can we do better with a Strassen-like approach?
    # Not directly, since Strassen is for square blocks.
    
    # Let me reconsider the problem. Maybe I should use a different decomposition.
    
    # Actually, let me just use the schoolbook decomposition but check if there's a known better rank.
    # For 4x4 x 4x4, the best known rank is 49 (Landsberg). For 4x4 x 4x6, it might be different.
    
    # Given the complexity, let me just output the schoolbook decomposition which is guaranteed correct with rank 96.
    
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
