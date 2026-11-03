"""Matrix multiplication 3x4x8 - rank-72 decomposition using block Strassen."""
DIMENSIONS = [3, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 3, 4, 8
    
    # Strategy: 
    # - Row 0 of A (1x4): schoolbook with B (4x8) -> 32 terms
    # - Rows 1-2 of A (2x4): split B into 4 blocks of 2x2, use rank-10 for 2x2x2
    #   Actually: split A rows 1-2 into 2x2, B into 4 blocks of 2x2
    #   For each 2x2 A block and 2x2 B block: use Strassen (7 terms)
    #   But we need 2x4 x 4x8 -> split into 2x2 x 2x2 blocks
    #   A has 1 block of 2x2 (rows 1-2, cols 0-1) and 1 block of 2x2 (rows 1-2, cols 2-3)
    #   B has 4 blocks of 2x2
    #   So 2 A-blocks x 4 B-blocks = 8 sub-problems of 2x2x2
    #   Each uses 7 Strassen terms = 56 terms
    #   Total: 32 + 56 = 88 (same as before)
    
    # Better approach: Use a rank-10 decomposition for 2x4x8
    # Split A (2x4) into two 2x2 blocks: A_L (cols 0-1), A_R (cols 2-3)
    # Split B (4x8) into four 2x2 blocks: B_00, B_01, B_10, B_11 (rows 0-1/2-3, cols 0-1/2-3/4-5/6-7)
    # Actually B is 4x8, so split into 2 row-blocks x 4 col-blocks of 2x2 each
    
    # Let me use a different strategy:
    # For the 2x4 x 4x8 part (rows 1-2 of A):
    # Split into 2x2 x 2x2 blocks
    # A has 2 blocks of 2x2 (left: cols 0-1, right: cols 2-3)
    # B has 8 blocks of 2x2 (2 row-blocks x 4 col-blocks)
    # That's 2*8 = 16 sub-problems, each 2x2x2 with rank 7 = 112 terms. Too many.
    
    # Alternative: Use the fact that we can share Strassen factors
    # Let me try a direct rank-10 for 2x2x2 with some coupling
    
    # Actually, let me use a simpler but better approach:
    # Keep the 32 schoolbook terms for row 0
    # For rows 1-2, use a rank-56 decomposition (same as before)
    # But try to reduce by using half-integer coefficients
    
    # Let me try: rank 84 by using a 6-term decomposition for some 2x2x2 blocks
    # This is getting complex. Let me use a verified approach.
    
    # Final approach: rank 80
    # Use schoolbook for row 0: 32 terms
    # For rows 1-2: use 8 Strassen blocks (2 j-blocks x 4 k-blocks) = 56 terms
    # Total: 88. To improve, I need a better 2x4x8 decomposition.
    
    # Let me implement a rank-50 decomposition for 2x4x8:
    # Split A (2x4) into 2x2 blocks: A0 (cols 0-1), A1 (cols 2-3)
    # Split B (4x8) into 2x2 blocks: B[i][j] for i in {0,1} (row block), j in {0,1,2,3} (col block)
    # C = A0 * B[0][j] + A1 * B[1][j] for each j
    
    # For each j, we compute A0*B[0][j] + A1*B[1][j]
    # This is a 2x2 x 2x2 + 2x2 x 2x2 = 2x2 x 4x2 problem
    # We can use a rank-10 decomposition for 2x2x4x2... 
    
    # This is too complex to get right without verification. Let me use a safe improvement.
    
    # I'll use rank 84: replace 4 of the Strassen blocks with a 6-term decomposition
    # For 2x2x2, the minimum rank is 7 (Strassen). Can't do better for exact.
    
    # Let me try a completely different partition:
    # Split A into 3 blocks of 1x4
    # Row 0: schoolbook 1x4x8 = 32 terms
    # Row 1: use 4 terms (1x4x8 with structure) - not possible, minimum is 32
    # Row 2: same
    
    # OK, let me just implement the rank-88 correctly and add a small optimization.
    
    rank = 88
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Part 1: i=0, schoolbook 1x4x8
    for j in range(4):
        for k in range(8):
            U[0 * m + j][col] = 1
            V[j * p + k][col] = 1
            W[k * n + 0][col] = 1
            col += 1
            
    # Part 2: i=1,2 using Strassen on 2x2 blocks
    # A rows 1-2, split cols into [0,1] and [2,3]
    # B split into 4 blocks of 2x2: cols [0,1], [2,3], [4,5], [6,7]
    
    strassen_U = [
        [1, 0, 0, 1],
        [0, 0, 1, 1],
        [1, 0, 0, 0],
        [0, 0, 0, 1],
        [1, 1, 0, 0],
        [-1, 0, 1, 0],
        [0, 1, 0, -1]
    ]
    strassen_V = [
        [1, 0, 0, 1],
        [1, 0, 0, 0],
        [0, 1, 0, -1],
        [-1, 0, 1, 0],
        [0, 0, 0, 1],
        [1, 1, 0, 0],
        [0, 0, 1, 1]
    ]
    strassen_W = [
        [1, 0, 0, 1],
        [0, 0, 1, -1],
        [0, 1, 0, 1],
        [1, 0, 1, 0],
        [-1, 1, 0, 0],
        [0, 0, 0, 1],
        [1, 0, 0, 0]
    ]
    
    for j_block in range(2):
        j_start = j_block * 2
        for k_block in range(4):
            k_start = k_block * 2
            for t in range(7):
                for l in range(4):
                    if strassen_U[t][l] != 0:
                        li = l // 2
                        lj = l % 2
                        gi = 1 + li
                        gj = j_start + lj
                        U[gi * m + gj][col] += strassen_U[t][l]
                for l in range(4):
                    if strassen_V[t][l] != 0:
                        lj = l // 2
                        lk = l % 2
                        gj = j_start + lj
                        gk = k_start + lk
                        V[gj * p + gk][col] += strassen_V[t][l]
                for l in range(4):
                    if strassen_W[t][l] != 0:
                        li = l // 2
                        lk = l % 2
                        gi = 1 + li
                        gk = k_start + lk
                        W[gk * n + gi][col] += strassen_W[t][l]
                col += 1
                
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
