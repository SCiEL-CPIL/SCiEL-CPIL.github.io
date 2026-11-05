"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 5, 5]

import time
import numpy as np

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin
    
    n, m, p = DIMENSIONS  # n=4, m=5, p=5
    
    # Strategy: Block decomposition using Strassen for 4x4x4 part
    # A (4x5) = [A4 (4x4) | A1 (4x1)]
    # B (5x5) = [B4 (4x5); B1 (1x5)]
    # C = A4*B4 + A1*B1
    # A4*B4: 4x4 * 4x4, use Strassen rank 19 (or better if known)
    # A1*B1: 4x1 * 1x5, schoolbook rank 20
    # Total: 19 + 20 = 39
    
    # However, we need to be careful: B is 5x5, split as [B_top (4x5); B_bot (1x5)]
    # A is 4x5, split as [A_left (4x4) | A_right (4x1)]
    # C = A_left * B_top + A_right * B_bot
    
    # A_left (4x4) * B_top (4x5): This is not a square multiplication.
    # We can split B_top into two 4x2 blocks and one 4x1 block? No, 5 = 2+2+1.
    # Let's stick to the previous successful structure from inspirations:
    # Split 5 = 4 + 1.
    # A (4x5) = [A4 (4x4) | A1 (4x1)]
    # B (5x5) = [B4 (4x5); B1 (1x5)]
    # C = A4 * B4 + A1 * B1
    
    # A4 * B4: 4x4 * 4x5.
    # We can split B4 into B4a (4x4) and B4b (4x1).
    # Then A4 * B4 = A4 * B4a + A4 * B4b.
    # A4 * B4a: 4x4 * 4x4. Use Strassen (rank 19? No, 2x2 Strassen is 7. 4x4 Strassen is 19? 
    #   Actually, recursive Strassen for 4x4: 7 * (2x2 mults) = 7*7=49? No.
    #   Strassen for n x n where n is power of 2: rank = 7^(log2(n/2) + 1)?
    #   For 2x2: 7.
    #   For 4x4: 7 * 7 = 49? No, that's if we use schoolbook for 2x2 inside.
    #   If we use Strassen recursively:
    #   4x4 = 7 * (2x2 multiplications). Each 2x2 is rank 7. Total rank 49.
    #   This is worse than schoolbook 64? No, 49 < 64.
    #   But can we do better?
    #   The inspiration 1 used a specific construction for 4x4x4 with rank 49.
    #   Wait, 2x2 Strassen is rank 7.
    #   4x4 Strassen uses 7 multiplications of 2x2 matrices.
    #   Each 2x2 multiplication is rank 7.
    #   So total rank is 7 * 7 = 49.
    #   Schoolbook for 4x4 is 16.
    #   So Strassen is worse for 4x4!
    #   
    #   Let's re-evaluate.
    #   2x2 schoolbook: rank 4. Strassen: rank 7.
    #   So for 2x2, schoolbook is better.
    #   For 4x4, schoolbook is 16. Strassen is 49.
    #   So we should use schoolbook for 4x4?
    #   
    #   Let's look at the dimensions again: 4x5 * 5x5.
    #   Schoolbook rank: 4*5*5 = 100.
    #   
    #   Let's try splitting 5 = 3 + 2.
    #   A (4x5) = [A3 (4x3) | A2 (4x2)]
    #   B (5x5) = [B3 (3x5); B2 (2x5)]
    #   C = A3*B3 + A2*B2
    #   A3*B3: 4x3 * 3x5. Schoolbook rank 4*3*5 = 60.
    #   A2*B2: 4x2 * 2x5. Schoolbook rank 4*2*5 = 40.
    #   Total: 100. No gain.
    #
    #   What if we use Strassen for the 2x2 parts?
    #   Split A2 (4x2) into two 2x2 blocks vertically? A2 = [A2a (2x2); A2b (2x2)]
    #   Split B2 (2x5) into... B2 is 2x5. Can't split rows into 2x2 easily.
    #
    #   Let's go back to the inspiration 1 approach which achieved score 1.1765 (rank 85).
    #   It used a 4x4x4 part with rank 49 and handled the remaining 5th dimension with schoolbook.
    #   49 + 16 + 16 + 4 = 85.
    #   Let's verify that construction.
    #   A (4x5) = [A4 (4x4) | A1 (4x1)]
    #   B (5x5) = [B4 (4x5); B1 (1x5)]
    #   C = A4*B4 + A1*B1
    #   A4*B4: 4x4 * 4x5.
    #   Split B4 (4x5) into B4a (4x4) and B4b (4x1).
    #   A4*B4a: 4x4 * 4x4.
    #   A4*B4b: 4x4 * 4x1.
    #   A1*B1: 4x1 * 1x5.
    #   
    #   If we use schoolbook for 4x4*4x4, rank is 16.
    #   4x4*4x1: rank 16.
    #   4x1*1x5: rank 20.
    #   Total: 16 + 16 + 20 = 52.
    #   This is much better than 100.
    #   Why did the inspiration use 49 for the 4x4x4 part?
    #   Maybe I am misinterpreting the dimensions or the Strassen rank.
    #   Wait, 4x4 schoolbook is 16. Strassen is 49. So schoolbook is better.
    #   So why did inspiration 1 use 49?
    #   Let's re-read inspiration 1.
    #   It says: "4x4x4 part: 49".
    #   And total rank 85.
    #   If I use schoolbook for 4x4x4, I get rank 16 for that part.
    #   Then total rank would be 16 + 16 + 20 = 52.
    #   Let's check if this decomposition is valid.
    #   A (4x5) = [A4 (4x4) | A1 (4x1)]
    #   B (5x5) = [B4 (4x5); B1 (1x5)]
    #   C = A4 * B4 + A1 * B1
    #   B4 (4x5) = [B4a (4x4) | B4b (4x1)]
    #   C = A4 * [B4a | B4b] + A1 * B1
    #   C = [A4*B4a + A1*B1_left | A4*B4b + A1*B1_right]
    #   Wait, A1 is 4x1, B1 is 1x5.
    #   A1*B1 is 4x5.
    #   A4*B4 is 4x5.
    #   So C = A4*B4 + A1*B1 is correct.
    #   A4*B4: A4 is 4x4, B4 is 4x5.
    #   We can decompose A4*B4 by splitting B4 into 4x4 and 4x1.
    #   A4*B4a (4x4*4x4) + A4*B4b (4x4*4x1).
    #   Schoolbook for 4x4*4x4: 16 terms.
    #   Schoolbook for 4x4*4x1: 16 terms.
    #   Schoolbook for 4x1*1x5: 20 terms.
    #   Total: 16 + 16 + 20 = 52.
    
    # Let's implement this rank 52 decomposition.
    
    rank = 52
    U = [[0] * rank for _ in range(n * m)]  # 20 x 52
    V = [[0] * rank for _ in range(m * p)]  # 25 x 52
    W = [[0] * rank for _ in range(p * n)]  # 20 x 52
    
    col = 0
    # Part 1: A4 * B4a (4x4 * 4x4)
    # A4 is rows 0-3, cols 0-3 of A (4x5). Index in U: i*5 + j
    # B4a is rows 0-3, cols 0-3 of B (5x5). Index in V: j*5 + k
    # Result goes to C[i,k] for i=0-3, k=0-3. Index in W: k*4 + i
    for i in range(4):
        for j in range(4):
            for k in range(4):
                U[i*5 + j][col] = 1
                V[j*5 + k][col] = 1
                W[k*4 + i][col] = 1
                col += 1
    
    # Part 2: A4 * B4b (4x4 * 4x1)
    # A4 is rows 0-3, cols 0-3 of A.
    # B4b is rows 0-3, col 4 of B.
    # Result goes to C[i,4] for i=0-3.
    for i in range(4):
        for j in range(4):
            U[i*5 + j][col] = 1
            V[j*5 + 4][col] = 1
            W[4*4 + i][col] = 1  # k=4, so index 4*4 + i = 16 + i
            col += 1
    
    # Part 3: A1 * B1 (4x1 * 1x5)
    # A1 is rows 0-3, col 4 of A.
    # B1 is row 4, cols 0-4 of B.
    # Result goes to C[i,k] for i=0-3, k=0-4.
    for i in range(4):
        for k in range(5):
            U[i*5 + 4][col] = 1
            V[4*5 + k][col] = 1
            W[k*4 + i][col] = 1
            col += 1
    
    # Check if col == rank
    if col != rank:
        # This should not happen
        pass
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
