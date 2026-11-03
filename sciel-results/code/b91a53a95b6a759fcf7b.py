import time
import sys

DIMENSIONS = [2, 4, 7]

_START_TIME = time.time()
TIME_LIMIT = 140.0  # Leave 10s margin before 150s kill

def _time_remaining():
    return TIME_LIMIT - (time.time() - _START_TIME)

def construct():
    """
    Constructs a rank-50 tensor decomposition for 2x4 by 4x7 matrix multiplication.
    Uses Strassen's algorithm for 2x2x2 blocks (rank 7) and schoolbook for 2x2x3 blocks (rank 12).
    Total: 4*7 + 2*12 = 28 + 24 = 52... wait.
    Let's re-calculate.
    A is 2x4, B is 4x7.
    Split A into A_L (2x2) and A_R (2x2).
    Split B into B_1 (2x7), B_2 (2x7)? No, B is 4x7.
    Split B by columns of A (rows of B).
    B_L1 (2x7) corresponds to A_L (rows 0,1 of B).
    B_R1 (2x7) corresponds to A_R (rows 2,3 of B).
    
    So C = A_L @ B_L1 + A_R @ B_R1.
    B_L1 is 2x7. We can split B_L1 into B_L1_1 (2x2), B_L1_2 (2x2), B_L1_3 (2x3).
    A_L is 2x2.
    C_L1_1 = A_L @ B_L1_1 (2x2 * 2x2) -> Strassen rank 7.
    C_L1_2 = A_L @ B_L1_2 (2x2 * 2x2) -> Strassen rank 7.
    C_L1_3 = A_L @ B_L1_3 (2x2 * 2x3) -> Schoolbook rank 12? Or can we do better?
    R(2,2,3) = ?
    Schoolbook is 2*2*3 = 12.
    Is there a better decomposition for 2x2x3?
    Yes, R(2,2,3) = 11 is known? Or 10?
    Actually, R(2,2,3) = 11 is a known bound.
    Let's check if we can use a better method for 2x2x3.
    Or maybe just stick to Strassen for 2x2x2 and schoolbook for the rest to be safe and simple.
    Current program uses schoolbook for 2x2x3 (rank 12).
    Total rank = 4 * 7 (for 2x2x2 blocks) + 2 * 12 (for 2x2x3 blocks) = 28 + 24 = 52.
    
    The previous score was 1.0769.
    The seed program (score 1.0) likely had rank 56 (schoolbook 2*4*7 = 56).
    52 is an improvement.
    Can we do better?
    
    What about R(2,2,3)?
    If R(2,2,3) = 11, then total rank = 28 + 22 = 50.
    If R(2,2,3) = 10, then total rank = 28 + 20 = 48.
    
    Let's try to find a rank 11 decomposition for 2x2x3.
    Or maybe use a different block partitioning.
    
    Alternative:
    Split A (2x4) into 2x2 blocks.
    Split B (4x7) into 2x7 blocks.
    This is what we did.
    
    What if we split B (4x7) into 4x2, 4x2, 4x3?
    Then A (2x4) is multiplied by 4x2, 4x2, 4x3.
    A @ (4x2) -> 2x2 result.
    A @ (4x3) -> 2x3 result.
    This is the same structure.
    
    Let's look for R(2,2,3).
    A known decomposition for 2x2x3 with rank 11:
    There is a paper "The rank of 2x2x3 matrix multiplication" or similar.
    Actually, R(2,2,3) = 11 is established.
    Schoolbook is 12.
    So we can save 1 rank per 2x2x3 block.
    We have two such blocks (C_L1_3 and C_R1_3).
    So we save 2 ranks.
    New rank = 52 - 2 = 50.
    
    Let's try to construct the rank 11 decomposition for 2x2x3.
    Or, we can just use the known result that R(2,2,3)=11.
    But I need the explicit factors.
    
    If I cannot find the explicit factors quickly, I should stick to the rank 52 solution which is already an improvement over the seed (56).
    The current program has rank 52.
    Score 1.0769.
    
    Wait, the inspiration 3 says "find a tensor decomposition with a rank lower than the schoolbook rank of 2*4*7".
    The seed program likely had rank 56.
    52 is better.
    
    Is it possible to get rank 48 or lower?
    R(2,4,7) is bounded by R(2,4,4) * R(2,4,3)? No.
    R(2,4,7) <= R(2,4,4) + R(2,4,3)?
    R(2,4,4) = R(2,2,4)^2? No.
    R(2,4,4) can be computed by splitting 2x4 * 4x4 into 2x2 * 2x4 blocks?
    A (2x4) * B (4x4).
    Split B into B1 (4x2), B2 (4x2).
    A * B1 (2x2), A * B2 (2x2).
    R(2,4,2) = ?
    A (2x4) * B (4x2) -> 2x2.
    Split A into A1 (2x2), A2 (2x2).
    A1 * B1 (2x2), A1 * B2 (2x2), A2 * B1 (2x2), A2 * B2 (2x2).
    Each 2x2*2x2 -> 2x2 has rank 7.
    So R(2,4,2) <= 4 * 7 = 28.
    Then R(2,4,4) <= 2 * 28 = 56? No, that's not right.
    R(2,4,4) is the rank of 2x4 * 4x4.
    Using the same logic:
    Split B (4x4) into B_L (4x2), B_R (4x2).
    C_L = A * B_L (2x2), C_R = A * B_R (2x2).
    R(2,4,2) is the rank of 2x4 * 4x2.
    Split A (2x4) into A_L (2x2), A_R (2x2).
    Split B_L (4x2) into B_L1 (2x2), B_L2 (2x2).
    C_L = A_L * B_L1 + A_R * B_L2.
    Each 2x2*2x2 -> 2x2 is rank 7.
    So R(2,4,2) <= 7 + 7 = 14.
    Then R(2,4,4) <= 14 + 14 = 28.
    
    Similarly, R(2,4,3):
    Split B (4x3) into B_L (4x2), B_R (4x1).
    C_L = A * B_L (2x2), C_R = A * B_R (2x1).
    R(2,4,2) <= 14.
    R(2,4,1) = ?
    A (2x4) * B (4x1) -> 2x1.
    Split A into A_L (2x2), A_R (2x2).
    Split B into B_L (2x1), B_R (2x1).
    C = A_L * B_L + A_R * B_R.
    Each 2x2 * 2x1 -> 2x1 has rank 4.
    So R(2,4,1) <= 4 + 4 = 8.
    Then R(2,4,3) <= 14 + 8 = 22.
    
    Total R(2,4,7) <= R(2,4,4) + R(2,4,3) = 28 + 22 = 50.
    
    This matches the previous calculation.
    So rank 50 is achievable.
    
    To achieve rank 50, I need:
    1. A rank 7 decomposition for 2x2x2 (Strassen).
    2. A rank 4 decomposition for 2x2x1 (Schoolbook).
    3. A rank 11 decomposition for 2x2x3.
    
    If I use schoolbook for 2x2x3 (rank 12), I get rank 52.
    If I use rank 11 for 2x2x3, I get rank 50.
    
    Let's try to find the rank 11 decomposition for 2x2x3.
    A = [[a00, a01], [a10, a11]]
    B = [[b00, b01, b02], [b10, b11, b12]]
    C = [[c00, c01, c02], [c10, c11, c12]]
    
    c00 = a00*b00 + a01*b10
    c01 = a00*b01 + a01*b11
    c02 = a00*b02 + a01*b12
    c10 = a10*b00 + a11*b10
    c11 = a10*b01 + a11*b11
    c12 = a10*b02 + a11*b12
    
    Schoolbook uses 12 mults.
    Strassen-like for 2x2x2 part + extra?
    If we view it as two 2x2x1 products:
    C_left (2x2) from B_left (2x2) and A.
    C_right (2x1) from B_right (2x1) and A.
    R(2,2,2) = 7.
    R(2,2,1) = 4.
    Total 11.
    
    So we can just decompose 2x2x3 into 2x2x2 + 2x2x1.
    C = A * [B_left, B_right] = [A*B_left, A*B_right].
    A*B_left is 2x2x2 (rank 7).
    A*B_right is 2x2x1 (rank 4).
    Total rank 11.
    
    This is much simpler! I don't need a special rank 11 formula.
    I just need to split the 2x2x3 block into a 2x2x2 block and a 2x2x1 block.
    
    So the strategy is:
    1. Split B (4x7) into B_1 (4x2), B_2 (4x2), B_3 (4x3).
    2. For each B_j, compute C_j = A * B_j.
    3. B_1 and B_2 are 4x2.
       Split A (2x4) into A_L (2x2), A_R (2x2).
       Split B_1 (4x2) into B_1_L (2x2), B_1_R (2x2).
       C_1 = A_L * B_1_L + A_R * B_1_R.
       Each 2x2 * 2x2 -> 2x2 is rank 7.
       So C_1 has rank 14.
       Similarly C_2 has rank 14.
    4. B_3 is 4x3.
       Split B_3 into B_3_L (4x2) and B_3_R (4x1).
       C_3_L = A * B_3_L (2x2). Rank 14 (same as above).
       C_3_R = A * B_3_R (2x1).
       Split A into A_L, A_R.
       Split B_3_R into B_3_R_L (2x1), B_3_R_R (2x1).
       C_3_R = A_L * B_3_R_L + A_R * B_3_R_R.
       Each 2x2 * 2x1 -> 2x1 is rank 4.
       So C_3_R has rank 8.
       Total for C_3 is 14 + 8 = 22.
       
    Total rank = 14 (for B_1) + 14 (for B_2) + 22 (for B_3) = 50.
    
    This is correct and simple to implement.
    
    Let's rewrite the program to use this decomposition.
    
    Steps:
    1. Define helper functions to add Strassen (2x2x2) and Schoolbook (2x2x1) blocks.
    2. Iterate through the blocks of B.
       - B_1 (cols 0,1): 4x2.
         - A_L * B_1_L (2x2 * 2x2) -> Strassen
         - A_R * B_1_R (2x2 * 2x2) -> Strassen
       - B_2 (cols 2,3): 4x2.
         - A_L * B_2_L (2x2 * 2x2) -> Strassen
         - A_R * B_2_R (2x2 * 2x2) -> Strassen
       - B_3 (cols 4,5,6): 4x3.
         - B_3_L (cols 4,5): 4x2.
           - A_L * B_3_L_L (2x2 * 2x2) -> Strassen
           - A_R * B_3_L_R (2x2 * 2x2) -> Strassen
         - B_3_R (col 6): 4x1.
           - A_L * B_3_R_L (2x2 * 2x1) -> Schoolbook
           - A_R * B_3_R_R (2x2 * 2x1) -> Schoolbook
    
    Total blocks:
    4 Strassen blocks (2x2x2) -> 4 * 7 = 28.
    2 Schoolbook blocks (2x2x1) -> 2 * 4 = 8.
    Wait, B_3_L is 4x2.
    A_L * B_3_L_L (2x2 * 2x2) -> Strassen.
    A_R * B_3_L_R (2x2 * 2x2) -> Strassen.
    So that's 2 more Strassen blocks.
    Total Strassen blocks: 2 (for B_1) + 2 (for B_2) + 2 (for B_3_L) = 6.
    6 * 7 = 42.
    Schoolbook blocks: 2 (for B_3_R).
    2 * 4 = 8.
    Total rank = 42 + 8 = 50.
    
    This matches.
    
    Let's code this up.
"""
    n, m, p = 2, 4, 7
    rank = 50

    if _time_remaining() < 5.0:
        # Fallback: return schoolbook rank-56 decomposition quickly
        U_fb = [[0] * 56 for _ in range(n * m)]
        V_fb = [[0] * 56 for _ in range(m * p)]
        W_fb = [[0] * 56 for _ in range(p * n)]
        col_fb = 0
        for i in range(n):
            for k in range(m):
                for j in range(p):
                    U_fb[i * m + k][col_fb] = 1
                    V_fb[k * p + j][col_fb] = 1
                    W_fb[j * n + i][col_fb] = 1
                    col_fb += 1
        return {"U": U_fb, "V": V_fb, "W": W_fb}

    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    col = 0
    
    def add_strassen_block(A_indices, B_indices, C_indices):
        nonlocal col
        # A: 2x2, B: 2x2, C: 2x2
        # Strassen 7 multiplications
        
        # M1 = (a00+a11)(b00+b11)
        # M2 = (a10+a11)b00
        # M3 = a00(b01-b11)
        # M4 = a11(b10-b00)
        # M5 = (a00+a01)b11
        # M6 = (a10-a00)(b00+b01)
        # M7 = (a01-a11)(b10+b11)
        
        # C00 = M1 + M4 - M5 + M7
        # C01 = M3 + M5
        # C10 = M2 + M4
        # C11 = M1 - M2 + M3 + M6
        
        # Local indices:
        # A: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
        # B: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
        # C: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
        
        def process_mk(A_coeffs, B_coeffs, C_coeffs):
            nonlocal col
            for local_a, coeff in A_coeffs.items():
                global_a = A_indices[local_a]
                U[global_a[0] * m + global_a[1]][col] += coeff
            for local_b, coeff in B_coeffs.items():
                global_b = B_indices[local_b]
                V[global_b[0] * p + global_b[1]][col] += coeff
            for local_c, coeff in C_coeffs.items():
                global_c = C_indices[local_c]
                W[global_c[1] * n + global_c[0]][col] += coeff
            col += 1

        # M1
        process_mk({0:1, 3:1}, {0:1, 3:1}, {0:1, 3:1})
        # M2
        process_mk({2:1, 3:1}, {0:1}, {2:1, 3:-1})
        # M3
        process_mk({0:1}, {1:1, 3:-1}, {1:1, 3:1})
        # M4
        process_mk({3:1}, {2:1, 0:-1}, {0:1, 2:1})
        # M5
        process_mk({0:1, 1:1}, {3:1}, {0:-1, 1:1})
        # M6
        process_mk({2:1, 0:-1}, {0:1, 1:1}, {3:1})
        # M7
        process_mk({1:1, 3:-1}, {2:1, 3:1}, {0:1})

    def add_schoolbook_221(A_indices, B_indices, C_indices):
        nonlocal col
        # A: 2x2, B: 2x1, C: 2x1
        # A: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
        # B: 0:(0,0), 1:(1,0)
        # C: 0:(0,0), 1:(1,0)
        
        # C00 = a00*b00 + a01*b10
        # C10 = a10*b00 + a11*b10
        
        # M1 = a00*b00
        # M2 = a01*b10
        # M3 = a10*b00
        # M4 = a11*b10
        
        def process_mult(A_idx, B_idx, C_idx, coeff=1):
            nonlocal col
            global_a = A_indices[A_idx]
            global_b = B_indices[B_idx]
            global_c = C_indices[C_idx]
            
            U[global_a[0] * m + global_a[1]][col] += coeff
            V[global_b[0] * p + global_b[1]][col] += coeff
            W[global_c[1] * n + global_c[0]][col] += coeff
            col += 1

        process_mult(0, 0, 0, 1)
        process_mult(1, 1, 0, 1)
        process_mult(2, 0, 1, 1)
        process_mult(3, 1, 1, 1)

    # A_L: (0,0), (0,1), (1,0), (1,1)
    A_L = [(0,0), (0,1), (1,0), (1,1)]
    # A_R: (0,2), (0,3), (1,2), (1,3)
    A_R = [(0,2), (0,3), (1,2), (1,3)]
    
    # B_1: cols 0,1. 4x2.
    # B_1_L: rows 0,1. 2x2.
    B_1_L = [(0,0), (0,1), (1,0), (1,1)]
    # B_1_R: rows 2,3. 2x2.
    B_1_R = [(2,0), (2,1), (3,0), (3,1)]
    # C_1: cols 0,1. 2x2.
    C_1 = [(0,0), (0,1), (1,0), (1,1)]
    
    add_strassen_block(A_L, B_1_L, C_1)
    add_strassen_block(A_R, B_1_R, C_1)
    
    # B_2: cols 2,3. 4x2.
    # B_2_L: rows 0,1. 2x2.
    B_2_L = [(0,2), (0,3), (1,2), (1,3)]
    # B_2_R: rows 2,3. 2x2.
    B_2_R = [(2,2), (2,3), (3,2), (3,3)]
    # C_2: cols 2,3. 2x2.
    C_2 = [(0,2), (0,3), (1,2), (1,3)]
    
    add_strassen_block(A_L, B_2_L, C_2)
    add_strassen_block(A_R, B_2_R, C_2)
    
    # B_3: cols 4,5,6. 4x3.
    # B_3_L: cols 4,5. 4x2.
    # B_3_L_L: rows 0,1. 2x2.
    B_3_L_L = [(0,4), (0,5), (1,4), (1,5)]
    # B_3_L_R: rows 2,3. 2x2.
    B_3_L_R = [(2,4), (2,5), (3,4), (3,5)]
    # C_3_L: cols 4,5. 2x2.
    C_3_L = [(0,4), (0,5), (1,4), (1,5)]
    
    add_strassen_block(A_L, B_3_L_L, C_3_L)
    add_strassen_block(A_R, B_3_L_R, C_3_L)
    
    # B_3_R: col 6. 4x1.
    # B_3_R_L: rows 0,1. 2x1.
    B_3_R_L = [(0,6), (1,6)]
    # B_3_R_R: rows 2,3. 2x1.
    B_3_R_R = [(2,6), (3,6)]
    # C_3_R: col 6. 2x1.
    C_3_R = [(0,6), (1,6)]
    
    add_schoolbook_221(A_L, B_3_R_L, C_3_R)
    add_schoolbook_221(A_R, B_3_R_R, C_3_R)

    # Verify col matches expected rank
    assert col == rank, f"Expected rank {rank}, got {col}"

    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
