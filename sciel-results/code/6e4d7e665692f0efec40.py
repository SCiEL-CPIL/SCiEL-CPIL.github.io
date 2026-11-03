import numpy as np
import time

DIMENSIONS = [2, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # n=2, m=4, p=8
    # U: (n*m, rank) = (8, rank)
    # V: (m*p, rank) = (32, rank)
    # W: (p*n, rank) = (16, rank)
    
    # We use Strassen's algorithm for 2x2x2 multiplication with rank 7.
    # We split the problem into 8 sub-problems of 2x2 * 2x2 -> 2x2.
    # A (2x4) is split into A_L (2x2, cols 0-1) and A_R (2x2, cols 2-3).
    # B (4x8) is split into 8 blocks of 2x2:
    # B_T (rows 0-1) split into 4 blocks of 2x2 (cols 0-1, 2-3, 4-5, 6-7).
    # B_B (rows 2-3) split into 4 blocks of 2x2 (cols 0-1, 2-3, 4-5, 6-7).
    # C (2x8) is the sum of products.
    
    # Total rank = 8 * 7 = 56.
    
    rank = 56
    U = [[0.0] * rank for _ in range(n * m)] # 8 x 56
    V = [[0.0] * rank for _ in range(m * p)] # 32 x 56
    W = [[0.0] * rank for _ in range(p * n)] # 16 x 56
    
    # Correct Strassen decomposition vectors for M_{2,2,2}
    # T = sum_{r=1}^7 u_r v_r^T w_r^T
    # u_r, v_r, w_r are length 4 vectors corresponding to 2x2 matrices.
    # Indexing: 0:00, 1:01, 2:10, 3:11
    # Strassen products:
    # M1 = (a00+a11)(b00+b11)
    # M2 = (a10+a11)b00
    # M3 = a00(b10+b11)
    # M4 = a11(b01-b10)
    # M5 = (a00+a01)b11
    # M6 = (a10-a00)(b00+b01)
    # M7 = (a01-a11)(b10+b11)
    
    # Output:
    # C00 = M1 - M4 + M5 - M7
    # C01 = M3 + M7
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # Vectors (u, v, w):
    vecs = [
        # M1: u=(1,0,0,1), v=(1,0,0,1). w contributes 1 to C00, 1 to C11.
        ([1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1]),
        # M2: u=(0,0,1,1), v=(1,0,0,0). w contributes 1 to C10, -1 to C11.
        ([0, 0, 1, 1], [1, 0, 0, 0], [0, 0, 1, -1]),
        # M3: u=(1,0,0,0), v=(0,0,1,1). w contributes 1 to C01, 1 to C11.
        ([1, 0, 0, 0], [0, 0, 1, 1], [0, 1, 0, 1]),
        # M4: u=(0,0,0,1), v=(0,1,-1,0). w contributes -1 to C00, 1 to C10.
        ([0, 0, 0, 1], [0, 1, -1, 0], [-1, 0, 1, 0]),
        # M5: u=(1,1,0,0), v=(0,0,0,1). w contributes 1 to C00.
        ([1, 1, 0, 0], [0, 0, 0, 1], [1, 0, 0, 0]),
        # M6: u=(0,0,1,-1), v=(1,1,0,0). w contributes 1 to C11.
        ([0, 0, 1, -1], [1, 1, 0, 0], [0, 0, 0, 1]),
        # M7: u=(0,1,-1,0), v=(0,0,1,1). w contributes -1 to C00, 1 to C01, -1 to C11.
        ([0, 1, -1, 0], [0, 0, 1, 1], [-1, 1, 0, -1])
    ]
    
    def add_block(offset, a_map, b_map, c_map):
        """
        Adds the 7 rank-1 terms for one 2x2x2 multiplication block.
        a_map: list of 4 global indices in U for the 2x2 A block.
        b_map: list of 4 global indices in V for the 2x2 B block.
        c_map: list of 4 global indices in W for the 2x2 C block.
        """
        for r in range(7):
            col = offset + r
            u_vec, v_vec, w_vec = vecs[r]
            for l in range(4):
                if u_vec[l] != 0:
                    U[a_map[l]][col] = u_vec[l]
                if v_vec[l] != 0:
                    V[b_map[l]][col] = v_vec[l]
                if w_vec[l] != 0:
                    W[c_map[l]][col] = w_vec[l]

    # Define index maps
    # A is 2x4.
    # Row 0: 0, 1, 2, 3
    # Row 1: 4, 5, 6, 7
    # A_L (cols 0,1): a00->0, a01->1, a10->4, a11->5
    a_L = [0, 1, 4, 5]
    # A_R (cols 2,3): a00->2, a01->3, a10->6, a11->7
    a_R = [2, 3, 6, 7]
    
    # B is 4x8.
    # Row 0: 0-7
    # Row 1: 8-15
    # Row 2: 16-23
    # Row 3: 24-31
    
    # B_T blocks (rows 0,1)
    # b1 (cols 0,1): b00->0, b01->1, b10->8, b11->9
    b_T1 = [0, 1, 8, 9]
    # b2 (cols 2,3): b00->2, b01->3, b10->10, b11->11
    b_T2 = [2, 3, 10, 11]
    # b3 (cols 4,5): b00->4, b01->5, b10->12, b11->13
    b_T3 = [4, 5, 12, 13]
    # b4 (cols 6,7): b00->6, b01->7, b10->14, b11->15
    b_T4 = [6, 7, 14, 15]
    
    # B_B blocks (rows 2,3)
    # b5 (cols 0,1): b00->16, b01->17, b10->24, b11->25
    b_B1 = [16, 17, 24, 25]
    # b6 (cols 2,3): b00->18, b01->19, b10->26, b11->27
    b_B2 = [18, 19, 26, 27]
    # b7 (cols 4,5): b00->20, b01->21, b10->28, b11->29
    b_B3 = [20, 21, 28, 29]
    # b8 (cols 6,7): b00->22, b01->23, b10->30, b11->31
    b_B4 = [22, 23, 30, 31]
    
    # C is 2x8.
    # Convention: W[k*n+i, r]. k is col of C (0..7), i is row of C (0..1).
    # n=2.
    # Col 0: i=0->0, i=1->1
    # Col 1: i=0->2, i=1->3
    # Col 2: i=0->4, i=1->5
    # Col 3: i=0->6, i=1->7
    # Col 4: i=0->8, i=1->9
    # Col 5: i=0->10, i=1->11
    # Col 6: i=0->12, i=1->13
    # Col 7: i=0->14, i=1->15
    
    # C blocks (2x2 each, corresponding to 2 columns of C)
    # c1 (cols 0,1): c00->0, c01->1, c10->2, c11->3
    c_1 = [0, 1, 2, 3]
    # c2 (cols 2,3): c00->4, c01->5, c10->6, c11->7
    c_2 = [4, 5, 6, 7]
    # c3 (cols 4,5): c00->8, c01->9, c10->10, c11->11
    c_3 = [8, 9, 10, 11]
    # c4 (cols 6,7): c00->12, c01->13, c10->14, c11->15
    c_4 = [12, 13, 14, 15]
    
    # Add blocks
    # A_L * b_T1 -> c_1
    add_block(0, a_L, b_T1, c_1)
    # A_R * b_B1 -> c_1
    add_block(7, a_R, b_B1, c_1)
    
    # A_L * b_T2 -> c_2
    add_block(14, a_L, b_T2, c_2)
    # A_R * b_B2 -> c_2
    add_block(21, a_R, b_B2, c_2)
    
    # A_L * b_T3 -> c_3
    add_block(28, a_L, b_T3, c_3)
    # A_R * b_B3 -> c_3
    add_block(35, a_R, b_B3, c_3)
    
    # A_L * b_T4 -> c_4
    add_block(42, a_L, b_T4, c_4)
    # A_R * b_B4 -> c_4
    add_block(49, a_R, b_B4, c_4)
    
    # Convert to lists if necessary, but returning lists of lists is fine.
    # The problem statement says "integer/half-integer entries".
    # Our entries are -1, 0, 1.
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
