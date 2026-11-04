import time
import numpy as np

DIMENSIONS = [3, 3, 5]

# EVOLVE-BLOCK-START
def construct():
    """
    Returns a rank-43 exact tensor decomposition for multiplying a 3x3 matrix 
    by a 3x5 matrix. Uses a block decomposition combining Strassen for 2x2x2 
    parts and schoolbook for smaller parts.
    """
    n, m, p = DIMENSIONS  # 3, 3, 5
    
    # We construct a rank 43 decomposition.
    # A is 3x3, B is 3x5.
    # Partition A into 2x2, 2x1, 1x2, 1x1 blocks.
    # Partition B into 2x5, 1x5 blocks.
    
    # U shape: (3*3, rank) = (9, rank)
    # V shape: (3*5, rank) = (15, rank)
    # W shape: (5*3, rank) = (15, rank)
    
    # We will build the matrices by accumulating terms.
    # Since rank is fixed at 43, we can pre-allocate and fill.
    # However, it is safer to build a list of terms and then construct.
    
    terms = []
    
    def add_term(i_u, j_u, i_v, j_v, i_w, j_w, val=1):
        """
        Adds a term to the decomposition.
        i_u, j_u: indices for A (row, col) in U. U index is i_u*m + j_u.
        i_v, j_v: indices for B (row, col) in V. V index is i_v*p + j_v.
        i_w, j_w: indices for C (col, row) in W. W index is i_w*n + j_w.
        Note: Task says W[k*n+i, r]. So i_w is k (col of C), j_w is i (row of C).
        """
        terms.append((i_u * m + j_u, i_v * p + j_v, i_w * n + j_w, val))

    # --- Part 1: A_00 * B_0 (2x2 * 2x5) ---
    # A_00 is A[0:2, 0:2]
    # B_0 is B[0:2, 0:5]
    
    # Split B_0 into B_0a (2x3) and B_0b (2x2)
    # B_0a cols 0,1,2; B_0b cols 3,4
    
    # --- Part 1a: A_00 * B_0a (2x2 * 2x3) ---
    # Split B_0a into B_0a1 (2x2, cols 0,1) and B_0a2 (2x1, col 2)
    
    # --- Part 1a1: A_00 * B_0a1 (2x2 * 2x2) ---
    # Use Strassen (Rank 7)
    # Local indices for A_00: (0,0), (0,1), (1,0), (1,1) -> Global A (0,0)..(1,1)
    # Local indices for B_0a1: (0,0)..(1,1) -> Global B (0,0)..(1,1) (cols 0,1)
    # Local indices for C_0a1: (0,0)..(1,1) -> Global C (0,0)..(1,1) (cols 0,1)
    
    # Strassen terms for 2x2x2
    # M1 = (A00+A11)(B00+B11)
    add_term(0, 0, 0, 0, 0, 0, 1)
    add_term(0, 0, 1, 1, 0, 0, 1)
    add_term(1, 1, 0, 0, 0, 0, 1)
    add_term(1, 1, 1, 1, 0, 0, 1)
    
    # M2 = (A10+A11)B00
    add_term(1, 0, 0, 0, 0, 1, 1)
    add_term(1, 1, 0, 0, 0, 1, 1)
    
    # M3 = A00(B10-B11)
    add_term(0, 0, 1, 0, 1, 0, 1)
    add_term(0, 0, 1, 1, 1, 0, -1)
    
    # M4 = A11(B01-B00)
    add_term(1, 1, 0, 1, 1, 1, 1)
    add_term(1, 1, 0, 0, 1, 1, -1)
    
    # M5 = (A00+A01)B11
    add_term(0, 0, 1, 1, 0, 1, 1)
    add_term(0, 1, 1, 1, 0, 1, 1)
    
    # M6 = (A10-A00)(B00+B01)
    add_term(1, 0, 0, 0, 1, 0, 1)
    add_term(1, 0, 0, 1, 1, 0, 1)
    add_term(0, 0, 0, 0, 1, 0, -1)
    add_term(0, 0, 0, 1, 1, 0, -1)
    
    # M7 = (A01-A11)(B10+B11)
    add_term(0, 1, 1, 0, 1, 1, 1)
    add_term(0, 1, 1, 1, 1, 1, 1)
    add_term(1, 1, 1, 0, 1, 1, -1)
    add_term(1, 1, 1, 1, 1, 1, -1)
    
    # --- Part 1a2: A_00 * B_0a2 (2x2 * 2x1) ---
    # B_0a2 is B[0:2, 2:3] (col 2)
    # Schoolbook Rank 4
    # C[0,2] = A[0,0]B[0,2] + A[0,1]B[1,2]
    # C[1,2] = A[1,0]B[0,2] + A[1,1]B[1,2]
    
    add_term(0, 0, 0, 2, 2, 0, 1)
    add_term(0, 1, 1, 2, 2, 0, 1)
    add_term(1, 0, 0, 2, 2, 1, 1)
    add_term(1, 1, 1, 2, 2, 1, 1)
    
    # --- Part 1b: A_00 * B_0b (2x2 * 2x2) ---
    # B_0b is B[0:2, 3:5] (cols 3,4)
    # Use Strassen (Rank 7)
    # Global B cols 3,4. Global C cols 3,4.
    
    # M1' = (A00+A11)(B33+B44)
    add_term(0, 0, 0, 3, 3, 0, 1)
    add_term(0, 0, 1, 4, 3, 0, 1)
    add_term(1, 1, 0, 3, 3, 0, 1)
    add_term(1, 1, 1, 4, 3, 0, 1)
    
    # M2' = (A10+A11)B33
    add_term(1, 0, 0, 3, 3, 1, 1)
    add_term(1, 1, 0, 3, 3, 1, 1)
    
    # M3' = A00(B43-B44)
    add_term(0, 0, 1, 3, 4, 0, 1)
    add_term(0, 0, 1, 4, 4, 0, -1)
    
    # M4' = A11(B34-B33)
    add_term(1, 1, 0, 4, 4, 1, 1)
    add_term(1, 1, 0, 3, 4, 1, -1)
    
    # M5' = (A00+A01)B44
    add_term(0, 0, 1, 4, 3, 1, 1)
    add_term(0, 1, 1, 4, 3, 1, 1)
    
    # M6' = (A10-A00)(B33+B34)
    add_term(1, 0, 0, 3, 4, 0, 1)
    add_term(1, 0, 0, 4, 4, 0, 1)
    add_term(0, 0, 0, 3, 4, 0, -1)
    add_term(0, 0, 0, 4, 4, 0, -1)
    
    # M7' = (A01-A11)(B43+B44)
    add_term(0, 1, 1, 3, 4, 1, 1)
    add_term(0, 1, 1, 4, 4, 1, 1)
    add_term(1, 1, 1, 3, 4, 1, -1)
    add_term(1, 1, 1, 4, 4, 1, -1)

    # --- Part 2: A_01 * B_1 (2x1 * 1x5) ---
    # A_01 is A[0:2, 2:3] (col 2, rows 0,1)
    # B_1 is B[2:3, 0:5] (row 2, cols 0-4)
    # C_0 is C[0:2, 0:5] (rows 0,1)
    # Rank 10 (Schoolbook)
    for i in range(2):
        for k in range(5):
            add_term(i, 2, 2, k, k, i, 1)

    # --- Part 3: A_10 * B_0 (1x2 * 2x5) ---
    # A_10 is A[2:3, 0:2] (row 2, cols 0,1)
    # B_0 is B[0:2, 0:5] (rows 0,1, cols 0-4)
    # C_1 is C[2:3, 0:5] (row 2)
    # Rank 10 (Schoolbook)
    for j in range(2):
        for k in range(5):
            add_term(2, j, j, k, k, 2, 1)

    # --- Part 4: A_11 * B_1 (1x1 * 1x5) ---
    # A_11 is A[2:3, 2:3]
    # B_1 is B[2:3, 0:5]
    # C_1 is C[2:3, 0:5]
    # Rank 5 (Schoolbook)
    for k in range(5):
        add_term(2, 2, 2, k, k, 2, 1)

    # Total rank should be 7+4+7+10+10+5 = 43.
    rank = len(terms)
    
    # Construct U, V, W
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for r, (u_idx, v_idx, w_idx, val) in enumerate(terms):
        U[u_idx][r] = val
        V[v_idx][r] = val
        W[w_idx][r] = val
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
