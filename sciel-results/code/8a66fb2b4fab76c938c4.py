"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [2, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    """Find rank-minimal exact tensor decomposition for 2x4 * 4x6 matrix multiplication."""
    n, m, p = DIMENSIONS
    # Current best known: 42
    rank = 42
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen coefficients for U (2x2 block of A)
    # Local indices: (i, j) where i in {0,1}, j in {0,1}
    U_coeffs = [
        {(0, 0): 1, (1, 1): 1},
        {(1, 0): 1, (1, 1): 1},
        {(0, 0): 1},
        {(1, 1): 1},
        {(0, 0): 1, (0, 1): 1},
        {(1, 0): 1, (0, 0): -1},
        {(0, 1): 1, (1, 1): -1}
    ]
    
    # Strassen coefficients for V (2x2 block of B)
    # Local indices: (j, k) where j in {0,1}, k in {0,1}
    V_coeffs = [
        {(0, 0): 1, (1, 1): 1},
        {(0, 0): 1},
        {(0, 1): 1, (1, 1): -1},
        {(1, 0): 1, (0, 0): -1},
        {(1, 1): 1},
        {(0, 0): 1, (0, 1): 1},
        {(1, 0): 1, (1, 1): 1}
    ]
    
    # Strassen coefficients for W (2x2 block of C)
    # Local indices: (i, k) where i in {0,1}, k in {0,1}
    W_coeffs = [
        {(0, 0): 1, (1, 1): 1},
        {(1, 0): 1, (1, 1): -1},
        {(0, 1): 1, (1, 1): 1},
        {(0, 0): 1, (1, 0): 1},
        {(0, 0): -1, (0, 1): 1},
        {(1, 1): 1},
        {(0, 0): 1}
    ]

    col = 0
    # Split A into 2 blocks of width 2
    # Split B into 2 blocks of height 2
    # Wait, B is 4x6. Rows are split into 2 blocks of height 2.
    # A is 2x4. Cols are split into 2 blocks of width 2.
    # This gives 2 * 2 = 4 blocks of (2x2) * (2x6)?
    # No, the inner dimension is 4.
    # A = [A0, A1] (2x2 each)
    # B = [B0; B1] (2x6 each)
    # C = A0 B0 + A1 B1
    # This is 2 independent 2x2 * 2x6 multiplications.
    # But the current code uses 3 blocks for B?
    # "for b in range(3)"
    # If B is split into 3 blocks, the width must be divisible by 3? 6/3=2.
    # So B blocks are 2x2?
    # If B is 4x6, and we split rows into 2 (height 2), and cols into 3 (width 2).
    # Then we have 2 * 3 = 6 blocks of (2x2) * (2x2).
    # Each block uses Strassen (rank 7).
    # Total rank 6 * 7 = 42.
    
    # So the structure is:
    # A is split into 2 column blocks of size 2x2. (a in 0..1)
    # B is split into 2 row blocks of size 2x6, and each 2x6 is split into 3 col blocks of size 2x2. (b in 0..2)
    # Wait, the loop is:
    # for a in range(2): # A block
    #   for b in range(3): # B col block
    #     for s in range(7): # Strassen
    # This implies B is treated as having 3 blocks of width 2.
    # And A is treated as having 2 blocks of width 2.
    # But the inner dimension must match.
    # A block a has columns 2a, 2a+1.
    # B block b has rows...?
    # In the code:
    # j = a * 2 + j_loc. This is the row index of B and col index of A.
    # So for a given a, we use rows 2a, 2a+1 of B.
    # And k = b * 2 + k_loc. This is the col index of B.
    # So for a given a, we use columns 2b, 2b+1 of B?
    # No, b iterates 0..2. So columns 0,1; 2,3; 4,5.
    # So for each a (2 choices), we use the corresponding 2 rows of B, and ALL 3 column blocks of B.
    # This means we compute A_a * B_{a, :} where B_{a, :} is the 2x6 submatrix.
    # And we decompose this 2x2 * 2x6 product into 3 independent 2x2 * 2x2 products?
    # Yes, because the columns of B are independent in the output C.
    # C_{i, k} depends on B_{j, k}.
    # So we can compute each 2-column slice of C independently.
    # Slice 0: C[:, 0:2] = A_a * B_{a, 0:2}
    # Slice 1: C[:, 2:4] = A_a * B_{a, 2:4}
    # Slice 2: C[:, 4:6] = A_a * B_{a, 4:6}
    # Each slice is 2x2 * 2x2, rank 7.
    # Total 3 slices * 2 A-blocks * 7 = 42.
    
    # This confirms the 42 rank structure.
    
    for a in range(2):
        for b in range(3):
            for s in range(7):
                # Update U
                for (i_loc, j_loc), coeff in U_coeffs[s].items():
                    i = i_loc
                    j = a * 2 + j_loc
                    u_idx = i * m + j
                    U[u_idx][col] += coeff
                
                # Update V
                for (j_loc, k_loc), coeff in V_coeffs[s].items():
                    j = a * 2 + j_loc
                    k = b * 2 + k_loc
                    v_idx = j * p + k
                    V[v_idx][col] += coeff
                
                # Update W
                for (i_loc, k_loc), coeff in W_coeffs[s].items():
                    i = i_loc
                    k = b * 2 + k_loc
                    w_idx = k * n + i
                    W[w_idx][col] += coeff
                
                col += 1

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
