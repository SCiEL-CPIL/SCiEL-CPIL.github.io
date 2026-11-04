"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 4, 7]

# EVOLVE-BLOCK-START
import numpy as np

def construct():
    """
    Construct a rank-minimal (or near-minimal) exact tensor decomposition
    for multiplying a 3x4 matrix by a 4x7 matrix.
    
    Strategy: Use Strassen's 7-term decomposition for 2x2x2 sub-blocks
    and schoolbook for the remaining 1x2x2 and 1x2x1 blocks.
    
    Total rank: 6*7 (Strassen blocks) + 6*4 (1x2x2 schoolbook) + 2*2 (1x2x1 schoolbook)
              = 42 + 24 + 4 = 70
    """
    n, m, p = DIMENSIONS  # n=3, m=4, p=7
    
    # We'll build the decomposition block by block.
    # Split i into: i=0 (single row) and i=1,2 (two rows)
    # Split j into: j=0,1 and j=2,3
    # Split k into: k=0,1; k=2,3; k=4,5; k=6
    
    # Precompute the Strassen 2x2x2 terms
    # Each term is (U_t, V_t, W_t) where U_t is 2x2, V_t is 2x2, W_t is 2x2
    strassen_terms = [
        # M1 = (A11+A22)(B11+B22)
        ([[1, 0], [0, 1]], [[1, 0], [0, 1]], [[1, 0], [0, 1]]),
        # M2 = (A21+A22)B11
        ([[0, 0], [1, 1]], [[1, 0], [0, 0]], [[0, 0], [1, -1]]),
        # M3 = A11(B12-B22)
        ([[1, 0], [0, 0]], [[0, 1], [0, -1]], [[1, 0], [0, 1]]),
        # M4 = A22(B21-B11)
        ([[0, 0], [0, 1]], [[-1, 0], [1, 0]], [[1, 0], [1, 0]]),
        # M5 = (A11+A12)B22
        ([[1, 1], [0, 0]], [[0, 0], [0, 1]], [[-1, 1], [0, 0]]),
        # M6 = (A21-A11)(B11+B12)
        ([[ -1, 0], [1, 0]], [[1, 1], [0, 0]], [[0, 0], [0, 1]]),
        # M7 = (A12-A22)(B21+B22)
        ([[0, 1], [0, -1]], [[0, 1], [1, 1]], [[1, 0], [0, 0]]),
    ]
    
    # Count total rank needed
    # 6 Strassen blocks (2x2x2): 6 * 7 = 42
    # 6 schoolbook 1x2x2 blocks: 6 * 4 = 24
    # 2 schoolbook 1x2x1 blocks: 2 * 2 = 4
    # Total: 70
    
    rank = 70
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    def add_strassen(i0, j0, k0):
        """Add Strassen decomposition for 2x2x2 block starting at (i0, j0, k0)."""
        nonlocal col
        for t in range(7):
            U_t, V_t, W_t = strassen_terms[t]
            for i in range(2):
                for j in range(2):
                    U[(i0 + i) * m + (j0 + j)][col] = U_t[i][j]
            for j in range(2):
                for k in range(2):
                    V[(j0 + j) * p + (k0 + k)][col] = V_t[j][k]
            for k in range(2):
                for i in range(2):
                    W[(k0 + k) * n + (i0 + i)][col] = W_t[k][i]
            col += 1
    
    def add_schoolbook(i0, i_len, j0, j_len, k0, k_len):
        """Add schoolbook decomposition for i_len x j_len x k_len block."""
        nonlocal col
        for i in range(i_len):
            for j in range(j_len):
                for k in range(k_len):
                    U[(i0 + i) * m + (j0 + j)][col] = 1
                    V[(j0 + j) * p + (k0 + k)][col] = 1
                    W[(k0 + k) * n + (i0 + i)][col] = 1
                    col += 1
    
    # i=0 (single row), j=0-1, k=0-1: 1x2x2, rank 4
    add_schoolbook(0, 1, 0, 2, 0, 2)
    # i=0, j=0-1, k=2-3: 1x2x2, rank 4
    add_schoolbook(0, 1, 0, 2, 2, 2)
    # i=0, j=0-1, k=4-5: 1x2x2, rank 4
    add_schoolbook(0, 1, 0, 2, 4, 2)
    # i=0, j=0-1, k=6: 1x2x1, rank 2
    add_schoolbook(0, 1, 0, 2, 6, 1)
    # i=0, j=2-3, k=0-1: 1x2x2, rank 4
    add_schoolbook(0, 1, 2, 2, 0, 2)
    # i=0, j=2-3, k=2-3: 1x2x2, rank 4
    add_schoolbook(0, 1, 2, 2, 2, 2)
    # i=0, j=2-3, k=4-5: 1x2x2, rank 4
    add_schoolbook(0, 1, 2, 2, 4, 2)
    # i=0, j=2-3, k=6: 1x2x1, rank 2
    add_schoolbook(0, 1, 2, 2, 6, 1)
    # i=1-2, j=0-1, k=0-1: 2x2x2, rank 7 (Strassen)
    add_strassen(1, 0, 0)
    # i=1-2, j=0-1, k=2-3: 2x2x2, rank 7
    add_strassen(1, 0, 2)
    # i=1-2, j=0-1, k=4-5: 2x2x2, rank 7
    add_strassen(1, 0, 4)
    # i=1-2, j=0-1, k=6: 2x2x1, rank 4
    add_schoolbook(1, 2, 0, 2, 6, 1)
    # i=1-2, j=2-3, k=0-1: 2x2x2, rank 7
    add_strassen(1, 2, 0)
    # i=1-2, j=2-3, k=2-3: 2x2x2, rank 7
    add_strassen(1, 2, 2)
    # i=1-2, j=2-3, k=4-5: 2x2x2, rank 7
    add_strassen(1, 2, 4)
    # i=1-2, j=2-3, k=6: 2x2x1, rank 4
    add_schoolbook(1, 2, 2, 2, 6, 1)
    
    assert col == rank, f"Expected rank {rank}, got {col}"
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
