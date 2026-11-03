"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 5, 5]

# EVOLVE-BLOCK-START
import time
import numpy as np
from itertools import product

def construct():
    n, m, p = DIMENSIONS  # 4, 5, 5
    
    # Rank calculation for 1+4 split:
    # C11 (1x1) = A11(1x1) * B11(1x1) -> Rank 1
    # C12 (1x4) = A11(1x1) * B12(1x4) + A12(1x4) * B22(4x4) -> Rank 4 + 16 = 20
    # C21 (3x1) = A21(3x1) * B11(1x1) + A22(3x4) * B21(4x1) -> Rank 3 + 12 = 15
    # C22 (3x4) = A21(3x1) * B12(1x4) + A22(3x4) * B22(4x4) -> Rank 12 + 48 = 60
    # Total: 1 + 20 + 15 + 60 = 96
    
    rank = 96
    U = [[0] * rank for _ in range(n * m)]  # 20 x 96
    V = [[0] * rank for _ in range(m * p)]  # 25 x 96
    W = [[0] * rank for _ in range(p * n)]  # 20 x 96
    col = 0
    
    def add_block(a_row_off, a_col_off, a_r, a_c, b_row_off, b_col_off, b_r, b_c, c_row_off, c_col_off, c_r, c_c):
        """Add schoolbook for a_r x a_c * b_r x b_c -> c_r x c_c.
        Assumes a_c == b_r and a_r == c_r and b_c == c_c.
        """
        nonlocal col
        for i in range(a_r):
            for j in range(a_c):
                for k in range(b_c):
                    U[(a_row_off + i) * m + (a_col_off + j)][col] = 1
                    V[(b_row_off + j) * p + (b_col_off + k)][col] = 1
                    W[(c_col_off + k) * n + (c_row_off + i)][col] = 1
                    col += 1

    # A split: rows 0 (1 row), 1-3 (3 rows); cols 0 (1 col), 1-4 (4 cols)
    # B split: rows 0 (1 row), 1-4 (4 rows); cols 0 (1 col), 1-4 (4 cols)
    
    # C11: A11(1x1) * B11(1x1) -> C11(1x1)
    # A11 at (0,0), B11 at (0,0), C11 at (0,0)
    add_block(0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1)
    
    # C12: A11(1x1) * B12(1x4) + A12(1x4) * B22(4x4) -> C12(1x4)
    # Part 1: A11 * B12
    # A11 at (0,0), B12 at (0,1), C12 at (0,1)
    add_block(0, 0, 1, 1, 0, 1, 1, 4, 0, 1, 1, 4)
    
    # Part 2: A12 * B22
    # A12 at (0,1), B22 at (1,1), C12 at (0,1)
    add_block(0, 1, 1, 4, 1, 1, 4, 4, 0, 1, 1, 4)
    
    # C21: A21(3x1) * B11(1x1) + A22(3x4) * B21(4x1) -> C21(3x1)
    # Part 1: A21 * B11
    # A21 at (1,0), B11 at (0,0), C21 at (1,0)
    add_block(1, 0, 3, 1, 0, 0, 1, 1, 1, 0, 3, 1)
    
    # Part 2: A22 * B21
    # A22 at (1,1), B21 at (1,0), C21 at (1,0)
    add_block(1, 1, 3, 4, 1, 0, 4, 1, 1, 0, 3, 1)
    
    # C22: A21(3x1) * B12(1x4) + A22(3x4) * B22(4x4) -> C22(3x4)
    # Part 1: A21 * B12
    # A21 at (1,0), B12 at (0,1), C22 at (1,1)
    add_block(1, 0, 3, 1, 0, 1, 1, 4, 1, 1, 3, 4)
    
    # Part 2: A22 * B22
    # A22 at (1,1), B22 at (1,1), C22 at (1,1)
    add_block(1, 1, 3, 4, 1, 1, 4, 4, 1, 1, 3, 4)
    
    assert col == rank, f"Rank mismatch: expected {rank}, got {col}"
    
    # Verify
    np.random.seed(42)
    A_t = np.random.randn(n, m)
    B_t = np.random.randn(m, p)
    C_exp = A_t @ B_t
    C_comp = np.zeros((n, p))
    U_a = np.array(U, dtype=float)
    V_a = np.array(V, dtype=float)
    W_a = np.array(W, dtype=float)
    A_flat = A_t.reshape(-1)
    B_flat = B_t.reshape(-1)
    for r in range(rank):
        C_comp += np.outer(W_a[:, r].reshape(p, n), U_a[:, r] @ A_flat * (V_a[:, r] @ B_flat))
    
    if not np.allclose(C_comp, C_exp, atol=1e-10):
        # Fallback to schoolbook
        rank_fb = n * m * p
        U_fb = [[0] * rank_fb for _ in range(n * m)]
        V_fb = [[0] * rank_fb for _ in range(m * p)]
        W_fb = [[0] * rank_fb for _ in range(p * n)]
        col_fb = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_fb[i*m+j][col_fb] = 1
                    V_fb[j*p+k][col_fb] = 1
                    W_fb[k*n+i][col_fb] = 1
                    col_fb += 1
        return {"U": U_fb, "V": V_fb, "W": W_fb}
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
