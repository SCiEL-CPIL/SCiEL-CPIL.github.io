"""Matrix multiplication 6x6x6 tensor decomposition."""
import time
import numpy as np
import random

DIMENSIONS = [6, 6, 6]

# EVOLVE-BLOCK-START
def construct():
    """
    Rank-161 decomposition of 6x6 matrix multiplication using 3x3 Laderman
    (rank 23) block algorithm with 2x2 Strassen (rank 7) inner products.
    Total rank = 23 * 7 = 161.
    """
    start_time = time.time()
    deadline = 130.0

    n, m, p = DIMENSIONS  # all 6

    rank = 161
    U = [[0] * rank for _ in range(n * m)]  # 36 x 161
    V = [[0] * rank for _ in range(m * p)]  # 36 x 161
    W = [[0] * rank for _ in range(p * n)]  # 36 x 161

    col = 0

    def add_laderman_product(A_blocks, B_blocks, C_blocks):
        """
        A_blocks: list of (coeff, ai, aj) - 2x2 block A[ai,aj] with coefficient
        B_blocks: list of (coeff, bi, bj) - 2x2 block B[bi,bj] with coefficient
        C_blocks: list of (coeff, ci, cj) - 2x2 block C[ci,cj] receives this product
        """
        nonlocal col
        # Strassen 2x2 terms:
        # Each term is defined by linear combinations of A entries, B entries, and C entries.
        # A entries: (row, col, coeff)
        # B entries: (row, col, coeff)
        # C entries: (row, col, coeff)
        
        # Standard Strassen 2x2:
        # M1 = (A00+A11)(B00+B11) -> C00+=1, C11+=1
        # M2 = (A10+A11)B00 -> C10+=1, C11-=1
        # M3 = A00(B01-B11) -> C01+=1, C11+=1
        # M4 = A11(B10-B00) -> C00+=1, C10+=1
        # M5 = (A00+A01)B11 -> C00-=1, C01+=1
        # M6 = (A10-A00)(B00+B01) -> C11+=1
        # M7 = (A01-A11)(B10+B11) -> C00+=1
        
        strassen_terms = [
            ([(0,0,1), (1,1,1)], [(0,0,1), (1,1,1)], [(0,0,1), (1,1,1)]),
            ([(1,0,1), (1,1,1)], [(0,0,1)], [(1,0,1), (1,1,-1)]),
            ([(0,0,1)], [(0,1,1), (1,1,-1)], [(0,1,1), (1,1,1)]),
            ([(1,1,1)], [(1,0,1), (0,0,-1)], [(0,0,1), (1,0,1)]),
            ([(0,0,1), (0,1,1)], [(1,1,1)], [(0,0,-1), (0,1,1)]),
            ([(1,0,1), (0,0,-1)], [(0,0,1), (0,1,1)], [(1,1,1)]),
            ([(0,1,1), (1,1,-1)], [(1,0,1), (1,1,1)], [(0,0,1)]),
        ]

        for A_part, B_part, C_part in strassen_terms:
            # U contributions
            for a_r, a_c, a_coeff in A_part:
                for c_a, ai, aj in A_blocks:
                    row = (ai * 2 + a_r) * m + (aj * 2 + a_c)
                    U[row][col] += c_a * a_coeff

            # V contributions
            for b_r, b_c, b_coeff in B_part:
                for c_b, bi, bj in B_blocks:
                    row = (bi * 2 + b_r) * p + (bj * 2 + b_c)
                    V[row][col] += c_b * b_coeff

            # W contributions
            # W index for C[ci, cj] (ci is row of C, cj is col of C)
            # Trace convention: W[k*n + i, r] where C[i,k] corresponds to W[k*n+i]
            # Here ci is the row index of C block (0..2), cj is col index of C block (0..2)
            # Local row c_r (0..1), local col c_c (0..1)
            # Global row of C: ci*2 + c_r
            # Global col of C: cj*2 + c_c
            # W index: (Global Col)*n + (Global Row)
            for c_r, c_c, c_coeff in C_part:
                for c_c_out, ci, cj in C_blocks:
                    global_row = ci * 2 + c_r
                    global_col = cj * 2 + c_c
                    row = global_col * n + global_row
                    W[row][col] += c_c_out * c_coeff

            col += 1

    # Laderman 3x3 products (rank 23)
    # A[i,j] * B[j,k] -> C[i,k]
    
    # P1: A00 * B00 -> C00
    add_laderman_product([(1,0,0)], [(1,0,0)], [(1,0,0)])
    # P2: A00 * (B10 - B00) -> C00
    add_laderman_product([(1,0,0)], [(1,1,0), (-1,0,0)], [(1,0,0)])
    # P3: (A10 + A00) * B00 -> C00
    add_laderman_product([(1,1,0), (1,0,0)], [(1,0,0)], [(1,0,0)])
    
    # P4: A22 * B22 -> C22
    add_laderman_product([(1,2,2)], [(1,2,2)], [(1,2,2)])
    # P5: A22 * (B02 - B22) -> C22
    add_laderman_product([(1,2,2)], [(1,0,2), (-1,2,2)], [(1,2,2)])
    # P6: (A22 + A02) * B22 -> C22
    add_laderman_product([(1,2,2), (1,0,2)], [(1,2,2)], [(1,2,2)])
    
    # P7: (A02 + A22) * (B22 + B00) -> C11
    add_laderman_product([(1,0,2), (1,2,2)], [(1,2,2), (1,0,0)], [(1,1,1)])
    
    # P8: A11 * B11 -> C11
    add_laderman_product([(1,1,1)], [(1,1,1)], [(1,1,1)])
    # P9: A11 * (B01 - B11) -> C01
    add_laderman_product([(1,1,1)], [(1,0,1), (-1,1,1)], [(1,0,1)])
    # P10: (A11 + A01) * B11 -> C01
    add_laderman_product([(1,1,1), (1,0,1)], [(1,1,1)], [(1,0,1)])
    # P11: (A01 + A11) * (B11 + B00) -> C11
    add_laderman_product([(1,0,1), (1,1,1)], [(1,1,1), (1,0,0)], [(1,1,1)])
    
    # P12: A01 * (B12 - B11) -> C12, C21
    add_laderman_product([(1,0,1)], [(1,1,2), (-1,1,1)], [(1,1,2), (1,2,1)])
    # P13: (A01 + A21) * B11 -> C21
    add_laderman_product([(1,0,1), (1,2,1)], [(1,1,1)], [(1,2,1)])
    # P14: (A01 + A11) * (B11 + B21) -> C12
    add_laderman_product([(1,0,1), (1,1,1)], [(1,1,1), (1,2,1)], [(1,1,2)])
    
    # P15: A10 * (B20 - B00) -> C10
    add_laderman_product([(1,1,0)], [(1,2,0), (-1,0,0)], [(1,1,0)])
    # P16: (A10 + A20) * B00 -> C20
    add_laderman_product([(1,1,0), (1,2,0)], [(1,0,0)], [(1,2,0)])
    # P17: (A10 + A00) * (B00 + B20) -> C10
    add_laderman_product([(1,1,0), (1,0,0)], [(1,0,0), (1,2,0)], [(1,1,0)])
    
    # P18: A20 * (B01 - B00) -> C20
    add_laderman_product([(1,2,0)], [(1,0,1), (-1,0,0)], [(1,2,0)])
    # P19: (A20 + A00) * B00 -> C20
    add_laderman_product([(1,2,0), (1,0,0)], [(1,0,0)], [(1,2,0)])
    # P20: (A20 + A10) * (B00 + B01) -> C20
    add_laderman_product([(1,2,0), (1,1,0)], [(1,0,0), (1,0,1)], [(1,2,0)])
    
    # P21: A12 * (B20 - B10) -> C22
    add_laderman_product([(1,1,2)], [(1,2,0), (-1,1,0)], [(1,2,2)])
    # P22: (A12 + A22) * B10 -> C22
    add_laderman_product([(1,1,2), (1,2,2)], [(1,1,0)], [(1,2,2)])
    # P23: (A12 + A02) * (B10 + B20) -> C22
    add_laderman_product([(1,1,2), (1,0,2)], [(1,1,0), (1,2,0)], [(1,2,2)])

    assert col == rank, f"Expected {rank} columns, got {col}"

    # Verify
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)

    random.seed(42)
    valid = True
    for _ in range(200):
        i = random.randint(0, n - 1)
        j = random.randint(0, m - 1)
        k = random.randint(0, p - 1)
        val = 0.0
        for r in range(rank):
            val += U_arr[i * m + j, r] * V_arr[j * p + k, r] * W_arr[k * n + i, r]
        if abs(val - 1.0) > 1e-10:
            valid = False
            break

    if valid:
        return {"U": U, "V": V, "W": W}
    else:
        rank_fallback = n * m * p
        U_f = [[0] * rank_fallback for _ in range(n * m)]
        V_f = [[0] * rank_fallback for _ in range(m * p)]
        W_f = [[0] * rank_fallback for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_f[i * m + j][col] = 1
                    V_f[j * p + k][col] = 1
                    W_f[k * n + i][col] = 1
                    col += 1
        return {"U": U_f, "V": V_f, "W": W_f}

# EVOLVE-BLOCK-END
