import numpy as np
import time

DIMENSIONS = [4, 4, 5]

def construct():
    n, m, p = DIMENSIONS  # 4, 4, 5
    deadline = time.time() + 130

    # Target tensor
    target = np.zeros((n * m, m * p, p * n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                target[a, b, c] = 1

    # Rank 64 construction
    U = np.zeros((n * m, 64))
    V = np.zeros((m * p, 64))
    W = np.zeros((p * n, 64))
    
    col = 0

    # Strassen 2x2x2 terms
    strassen_terms = [
        ([1,0,0,1], [1,0,0,1], [1,1,1,1]),
        ([0,0,1,1], [1,0,0,0], [0,1,0,0]),
        ([1,0,0,0], [0,1,0,-1], [0,0,1,0]),
        ([0,0,0,1], [-1,0,1,0], [0,1,0,0]),
        ([1,1,0,0], [0,0,0,1], [0,0,0,1]),
        ([-1,0,1,0], [1,1,0,0], [0,0,0,1]),
        ([0,1,0,-1], [0,0,1,1], [1,0,0,0])
    ]

    # Helper to map local 2x2 index to global 4x4 index
    def map_a(local_i, local_j, block_i, block_j):
        # block_i, block_j in {0, 1}
        # local_i, local_j in {0, 1}
        gi = block_i * 2 + local_i
        gj = block_j * 2 + local_j
        return gi * m + gj

    def map_b(local_i, local_j, block_i, block_j):
        gi = block_i * 2 + local_i
        gj = block_j * 2 + local_j
        return gi * p + gj

    def map_c(local_i, local_j, block_i, block_j):
        # local_i is row of C (i), local_j is col of C (k)
        # block_i, block_j in {0, 1}
        gi = block_i * 2 + local_i
        gj = block_j * 2 + local_j
        return gj * n + gi

    # Add Strassen product
    def add_strassen(a_block_i, a_block_j, b_block_i, b_block_j, c_block_i, c_block_j):
        nonlocal col
        for u, v, w in strassen_terms:
            for li in range(2):
                for lj in range(2):
                    a_idx = map_a(li, lj, a_block_i, a_block_j)
                    b_idx = map_b(li, lj, b_block_i, b_block_j)
                    c_idx = map_c(li, lj, c_block_i, c_block_j)
                    val = u[li*2+lj] * v[li*2+lj] * w[li*2+lj]
                    if val != 0:
                        U[a_idx, col] += val
                        V[b_idx, col] += val
                        W[c_idx, col] += val
            col += 1

    # Add schoolbook 2x2x1 product
    def add_schoolbook_2x2x1(a_block_i, a_block_j, b_block_i, b_block_j, c_block_i, c_block_j, c_col_offset):
        nonlocal col
        # A is 2x2, B is 2x1, C is 2x1
        for li in range(2):
            for lj in range(2):
                a_idx = map_a(li, lj, a_block_i, a_block_j)
                b_idx = map_b(li, 0, b_block_i, b_block_j) # B col 0
                c_idx = map_c(lj, 0, c_block_i, c_block_j) # C col 0
                U[a_idx, col] = 1
                V[b_idx, col] = 1
                W[c_idx, col] = 1
                col += 1

    # 8 Strassen products
    # A00*B00 -> C00
    add_strassen(0, 0, 0, 0, 0, 0)
    # A00*B01 -> C01
    add_strassen(0, 0, 0, 1, 0, 1)
    # A01*B10 -> C00
    add_strassen(0, 1, 1, 0, 0, 0)
    # A01*B11 -> C01
    add_strassen(0, 1, 1, 1, 0, 1)
    # A10*B00 -> C10
    add_strassen(1, 0, 0, 0, 1, 0)
    # A10*B01 -> C11
    add_strassen(1, 0, 0, 1, 1, 1)
    # A11*B10 -> C10
    add_strassen(1, 1, 1, 0, 1, 0)
    # A11*B11 -> C11
    add_strassen(1, 1, 1, 1, 1, 1)

    # 4 Schoolbook products (2x2 * 2x1)
    # A00*B02 -> C02
    add_schoolbook_2x2x1(0, 0, 0, 2, 0, 2, 0)
    # A01*B12 -> C02
    add_schoolbook_2x2x1(0, 1, 1, 2, 0, 2, 0)
    # A10*B02 -> C12
    add_schoolbook_2x2x1(1, 0, 0, 2, 1, 2, 0)
    # A11*B12 -> C12
    add_schoolbook_2x2x1(1, 1, 1, 2, 1, 2, 0)

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
