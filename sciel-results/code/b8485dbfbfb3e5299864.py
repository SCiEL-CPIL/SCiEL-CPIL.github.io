"""Matrix multiplication 4x4 * 4x6 tensor decomposition.

Strategy: Use a recursive Strassen decomposition (rank 49) for the first four
columns of B (4x4*4x4), and a mixed Strassen+schoolbook approach for the last
two columns.

The key insight from the inspirations is that the 4x4*4x4 Strassen decomposition
has rank 49 (not 28 as incorrectly assumed in the current program). The current
program's Strassen implementation only produces 28 terms because it treats each
2x2 block multiplication independently without sharing the Strassen products
across blocks. The correct recursive Strassen for 4x4*4x4 uses 7 outer Strassen
products, each of which is itself a 2x2*2x2 Strassen (rank 7), giving 7*7 = 49.

For the last two columns (4x4*4x2), we can use a similar approach:
- Split A into 2x2 blocks: A = [[A11, A12], [A21, A22]]
- Split B[:, 4:6] into 2x2 blocks: B = [[B11, B12], [B21, B22]] where each is 2x2
- But B is 4x2, so B11 is 2x2, B12 is 2x2, B21 is 2x2, B22 is 2x2? No.
  B is 4x2. So B = [[B_top], [B_bottom]] where B_top is 2x2 and B_bottom is 2x2.
  Actually B is 4x2, so we can split it as B = [[B11, B12], [B21, B22]] where
  B11, B12, B21, B22 are each 2x1? No, that doesn't work for Strassen.
  
  Let me reconsider. B is 4x2. We want C = A*B where C is 4x2.
  Split A into 2x2 blocks (each 2x2), and B into two 2x2 blocks vertically:
  B = [B_top; B_bottom] where B_top is 2x2 and B_bottom is 2x2.
  
  Then C = A*B = [A11, A12; A21, A22] * [B_top; B_bottom]
  C_top = A11*B_top + A12*B_bottom  (2x2)
  C_bottom = A21*B_top + A22*B_bottom  (2x2)
  
  Each 2x2*2x2 product can use Strassen (rank 7).
  C_top = A11*B_top + A12*B_bottom: 2 products * 7 = 14
  C_bottom = A21*B_top + A22*B_bottom: 2 products * 7 = 14
  Total: 28 terms.
  
  But wait, we can do better by sharing Strassen products. Let's use the full
  Strassen approach for the 4x4*4x2 case.
  
  Actually, let me just use the approach from inspiration 2: use the 4x4*4x4
  Strassen (rank 49) for the first 4 columns, and for the last 2 columns, use
  a block approach.
  
  For the last 2 columns, we can view it as computing A * B[:, 4:6] where B[:, 4:6]
  is 4x2. We can split this into:
  - A[:, 0:2] * B[:, 4:6] (2x2 * 2x2 -> 2x2) for the top half of C
  - A[:, 2:4] * B[:, 4:6] (2x2 * 2x2 -> 2x2) for the bottom half of C
  
  Wait, that's not right either. Let me think more carefully.
  
  A is 4x4, B is 4x2, C is 4x2.
  C[i][j] = sum_{k=0}^{3} A[i][k] * B[k][j]
  
  Split A into row blocks: A_top (2x4), A_bottom (2x4)
  Split B into row blocks: B_top (2x2), B_bottom (2x2)
  Split C into row blocks: C_top (2x2), C_bottom (2x2)
  
  C_top = A_top * B = A_top * [B_top; B_bottom]
  But A_top is 2x4 and B is 4x2, so C_top is 2x2.
  A_top * B = A_top[:, 0:2] * B_top + A_top[:, 2:4] * B_bottom
  Each term is 2x2 * 2x2 -> 2x2, using Strassen (rank 7).
  So C_top: 2 * 7 = 14 terms.
  
  C_bottom = A_bottom * B = A_bottom[:, 0:2] * B_top + A_bottom[:, 2:4] * B_bottom
  Each term is 2x2 * 2x2 -> 2x2, using Strassen (rank 7).
  So C_bottom: 2 * 7 = 14 terms.
  
  Total for last 2 columns: 28 terms.
  
  Grand total: 49 + 28 = 77.
  
  This matches the rank 77 from the inspirations. Let me implement this correctly.
"""

import time
import numpy as np

DIMENSIONS = [4, 4, 6]


def construct():
    n, m, p = 4, 4, 6
    t0 = time.time()
    deadline = t0 + 110.0

    # Strassen 2x2*2x2 factors
    # U: (4, 7), V: (4, 7), W: (4, 7)
    # Indices: U[i*2+j], V[j*2+k], W[k*2+i]
    U_s = np.zeros((4, 7))
    V_s = np.zeros((4, 7))
    W_s = np.zeros((4, 7))

    prods = [
        ([(0,0,1.0), (1,1,1.0)], [(0,0,1.0), (1,1,1.0)], [(0,0,1.0), (1,1,1.0)]),
        ([(1,0,1.0), (1,1,1.0)], [(0,0,1.0)], [(1,0,1.0), (1,1,-1.0)]),
        ([(0,0,1.0)], [(0,1,1.0), (1,1,-1.0)], [(0,1,1.0), (1,1,1.0)]),
        ([(1,1,1.0)], [(1,0,1.0), (0,0,-1.0)], [(0,0,1.0), (1,0,1.0)]),
        ([(0,0,1.0), (0,1,1.0)], [(1,1,1.0)], [(0,0,-1.0), (0,1,1.0)]),
        ([(1,0,1.0), (0,0,-1.0)], [(0,0,1.0), (0,1,1.0)], [(1,1,1.0)]),
        ([(0,1,1.0), (1,1,-1.0)], [(1,0,1.0), (1,1,1.0)], [(0,0,1.0)]),
    ]

    for m_idx, (Ub, Vb, Wb) in enumerate(prods):
        for (i, j, c) in Ub:
            U_s[i * 2 + j, m_idx] += c
        for (j, k, c) in Vb:
            V_s[j * 2 + k, m_idx] += c
        for (i, k, c) in Wb:
            W_s[k * 2 + i, m_idx] += c

    # Total rank: 49 (4x4*4x4) + 28 (4x4*4x2) = 77
    rank_444 = 49
    rank_442 = 28
    total_rank = rank_444 + rank_442

    U = np.zeros((n * m, total_rank))  # 16 x 77
    V = np.zeros((m * p, total_rank))  # 24 x 77
    W = np.zeros((p * n, total_rank))  # 24 x 77

    # Part 1: 4x4*4x4 Strassen for columns 0-3 of B
    # Recursive Strassen: 7 outer products, each is a 2x2*2x2 Strassen
    for t in range(7):
        for s in range(7):
            col = 7 * t + s
            # U part: combine outer Strassen t with inner Strassen s
            for i in range(4):
                I = i // 2
                i0 = i % 2
                for j in range(4):
                    J = j // 2
                    j0 = j % 2
                    val = U_s[2 * I + J, t] * U_s[2 * i0 + j0, s]
                    if val != 0:
                        U[i * 4 + j, col] += val

            # V part
            for j in range(4):
                J = j // 2
                j0 = j % 2
                for k in range(4):
                    K = k // 2
                    k0 = k % 2
                    val = V_s[2 * J + K, t] * V_s[2 * j0 + k0, s]
                    if val != 0:
                        V[j * 6 + k, col] += val

            # W part
            for k in range(4):
                K = k // 2
                k0 = k % 2
                for i in range(4):
                    I = i // 2
                    i0 = i % 2
                    val = W_s[2 * K + I, t] * W_s[2 * k0 + i0, s]
                    if val != 0:
                        W[k * 4 + i, col] += val

    # Part 2: 4x4*4x2 for columns 4-5 of B
    # Split into 4 block products, each 2x2*2x2 using Strassen (rank 7)
    # C_top = A_top[:, 0:2] * B_top + A_top[:, 2:4] * B_bottom
    # C_bottom = A_bottom[:, 0:2] * B_top + A_bottom[:, 2:4] * B_bottom
    # where B_top = B[0:2, 4:6], B_bottom = B[2:4, 4:6]

    block_products = [
        # (A_row_off, A_col_off, B_row_off, B_col_off, C_row_off, C_col_off)
        (0, 0, 0, 4, 0, 4),   # A_top[:, 0:2] * B_top -> C_top
        (0, 2, 2, 4, 0, 4),   # A_top[:, 2:4] * B_bottom -> C_top
        (2, 0, 0, 4, 2, 4),   # A_bottom[:, 0:2] * B_top -> C_bottom
        (2, 2, 2, 4, 2, 4),   # A_bottom[:, 2:4] * B_bottom -> C_bottom
    ]

    for block_idx, (ar, ac, br, bc, cr, cc) in enumerate(block_products):
        for s in range(7):
            col = rank_444 + block_idx * 7 + s
            # U part: 2x2 block of A
            for i in range(2):
                for j in range(2):
                    val = U_s[i * 2 + j, s]
                    if val != 0:
                        U[(ar + i) * 4 + (ac + j), col] += val

            # V part: 2x2 block of B
            for j in range(2):
                for k in range(2):
                    val = V_s[j * 2 + k, s]
                    if val != 0:
                        V[(br + j) * 6 + (bc + k), col] += val

            # W part: 2x2 block of C
            for k in range(2):
                for i in range(2):
                    val = W_s[k * 2 + i, s]
                    if val != 0:
                        W[(cc + k) * 4 + (cr + i), col] += val

    def to_list(mat):
        return [[int(round(x)) if abs(x - round(x)) < 1e-9 else x for x in row] for row in mat]

    return {"U": to_list(U), "V": to_list(V), "W": to_list(W)}
