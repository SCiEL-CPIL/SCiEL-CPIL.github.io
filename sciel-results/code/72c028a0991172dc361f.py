"""
Matrix multiplication 6x6x6 using Strassen at the 2x2 block level (3x3 blocks)
with schoolbook rank-27 decomposition for each 3x3 block product.

The 6x6 matrix is partitioned into 4 blocks of size 3x3:
  A = [[A00, A01], [A10, A11]]
  B = [[B00, B01], [B10, B11]]
  C = [[C00, C01], [C10, C11]]

Strassen gives 7 virtual products:
  P1 = (A00+A11)(B00+B11)
  P2 = (A10+A11)B00
  P3 = A00(B01-B11)
  P4 = A11(B10-B00)
  P5 = (A00+A01)B11
  P6 = (A10-A00)(B00+B01)
  P7 = (A01-A11)(B10+B11)

Output:
  C00 = P1 + P4 - P5 + P7
  C01 = P3 + P5
  C10 = P2 + P4
  C11 = P1 - P2 + P3 + P6

Each 3x3 product uses schoolbook rank-27 decomposition.
Total rank = 7 * 27 = 189.

Index conventions:
  U[i*m+k, r] = coefficient of A[i,k] in bilinear form r
  V[k*p+j, r] = coefficient of B[k,j] in bilinear form r
  W[j*n+i, r] = coefficient of C[i,j] in bilinear form r (trace convention)
"""

import time

DIMENSIONS = [6, 6, 6]

# EVOLVE-BLOCK-START

def _schoolbook_rank27_terms(bs=3):
    """
    Return the 27 terms of the schoolbook decomposition for bs x bs matrix multiplication.
    Each term is a tuple (u, v, w) where u, v, w are bs*bs-element lists.
    u[i*bs+k] = coefficient of A[i,k]
    v[k*bs+j] = coefficient of B[k,j]
    w[j*bs+i] = coefficient of C[i,j] (W convention)
    """
    terms = []
    for i in range(bs):
        for k in range(bs):
            for j in range(bs):
                u = [0] * (bs * bs)
                v = [0] * (bs * bs)
                w = [0] * (bs * bs)
                u[i * bs + k] = 1
                v[k * bs + j] = 1
                w[j * bs + i] = 1
                terms.append((u, v, w))
    return terms


def _verify_decomposition(terms, n):
    """
    Verify that the given terms form a valid decomposition of n x n matrix multiplication.
    For all i, j, k: sum_r u_r[i*n+k] * v_r[k*n+j] * w_r[j*n+i] = 1
    """
    for i in range(n):
        for k in range(n):
            for j in range(n):
                total = 0
                for (u, v, w) in terms:
                    total += u[i * n + k] * v[k * n + j] * w[j * n + i]
                if abs(total - 1) > 1e-10:
                    return False
    return True


def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin before 120s limit

    n, m, p = DIMENSIONS  # 6, 6, 6
    bs = 3  # block size

    # Schoolbook rank-27 for 3x3 blocks
    bclr_terms = _schoolbook_rank27_terms(bs)

    # Verify the decomposition
    if not _verify_decomposition(bclr_terms, bs):
        # Fallback: rebuild explicitly
        bclr_terms = []
        for i in range(bs):
            for k in range(bs):
                for j in range(bs):
                    u = [0] * (bs * bs)
                    v = [0] * (bs * bs)
                    w = [0] * (bs * bs)
                    u[i * bs + k] = 1
                    v[k * bs + j] = 1
                    w[j * bs + i] = 1
                    bclr_terms.append((u, v, w))

    rank_per_block = len(bclr_terms)  # 27

    # Strassen decomposition at the 2x2 block level
    # Block indices: 0=TL, 1=TR, 2=BL, 3=BR
    strassen_terms = [
        ([(0, 1), (3, 1)], [(0, 1), (3, 1)]),  # P1 = (A00+A11)(B00+B11)
        ([(2, 1), (3, 1)], [(0, 1)]),           # P2 = (A10+A11)B00
        ([(0, 1)], [(1, 1), (3, -1)]),          # P3 = A00(B01-B11)
        ([(3, 1)], [(2, 1), (0, -1)]),          # P4 = A11(B10-B00)
        ([(0, 1), (1, 1)], [(3, 1)]),           # P5 = (A00+A01)B11
        ([(2, 1), (0, -1)], [(0, 1), (1, 1)]),  # P6 = (A10-A00)(B00+B01)
        ([(1, 1), (3, -1)], [(2, 1), (3, 1)]),  # P7 = (A01-A11)(B10+B11)
    ]

    # Output reconstruction: which C blocks each P contributes to and with what sign
    output_contrib = [
        [(0, 1), (3, 1)],    # P1 -> C00(+1), C11(+1)
        [(2, 1), (3, -1)],   # P2 -> C10(+1), C11(-1)
        [(1, 1), (3, 1)],    # P3 -> C01(+1), C11(+1)
        [(0, 1), (2, 1)],    # P4 -> C00(+1), C10(+1)
        [(0, -1), (1, 1)],   # P5 -> C00(-1), C01(+1)
        [(3, 1)],            # P6 -> C11(+1)
        [(0, 1)],            # P7 -> C00(+1)
    ]

    total_rank = 7 * rank_per_block

    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]

    col = 0
    for term_idx in range(7):
        a_comb, b_comb = strassen_terms[term_idx]
        out_comb = output_contrib[term_idx]

        for (u_local, v_local, w_local) in bclr_terms:
            # For this bilinear form, accumulate contributions
            for i in range(bs):
                for k in range(bs):
                    u_coeff = u_local[i * bs + k]
                    if u_coeff == 0:
                        continue
                    for j in range(bs):
                        v_coeff = v_local[k * bs + j]
                        if v_coeff == 0:
                            continue
                        w_coeff = w_local[j * bs + i]
                        if w_coeff == 0:
                            continue

                        # U contributions: A entries
                        for (blk_a, coeff_a) in a_comb:
                            row_a = (blk_a // 2) * bs
                            col_a = (blk_a % 2) * bs
                            u_idx = (row_a + i) * m + (col_a + k)
                            U[u_idx][col] += coeff_a * u_coeff

                        # V contributions: B entries
                        for (blk_b, coeff_b) in b_comb:
                            row_b = (blk_b // 2) * bs
                            col_b = (blk_b % 2) * bs
                            v_idx = (row_b + k) * p + (col_b + j)
                            V[v_idx][col] += coeff_b * v_coeff

                        # W contributions: C entries (W index = global_col * n + global_row)
                        for (blk_c, coeff_c) in out_comb:
                            row_c = (blk_c // 2) * bs
                            col_c = (blk_c % 2) * bs
                            w_idx = (col_c + j) * n + (row_c + i)
                            W[w_idx][col] += coeff_c * w_coeff

            # Increment column once per bilinear form
            col += 1

    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
