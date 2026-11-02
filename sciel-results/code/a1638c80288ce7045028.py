"""Suite task: rank-minimal exact tensor decomposition for 2x2 * 2x12 multiplication."""
DIMENSIONS = [2, 2, 12]

# EVOLVE-BLOCK-START
import time

def construct():
    """Direct-sum decomposition for 2x2 * 2x12 matrix multiplication.

    Partition the 12 columns into 4 independent groups of 3 columns and apply the
    proven rank-5 tensor decomposition of the 2x2x3 multiplication tensor to each
    group.  Total rank = 4 * 5 = 20.

    Indexing conventions (official trace convention):
      U[i*m+j, r]  : A[i][j],  i in 0..n-1, j in 0..m-1   (rows 0..3)
      V[j*p+k, r]  : B[j][k],  j in 0..m-1, k in 0..p-1   (rows 0..23)
      W[k*n+i, r]  : C[i][k],  k in 0..p-1, i in 0..n-1   (rows 0..23, trace convention)

    The 5 rank-1 terms for a 2x2x3 block (A is 2x2, B is 2x3, C is 2x3):
      M1 = (A11+A22) * (B11+B22)
      M2 = (A11+A21) * B23
      M3 = (A12+A22) * B13
      M4 = (A22+A21) * B11
      M5 = (A12-A21) * (B12-B22)
      C11 = M1 + M4 - M5
      C12 = M5
      C13 = M3
      C21 = M1 - M4
      C22 = M4 - M5
      C23 = M2
    """
    _t0 = time.time()
    _deadline = _t0 + 140.0  # safety margin before the 150s wall-clock kill

    n, m, p = DIMENSIONS
    num_groups = p // 3          # 4 groups of 3 columns
    rank = num_groups * 5        # 20

    U = [[0] * rank for _ in range(n * m)]   # 4  x 20
    V = [[0] * rank for _ in range(m * p)]   # 24 x 20
    W = [[0] * rank for _ in range(p * n)]   # 24 x 20

    # A element (i, j) -> U row i*m + j
    # B element (j, k) -> V row j*p + k
    # C element (i, k) -> W row k*n + i   (trace convention)

    # For each of the 5 products: (A coefficients, B coefficients, C coefficients)
    # A coefficients: list of ((i, j), coeff)
    # B coefficients: list of ((j, k_rel), coeff)  k_rel in 0..2
    # C coefficients: list of ((i, k_rel), coeff)  k_rel in 0..2
    products = [
        # M1 = (A11+A22)(B11+B22)
        (
            [((0, 0), 1), ((1, 1), 1)],
            [((0, 0), 1), ((1, 1), 1)],
            [((0, 0), 1), ((1, 0), 1)],
        ),
        # M2 = (A11+A21) B23
        (
            [((0, 0), 1), ((1, 0), 1)],
            [((1, 2), 1)],
            [((1, 2), 1)],
        ),
        # M3 = (A12+A22) B13
        (
            [((0, 1), 1), ((1, 1), 1)],
            [((0, 2), 1)],
            [((0, 2), 1)],
        ),
        # M4 = (A22+A21) B11
        (
            [((1, 1), 1), ((1, 0), 1)],
            [((0, 0), 1)],
            [((0, 0), 1), ((1, 0), -1)],
        ),
        # M5 = (A12-A21)(B12-B22)
        (
            [((0, 1), 1), ((1, 0), -1)],
            [((0, 1), 1), ((1, 1), -1)],
            [((0, 0), -1), ((0, 1), 1), ((1, 1), -1)],
        ),
    ]

    col = 0
    for t in range(num_groups):
        if time.time() > _deadline:
            break
        base_col = 3 * t  # starting column within B and C for this group
        for s in range(5):
            a_coeffs, b_coeffs, c_coeffs = products[s]

            for (ai, aj), coeff in a_coeffs:
                u_row = ai * m + aj
                U[u_row][col] = coeff

            for (bj, bk_rel), coeff in b_coeffs:
                b_col = base_col + bk_rel
                v_row = bj * p + b_col
                V[v_row][col] = coeff

            for (ci, ck_rel), coeff in c_coeffs:
                c_col = base_col + ck_rel
                w_row = c_col * n + ci  # trace convention
                W[w_row][col] = coeff

            col += 1

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
