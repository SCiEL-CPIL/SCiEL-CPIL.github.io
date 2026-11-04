"""Matrix multiplication 2x4x4: rank-28 Strassen decomposition with time guard."""
import time

_DIMS = (2, 4, 4)
_T0 = time.time()
_LIMIT = 115  # seconds, safety margin before 120s kill


def _elapsed():
    return time.time() - _T0


def _time_ok(margin=3):
    return _elapsed() < _LIMIT - margin


def _strassen_2x2_patterns():
    """
    Return the 7 Strassen multiplication patterns for 2x2 matrix multiplication.

    For C = A*B where A,B,C are 2x2:
      A = [[a11, a12], [a21, a22]], B = [[b11, b12], [b21, b22]]

    Strassen's 7 products:
      M1 = (a11+a22)(b11+b22)
      M2 = (a21+a22)(b11)
      M3 = (a11)(b12-b22)
      M4 = (a22)(b21-b11)
      M5 = (a11+a12)(b22)
      M6 = (a21-a11)(b11+b12)
      M7 = (a12-a22)(b21+b22)

    C reconstruction:
      c11 = M1 + M4 - M5 + M7
      c12 = M3 + M5
      c21 = M2 + M4
      c22 = M1 - M2 + M3 + M6

    Each pattern is a dict with keys 'a', 'b', 'c' mapping local indices to coefficients.
    Local index convention: idx = row*2 + col, so:
      0=(0,0), 1=(0,1), 2=(1,0), 3=(1,1)
    """
    return [
        # M1 = (a11+a22)(b11+b22)
        {'a': {0: 1, 3: 1}, 'b': {0: 1, 3: 1}, 'c': {0: 1, 3: 1}},
        # M2 = (a21+a22)(b11)
        {'a': {2: 1, 3: 1}, 'b': {0: 1}, 'c': {2: 1, 3: -1}},
        # M3 = (a11)(b12-b22)
        {'a': {0: 1}, 'b': {1: 1, 3: -1}, 'c': {1: 1, 3: 1}},
        # M4 = (a22)(b21-b11)
        {'a': {3: 1}, 'b': {2: 1, 0: -1}, 'c': {0: 1, 2: 1}},
        # M5 = (a11+a12)(b22)
        {'a': {0: 1, 1: 1}, 'b': {3: 1}, 'c': {0: -1, 1: 1}},
        # M6 = (a21-a11)(b11+b12)
        {'a': {2: 1, 0: -1}, 'b': {0: 1, 1: 1}, 'c': {3: 1}},
        # M7 = (a12-a22)(b21+b22)
        {'a': {1: 1, 3: -1}, 'b': {2: 1, 3: 1}, 'c': {0: 1}},
    ]


def _construct_strassen_28():
    """
    Construct rank-28 tensor decomposition for 2x4 * 4x4 matrix multiplication.

    Splits A into two 2x2 blocks (A1=A[:,:2], A2=A[:,2:4]) and B into four 2x2 blocks.
    Each block product uses Strassen's 7-multiplication algorithm.
    Total rank: 4 * 7 = 28.

    Indexing:
      A[i,j]: i in {0,1}, j in {0..3}. U index: i*4+j
      B[j,k]: j in {0..3}, k in {0..3}. V index: j*4+k
      C[i,k]: i in {0,1}, k in {0..3}. W index: k*2+i (trace convention)
    """
    n, m, p = _DIMS  # 2, 4, 4
    rank = 28

    U = [[0] * rank for _ in range(n * m)]   # 8 x 28
    V = [[0] * rank for _ in range(m * p)]   # 16 x 28
    W = [[0] * rank for _ in range(p * n)]   # 8 x 28

    patterns = _strassen_2x2_patterns()

    # Four block products:
    #   (a_col_offset, b_row_offset, b_col_offset, c_col_offset)
    # A1*B11 -> C[:, 0:2]
    # A2*B21 -> C[:, 0:2]
    # A1*B12 -> C[:, 2:4]
    # A2*B22 -> C[:, 2:4]
    blocks = [
        (0, 0, 0, 0),
        (2, 2, 0, 0),
        (0, 0, 2, 2),
        (2, 2, 2, 2),
    ]

    col = 0
    for (a_cj, b_rj, b_ck, c_ck) in blocks:
        for pat in patterns:
            # Fill U: A entries. Local 2x2 index: lr*2+lc
            for li, coeff in pat['a'].items():
                lr = li // 2
                lc = li % 2
                gi = lr
                gj = a_cj + lc
                U[gi * m + gj][col] = coeff

            # Fill V: B entries. Local 2x2 index: lr*2+lc
            for li, coeff in pat['b'].items():
                lr = li // 2
                lc = li % 2
                gj = b_rj + lr
                gk = b_ck + lc
                V[gj * p + gk][col] = coeff

            # Fill W: C entries. Local 2x2 index: lr*2+lc
            for li, coeff in pat['c'].items():
                lr = li // 2
                lc = li % 2
                gi = lr
                gk = c_ck + lc
                W[gk * n + gi][col] = coeff

            col += 1

    return {"U": U, "V": V, "W": W}


def _construct_schoolbook():
    """Fallback rank-32 schoolbook decomposition."""
    n, m, p = _DIMS
    rank = n * m * p  # 32
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}


def construct():
    """
    Construct a rank-28 tensor decomposition for 2x4 * 4x4 matrix multiplication
    using Strassen's algorithm on four 2x2 block products.

    Returns dict with keys 'U', 'V', 'W':
      U: (8, 28) list of lists
      V: (16, 28) list of lists
      W: (8, 28) list of lists
    """
    if not _time_ok(margin=2):
        return _construct_schoolbook()

    return _construct_strassen_28()
