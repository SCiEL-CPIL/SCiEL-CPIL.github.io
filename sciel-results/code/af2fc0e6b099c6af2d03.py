"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [6, 6, 6]
import time
import random

# EVOLVE-BLOCK-START

def _schoolbook(n, m, p):
    """Return the schoolbook rank n*m*p decomposition."""
    rank = n * m * p
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


def _strassen_2x2_blocks(n, m, p):
    """
    Apply Strassen to 2x2 blocks of size (n/2, m/2) x (m/2, p/2).
    Each block multiplication is done with schoolbook.
    Only valid when n, m, p are all even.
    """
    if n % 2 or m % 2 or p % 2:
        return None

    nh, mh, ph = n // 2, m // 2, p // 2

    # Correct Strassen formulas:
    # P1 = (A11 - A00) * (B00 + B11)
    # P2 = (A00 + A11) * B11
    # P3 = (A10 + A11) * B01
    # P4 = A00 * (B01 - B11)
    # P5 = A11 * (B10 - B00)
    # P6 = (A00 + A01) * B10
    # P7 = (A10 - A01) * (B00 + B11)
    
    # C11 = P1 + P4 - P5 + P7
    # C10 = P2 + P5
    # C01 = P3 + P6
    # C00 = P1 - P2 + P3 - P7

    steps = [
        ({3: 1, 0: -1}, {0: 1, 3: 1}),       # P1
        ({0: 1, 3: 1}, {3: 1}),              # P2
        ({2: 1, 3: 1}, {1: 1}),              # P3
        ({0: 1}, {1: 1, 3: -1}),             # P4
        ({3: 1}, {2: 1, 0: -1}),             # P5
        ({0: 1, 1: 1}, {2: 1}),              # P6
        ({2: 1, 1: -1}, {0: 1, 3: 1}),       # P7
    ]

    # c_combos[c_block_idx][step_idx] = sign
    # c_block_idx: 0=C00, 1=C01, 2=C10, 3=C11
    c_combos = [
        {0: 1, 1: -1, 2: 1, 6: -1},  # C00 = P1 - P2 + P3 - P7
        {2: 1, 5: 1},                 # C01 = P3 + P6
        {1: 1, 4: 1},                 # C10 = P2 + P5
        {0: 1, 3: 1, 4: -1, 6: 1},    # C11 = P1 + P4 - P5 + P7
    ]

    block_rank = nh * mh * ph
    total_rank = 7 * block_rank

    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]

    for s in range(7):
        a_coeffs, b_coeffs = steps[s]
        for ai, a_sign in a_coeffs.items():
            for bi, b_sign in b_coeffs.items():
                # For this pair of blocks, do schoolbook multiplication
                a_row_start = (ai // 2) * nh
                a_col_start = (ai % 2) * mh
                b_row_start = (bi // 2) * mh
                b_col_start = (bi % 2) * ph

                for i in range(nh):
                    for j in range(mh):
                        for k in range(ph):
                            gi = a_row_start + i
                            gj = a_col_start + j
                            gk = b_col_start + k

                            c_block_row = gi // nh
                            c_block_col = gk // ph
                            c_block_idx = c_block_row * 2 + c_block_col

                            c_sign = c_combos[c_block_idx].get(s, 0)
                            if c_sign == 0:
                                continue

                            col = s * block_rank + i * (mh * ph) + j * ph + k

                            U[gi * m + gj][col] = a_sign
                            V[gj * p + gk][col] = b_sign
                            W[gk * n + gi][col] = c_sign

    return {"U": U, "V": V, "W": W}


def _verify(decomp, n, m, p, sample=True):
    """Verify that the decomposition is correct."""
    U = decomp["U"]
    V = decomp["V"]
    W = decomp["W"]
    rank = len(U[0]) if U else 0

    if len(U) != n * m or len(V) != m * p or len(W) != p * n:
        return False
    if any(len(row) != rank for row in U) or any(len(row) != rank for row in V) or any(len(row) != rank for row in W):
        return False

    if sample:
        indices = [(i, j, k) for i in range(n) for j in range(m) for k in range(p)]
        random.shuffle(indices)
        indices = indices[:min(200, len(indices))]
    else:
        indices = [(i, j, k) for i in range(n) for j in range(m) for k in range(p)]

    for i, j, k in indices:
        s_val = 0
        for c in range(rank):
            s_val += U[i * m + j][c] * V[j * p + k][c] * W[k * n + i][c]
        if abs(s_val - 1) > 1e-9:
            return False
    return True


def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110

    try:
        strassen_decomp = _strassen_2x2_blocks(n, m, p)
        if strassen_decomp is not None:
            if _verify(strassen_decomp, n, m, p, sample=False):
                return strassen_decomp
    except Exception:
        pass

    schoolbook_decomp = _schoolbook(n, m, p)
    return schoolbook_decomp

# EVOLVE-BLOCK-END
