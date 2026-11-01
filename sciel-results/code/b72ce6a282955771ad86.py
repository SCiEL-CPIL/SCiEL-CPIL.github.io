"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 10]

# EVOLVE-BLOCK-START
import time
import numpy as np

_start_time = time.time()
_DEADLINE = 120.0

def construct():
    """
    Rank-minimal exact tensor decomposition for 2x2 by 2x10 matrix multiplication.
    Uses block-diagonal structure: 5 independent 2x2 blocks, each with Strassen rank-7.
    Total rank = 35.
    """
    n, m, p = DIMENSIONS  # 2, 2, 10
    num_blocks = p // 2  # 5 blocks
    rank = num_blocks * 7  # 35

    U = np.zeros((n * m, rank), dtype=np.float64)   # 4 x 35
    V = np.zeros((m * p, rank), dtype=np.float64)   # 20 x 35
    W = np.zeros((p * n, rank), dtype=np.float64)   # 20 x 35

    # Strassen's 7 multiplications for 2x2 matrix multiplication
    # A = [[A00, A01], [A10, A11]], B = [[B00, B01], [B10, B11]]
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6

    # U row indices: A00=0, A01=1, A10=2, A11=3
    # V row indices for block b: B00=0*10+2b, B01=0*10+2b+1, B10=1*10+2b, B11=1*10+2b+1
    # W row indices for block b: C[0][2b]=(2b)*2+0, C[0][2b+1]=(2b+1)*2+0, C[1][2b]=(2b)*2+1, C[1][2b+1]=(2b+1)*2+1

    # Each term: (U_side, V_side, W_side)
    # U_side: list of (u_row, coeff)
    # V_side: list of (v_koff, coeff) where v_koff: 0=B00, 1=B01, 2=B10, 3=B11
    # W_side: list of (w_koff, coeff) where w_koff: 0=C[0][2b], 1=C[0][2b+1], 2=C[1][2b], 3=C[1][2b+1]

    strassen_decomp = [
        # M1 = (A00+A11) * (B00+B11)
        (
            [(0, 1), (3, 1)],       # U: A00 + A11
            [(0, 1), (3, 1)],       # V: B00 + B11
            [(0, 1), (3, 1)],       # W: C[0][2b]:1, C[1][2b+1]:1
        ),
        # M2 = (A10+A11) * B00
        (
            [(2, 1), (3, 1)],       # U: A10 + A11
            [(0, 1)],               # V: B00
            [(2, 1), (3, -1)],      # W: C[1][2b]:1, C[1][2b+1]:-1
        ),
        # M3 = A00 * (B01-B11)
        (
            [(0, 1)],               # U: A00
            [(1, 1), (3, -1)],      # V: B01 - B11
            [(1, 1), (3, 1)],       # W: C[0][2b+1]:1, C[1][2b+1]:1
        ),
        # M4 = A11 * (B10-B00)
        (
            [(3, 1)],               # U: A11
            [(2, 1), (0, -1)],      # V: B10 - B00
            [(0, 1), (2, 1)],       # W: C[0][2b]:1, C[1][2b]:1
        ),
        # M5 = (A00+A01) * B11
        (
            [(0, 1), (1, 1)],       # U: A00 + A01
            [(3, 1)],               # V: B11
            [(1, 1), (0, -1)],      # W: C[0][2b+1]:1, C[0][2b]:-1
        ),
        # M6 = (A10-A00) * (B00+B01)
        (
            [(2, 1), (0, -1)],      # U: A10 - A00
            [(0, 1), (1, 1)],       # V: B00 + B01
            [(3, 1)],               # W: C[1][2b+1]:1
        ),
        # M7 = (A01-A11) * (B10+B11)
        (
            [(1, 1), (3, -1)],      # U: A01 - A11
            [(2, 1), (3, 1)],       # V: B10 + B11
            [(0, 1)],               # W: C[0][2b]:1
        ),
    ]

    for b in range(num_blocks):
        base_col = b * 7
        for t in range(7):
            col = base_col + t
            u_side, v_side, w_side = strassen_decomp[t]

            for u_row, u_coeff in u_side:
                U[u_row, col] = u_coeff

            for v_koff, v_coeff in v_side:
                j = v_koff // 2
                k = 2 * b + v_koff % 2
                V[j * p + k, col] = v_coeff

            for w_koff, w_coeff in w_side:
                i = w_koff // 2
                k = 2 * b + w_koff % 2
                W[k * n + i, col] = w_coeff

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
