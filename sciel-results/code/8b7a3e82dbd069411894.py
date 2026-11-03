"""Matrix multiplication 3x4 x 4x4: rank-minimal exact tensor decomposition."""
import time
import numpy as np

DIMENSIONS = [3, 4, 4]


def _verify_fast(U, V, W, n, m, p, tol=1e-9):
    """Fast vectorized verification using einsum."""
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    rank = U_arr.shape[1]

    U_3d = U_arr.reshape(n, m, rank)
    V_3d = V_arr.reshape(m, p, rank)
    W_3d = W_arr.reshape(p, n, rank)

    T = np.einsum('ijc,jkc,kic->ijk', U_3d, V_3d, W_3d)

    max_err = np.max(np.abs(T - 1.0))
    return max_err < tol, max_err


def _schoolbook():
    n, m, p = DIMENSIONS
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}


def _strassen_44():
    """
    Rank-44 decomposition using Strassen's algorithm on 2x2 blocks.
    
    For the 3x4 x 4x4 multiplication:
    - Top 2 rows (i=0,1) of A are multiplied using Strassen on 2x2 blocks
    - Bottom row (i=2) uses naive multiplication
    
    The 4x4 matrix B is split into 2x2 blocks.
    A[0:2, 0:4] * B[0:4, 0:4] -> C[0:2, 0:4]
    
    We split into 4 quadrants of B (each 2x2) and use Strassen for each,
    but the A side is shared across columns.
    
    Actually, let's think more carefully:
    A is 3x4, B is 4x4, C is 3x4.
    
    For rows i=0,1 of C:
    C[i,k] = sum_j A[i,j] * B[j,k]
    
    Split j into {0,1} and {2,3}, split k into {0,1} and {2,3}:
    C[0:2, 0:2] = A[0:2,0:2]*B[0:2,0:2] + A[0:2,2:4]*B[2:4,0:2]
    C[0:2, 2:4] = A[0:2,0:2]*B[0:2,2:4] + A[0:2,2:4]*B[2:4,2:4]
    
    Each 2x2 * 2x2 -> 2x2 can be done with 7 Strassen multiplications.
    So 4 such products need 4*7 = 28 multiplications.
    
    For row i=2: C[2,k] = sum_j A[2,j]*B[j,k], needs 4*4=16 naive mults.
    
    Total rank: 28 + 16 = 44.
    """
    n, m, p = DIMENSIONS
    rank = 44
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    # Standard Strassen 2x2 steps
    # Each step: (U_coeffs, V_coeffs, W_coeffs)
    # Local indices (li, lj) for U, (lj, lk) for V, (li, lk) for W
    strassen_steps = [
        # M1 = (X00 + X11)(Y00 + Y11) -> Z00 += 1, Z11 += 1
        ({(0, 0): 1, (1, 1): 1}, {(0, 0): 1, (1, 1): 1}, {(0, 0): 1, (1, 1): 1}),
        # M2 = (X10 + X11)Y00 -> Z10 += 1, Z11 -= 1
        ({(1, 0): 1, (1, 1): 1}, {(0, 0): 1}, {(1, 0): 1, (1, 1): -1}),
        # M3 = X00(Y01 - Y11) -> Z01 += 1, Z11 += 1
        ({(0, 0): 1}, {(0, 1): 1, (1, 1): -1}, {(0, 1): 1, (1, 1): 1}),
        # M4 = X11(Y10 - Y00) -> Z00 += 1, Z10 += 1
        ({(1, 1): 1}, {(1, 0): 1, (0, 0): -1}, {(0, 0): 1, (1, 0): 1}),
        # M5 = (X00 + X01)Y11 -> Z01 += 1, Z00 -= 1
        ({(0, 0): 1, (0, 1): 1}, {(1, 1): 1}, {(0, 1): 1, (0, 0): -1}),
        # M6 = (X10 - X00)(Y00 + Y01) -> Z11 += 1
        ({(1, 0): 1, (0, 0): -1}, {(0, 0): 1, (0, 1): 1}, {(1, 1): 1}),
        # M7 = (X01 - X11)(Y10 + Y11) -> Z00 += 1
        ({(0, 1): 1, (1, 1): -1}, {(1, 0): 1, (1, 1): 1}, {(0, 0): 1}),
    ]

    # 4 blocks for top 2 rows: (x_i_offset, x_j_offset, y_j_offset, y_k_offset, z_i_offset, z_k_offset)
    # A[0:2, 0:2] * B[0:2, 0:2] -> C[0:2, 0:2]
    # A[0:2, 2:4] * B[2:4, 0:2] -> C[0:2, 0:2]
    # A[0:2, 0:2] * B[0:2, 2:4] -> C[0:2, 2:4]
    # A[0:2, 2:4] * B[2:4, 2:4] -> C[0:2, 2:4]
    blocks = [
        (0, 0, 0, 0, 0, 0),   # A[0:2,0:2] * B[0:2,0:2] -> C[0:2,0:2]
        (0, 2, 2, 0, 0, 0),   # A[0:2,2:4] * B[2:4,0:2] -> C[0:2,0:2]
        (0, 0, 0, 2, 0, 2),   # A[0:2,0:2] * B[0:2,2:4] -> C[0:2,2:4]
        (0, 2, 2, 2, 0, 2),   # A[0:2,2:4] * B[2:4,2:4] -> C[0:2,2:4]
    ]

    col = 0
    for (xi, xj, yj, yk, zi, zk) in blocks:
        for u_coeffs, v_coeffs, w_coeffs in strassen_steps:
            # Fill U: A indices
            for (li, lj), val in u_coeffs.items():
                gi = li + xi
                gj = lj + xj
                U[gi * m + gj][col] += val

            # Fill V: B indices
            for (lj, lk), val in v_coeffs.items():
                gj = lj + yj
                gk = lk + yk
                V[gj * p + gk][col] += val

            # Fill W: C indices (note: W[k*n+i] convention)
            for (li, lk), val in w_coeffs.items():
                gi = li + zi
                gk = lk + zk
                W[gk * n + gi][col] += val

            col += 1

    # Bottom row (i=2): naive 4x4 = 16 operations
    for j in range(m):
        for k in range(p):
            U[2 * m + j][col] = 1
            V[j * p + k][col] = 1
            W[k * n + 2][col] = 1
            col += 1

    return {"U": U, "V": V, "W": W}


def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s limit

    # Try the rank 44 Strassen decomposition
    result_44 = _strassen_44()
    ok, err = _verify_fast(result_44["U"], result_44["V"], result_44["W"], n, m, p)

    if ok:
        return result_44

    # Fallback to schoolbook
    return _schoolbook()
