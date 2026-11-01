import time
import numpy as np

DIMENSIONS = [2, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    """
    Rank-28 decomposition of the 2x4 * 4x4 matrix multiplication tensor.
    
    Strategy:
      - Partition A (2x4) into A_L (2x2, cols 0-1) and A_R (2x2, cols 2-3).
      - Partition B (4x4) into 2x2 blocks: B_TL, B_TR (top), B_BL, B_BR (bottom).
      - C (2x4) = [C_L, C_R] where:
            C_L = A_L * B_TL + A_R * B_BL
            C_R = A_L * B_TR + A_R * B_BR
      - Each 2x2 * 2x2 product uses Strassen (rank 7).
      - Total rank = 4 * 7 = 28.
    
    This is an improvement over the naive rank 32.
    """
    n, m, p = DIMENSIONS  # 2, 4, 4
    rank = 28

    U = np.zeros((n * m, rank), dtype=int)  # shape (8, 28)
    V = np.zeros((m * p, rank), dtype=int)  # shape (16, 28)
    W = np.zeros((p * n, rank), dtype=int)  # shape (8, 28)

    # Strassen 2x2 * 2x2 = 2x2
    # X = [[x00, x01], [x10, x11]], Y = [[y00, y01], [y10, y11]]
    # P = [[p00, p01], [p10, p11]]
    #
    # M1 = (x00 + x11)(y00 + y11)
    # M2 = (x10 + x11)y00
    # M3 = x00(y01 - y11)
    # M4 = x11(y10 - y00)
    # M5 = (x00 + x01)y11
    # M6 = (x10 - x00)(y00 + y01)
    # M7 = (x01 - x11)(y10 + y11)
    #
    # P00 = M1 + M4 - M5 + M7
    # P01 = M3 + M5
    # P10 = M2 + M4
    # P11 = M1 - M2 + M3 + M6

    # Each term: (u_coeffs, v_coeffs, w_coeffs)
    # u_coeffs: list of (local_index, coeff) for X (2x2)
    # v_coeffs: list of (local_index, coeff) for Y (2x2)
    # w_coeffs: list of (local_index, coeff) for P (2x2)
    # local index: 0=(0,0), 1=(0,1), 2=(1,0), 3=(1,1)

    strassen_terms = [
        # M1 = (x00+x11)(y00+y11) -> P00, P11
        ([(0, 1), (3, 1)], [(0, 1), (3, 1)], [(0, 1), (3, 1)]),
        # M2 = (x10+x11)*y00 -> P10, P11
        ([(2, 1), (3, 1)], [(0, 1)], [(2, 1), (3, -1)]),
        # M3 = x00*(y01-y11) -> P01, P11
        ([(0, 1)], [(1, 1), (3, -1)], [(1, 1), (3, 1)]),
        # M4 = x11*(y10-y00) -> P00, P10
        ([(3, 1)], [(2, 1), (0, -1)], [(0, 1), (2, 1)]),
        # M5 = (x00+x01)*y11 -> P00, P01
        ([(0, 1), (1, 1)], [(3, 1)], [(0, -1), (1, 1)]),
        # M6 = (x10-x00)*(y00+y01) -> P11
        ([(2, 1), (0, -1)], [(0, 1), (1, 1)], [(3, 1)]),
        # M7 = (x01-x11)*(y10+y11) -> P00
        ([(1, 1), (3, -1)], [(2, 1), (3, 1)], [(0, 1)]),
    ]

    def map_u(a_side, local_idx):
        """Map local 2x2 index to global U index (A is 2x4)."""
        r = local_idx // 2
        c = local_idx % 2
        if a_side == 'L':
            return r * 4 + c
        else:
            return r * 4 + (2 + c)

    def map_v(block, local_idx):
        """Map local 2x2 index to global V index (B is 4x4)."""
        r = local_idx // 2
        c = local_idx % 2
        if block == 'TL':
            return r * 4 + c
        elif block == 'TR':
            return r * 4 + (2 + c)
        elif block == 'BL':
            return (2 + r) * 4 + c
        elif block == 'BR':
            return (2 + r) * 4 + (2 + c)

    def map_w(c_side, local_idx):
        """Map local 2x2 index to global W index (C is 2x4, W[k*n+i,r])."""
        r = local_idx // 2  # row of P
        c = local_idx % 2   # col of P
        if c_side == 'L':
            return c * 2 + r
        else:
            return (2 + c) * 2 + r

    def apply_strassen(a_side, b_block, c_side, start_col):
        for t in strassen_terms:
            u_local, v_local, w_local = t
            col = start_col
            for idx, coeff in u_local:
                U[map_u(a_side, idx), col] += coeff
            for idx, coeff in v_local:
                V[map_v(b_block, idx), col] += coeff
            for idx, coeff in w_local:
                W[map_w(c_side, idx), col] += coeff
            start_col += 1
        return start_col

    # 1. A_L * B_TL -> C_L
    col = apply_strassen('L', 'TL', 'L', 0)
    # 2. A_R * B_BL -> C_L
    col = apply_strassen('R', 'BL', 'L', col)
    # 3. A_L * B_TR -> C_R
    col = apply_strassen('L', 'TR', 'R', col)
    # 4. A_R * B_BR -> C_R
    col = apply_strassen('R', 'BR', 'R', col)

    assert col == rank, f"Expected {rank} columns, got {col}"

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END

if __name__ == "__main__":
    res = construct()
    n, m, p = DIMENSIONS
    U = np.array(res["U"], dtype=float)
    V = np.array(res["V"], dtype=float)
    W = np.array(res["W"], dtype=float)
    A = np.arange(n * m).reshape(n, m) + 1.0
    B = np.arange(m * p).reshape(m, p) + 2.0
    C = A @ B
    Cf = np.zeros((n, p))
    for ii in range(n):
        for kk in range(p):
            acc = 0.0
            for cc in range(U.shape[1]):
                for jj in range(m):
                    acc += U[ii * m + jj, cc] * V[jj * p + kk, cc] * W[kk * n + ii, cc]
            Cf[ii, kk] = acc
    assert np.allclose(C, Cf), (C, Cf)
    print("OK rank", U.shape[1])
