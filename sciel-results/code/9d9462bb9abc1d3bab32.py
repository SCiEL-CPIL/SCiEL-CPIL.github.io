"""Matrix multiplication 2x4x7 with rank-50 decomposition via tensor product of sub-blocks."""
import time
import numpy as np

DIMENSIONS = [2, 4, 7]

# EVOLVE-BLOCK-START

def _build_strassen_222():
    """Rank-7 decomposition for 2x2x2 matrix multiplication (Strassen)."""
    n, m, p = 2, 2, 2
    r = 7
    U = np.zeros((n * m, r))
    V = np.zeros((m * p, r))
    W = np.zeros((p * n, r))

    # M1 = (A00+A11)(B00+B11)
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    W[0, 0] = 1; W[3, 0] = 1

    # M2 = (A10+A11)B00
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    W[1, 1] = 1; W[3, 1] = -1

    # M3 = A00(B01-B11)
    U[0, 2] = 1
    V[1, 2] = 1; V[3, 2] = -1
    W[2, 2] = 1; W[3, 2] = 1

    # M4 = A11(B10-B00)
    U[3, 3] = 1
    V[2, 3] = 1; V[0, 3] = -1
    W[0, 3] = 1; W[1, 3] = 1

    # M5 = (A00+A01)B11
    U[0, 4] = 1; U[1, 4] = 1
    V[3, 4] = 1
    W[0, 4] = -1; W[2, 4] = 1

    # M6 = (A10-A00)(B00+B01)
    U[2, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[1, 5] = 1
    W[3, 5] = 1

    # M7 = (A01-A11)(B10+B11)
    U[1, 6] = 1; U[3, 6] = -1
    V[2, 6] = 1; V[3, 6] = 1
    W[0, 6] = 1

    return U, V, W


def _build_school_221():
    """Rank-4 decomposition for 2x2x1 matrix multiplication."""
    n, m, p = 2, 2, 1
    r = 4
    U = np.zeros((n * m, r))
    V = np.zeros((m * p, r))
    W = np.zeros((p * n, r))

    for c in range(r):
        i = c // m
        j = c % m
        k = 0
        U[i * m + j, c] = 1.0
        V[j * p + k, c] = 1.0
        W[k * n + i, c] = 1.0

    return U, V, W


def _verify_decomposition(U, V, W, n, m, p):
    """Check the tensor identity: sum_c W[k*n+i,c]*U[i*m+j,c]*V[jp*p+k,c] = delta_{j,jp}."""
    r = U.shape[1]
    for i in range(n):
        for j in range(m):
            for jp in range(m):
                for k in range(p):
                    s = 0.0
                    for c in range(r):
                        s += W[k * n + i, c] * U[i * m + j, c] * V[jp * p + k, c]
                    expected = 1.0 if j == jp else 0.0
                    if abs(s - expected) > 1e-9:
                        return False
    return True


def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110

    # Build sub-tensors
    U_s, V_s, W_s = _build_strassen_222()
    U_sc, V_sc, W_sc = _build_school_221()

    # Total rank: 2 m-blocks * (3*7 + 4) = 50
    total_rank = 50
    U = np.zeros((n * m, total_rank))
    V = np.zeros((m * p, total_rank))
    W = np.zeros((p * n, total_rank))

    col = 0
    m_blocks = [(0, 2), (2, 4)]
    p_blocks = [
        (0, 2, U_s, V_s, W_s, 7),
        (2, 4, U_s, V_s, W_s, 7),
        (4, 6, U_s, V_s, W_s, 7),
        (6, 7, U_sc, V_sc, W_sc, 4),
    ]

    for j0, j1 in m_blocks:
        for k0, k1, U_sub, V_sub, W_sub, rs in p_blocks:
            ps = k1 - k0
            for i in range(n):
                for js in range(j1 - j0):
                    U[i * m + j0 + js, col:col + rs] = U_sub[i * 2 + js, :]
            for js in range(j1 - j0):
                for ks in range(ps):
                    V[(j0 + js) * p + k0 + ks, col:col + rs] = V_sub[js * ps + ks, :]
            for ks in range(ps):
                for i in range(n):
                    W[(k0 + ks) * n + i, col:col + rs] = W_sub[ks * 2 + i, :]
            col += rs

    # Verify and return
    if time.time() < deadline:
        if _verify_decomposition(U, V, W, n, m, p):
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
        else:
            # Fallback: schoolbook rank 56
            r = n * m * p
            U_fb = [[0] * r for _ in range(n * m)]
            V_fb = [[0] * r for _ in range(m * p)]
            W_fb = [[0] * r for _ in range(p * n)]
            c = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        U_fb[i * m + j][c] = 1
                        V_fb[j * p + k][c] = 1
                        W_fb[k * n + i][c] = 1
                        c += 1
            return {"U": U_fb, "V": V_fb, "W": W_fb}
    else:
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

# EVOLVE-BLOCK-END
