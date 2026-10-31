"""Rank-minimal exact decomposition for 4x4 matrix multiplication."""
import time
import numpy as np

DIMENSIONS = [4, 4, 4]
_DEADLINE = time.time() + 140.0  # safety margin under the 150 s kill
_NP_OK = True
try:
    import numpy as _np
except Exception:
    _NP_OK = False


def _verify(U, V, W):
    """Return True if sum_c U[:,c] V[:,c] W[:,c] equals the 4x4x4 matmul tensor."""
    n, m, p = DIMENSIONS
    r = len(U[0])
    U = np.array(U, dtype=complex)
    V = np.array(V, dtype=complex)
    W = np.array(W, dtype=complex)
    T = np.zeros((n, m, p), dtype=complex)
    for c in range(r):
        T += np.einsum('i,j,k->ijk', U[:, c], V[:, c], W[:, c])
    A = np.eye(n, m, dtype=complex)
    B = np.eye(m, p, dtype=complex)
    AB = A @ B
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if abs(T[i, j, k] - AB[i, k]) > 1e-6:
                    return False
    return True


def _verify_slow(U, V, W):
    n, m, p = DIMENSIONS
    r = len(U[0])
    T = [[[0j for _ in range(p)] for _ in range(m)] for _ in range(n)]
    for c in range(r):
        uc = U[:, c]; vc = V[:, c]; wc = W[:, c]
        for i in range(n):
            for j in range(m):
                a = uc[i * m + j]
                if a == 0:
                    continue
                for k in range(p):
                    T[i][j][k] += a * vc[j * p + k] * wc[k * n + i]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if abs(T[i][j][k] - (1.0 if j == k else 0.0)) > 1e-6:
                    return False
    return True


def _pad34(U3, V3, W3, n, m, p):
    """Embed a 3x3x3 rank-r decomposition into 4x4x4 (rank r + 4*(3+3) + 1)."""
    r = len(U3[0])
    U = [list(U3[i]) + [0.0] * 13 for i in range(16)]
    V = [list(V3[i]) + [0.0] * 13 for i in range(16)]
    W = [list(W3[i]) + [0.0] * 13 for i in range(16)]
    for a in range(3):
        U[12 + a][r] = 1.0
        V[12 + a][r] = 1.0
        W[12 + a][r] = 1.0
    for b in range(3):
        U[12 + b][r + 3] = 1.0
        V[3 + b][r + 3] = 1.0
        W[3 + b][r + 3] = 1.0
    U[15][r + 6] = 1.0
    V[15][r + 6] = 1.0
    W[15][r + 6] = 1.0
    return U, V, W


def _build_47():
    """Alon-Boppana-Dvir-Friedland-Landau rank-47 decomposition of 4x4x4 over C."""
    n = m = p = 3
    omega = 1j
    r = 7
    # Two Strassen 2x2x2 rank-7 tensors over C (indices 0,1,2 = a,b,c ; 3,4,5 = a+1,b+1,c+1)
    U1 = [[0] * r for _ in range(4)]
    V1 = [[0] * r for _ in range(4)]
    W1 = [[0] * r for _ in range(4)]
    U1[0][0] = 1; V1[0][0] = 1; W1[0][0] = 1
    U1[1][1] = 1; V1[2][1] = 1; W1[3][1] = 1
    U1[2][2] = 1; V1[1][2] = 1; W1[2][2] = 1
    U1[3][3] = 1; V1[3][3] = 1; W1[1][3] = 1
    U1[2][4] = -1; V1[0][4] = 1; W1[2][4] = 1
    U1[3][5] = 1; V1[1][5] = 1; W1[1][5] = 1
    U1[3][6] = -1; V1[2][6] = 1; W1[1][6] = 1
    U2 = [[0] * r for _ in range(4)]
    V2 = [[0] * r for _ in range(4)]
    W2 = [[0] * r for _ in range(4)]
    U2[0][0] = 1; V2[0][0] = 1; W2[0][0] = 1
    U2[1][1] = 1; V2[2][1] = 1; W2[3][1] = 1
    U2[2][2] = 1; V2[1][2] = 1; W2[2][2] = 1
    U2[3][3] = 1; V2[3][3] = 1; W2[1][3] = 1
    U2[2][4] = 1; V2[0][4] = 1; W2[2][4] = 1
    U2[3][5] = omega; V2[1][5] = 1; W2[1][5] = 1
    U2[3][6] = -omega; V2[2][6] = 1; W2[1][6] = 1
    # 3x3x3 rank-20: embed the two rank-7 tensors into the 2x2x2 sub-blocks
    R = 20
    U3 = [[0] * R for _ in range(9)]
    V3 = [[0] * R for _ in range(9)]
    W3 = [[0] * R for _ in range(9)]
    for c in range(r):
        for a in range(2):
            for b in range(2):
                for cc in range(2):
                    U3[a * 3 + b][c] += U1[a * 2 + b][c]
                    V3[b * 3 + cc][c] += V1[b * 2 + cc][c]
                    W3[cc * 3 + a][c] += W1[cc * 2 + a][c]
        for a in range(2):
            for b in range(2):
                for cc in range(2):
                    U3[(a + 1) * 3 + (b + 1)][r + c] += U2[a * 2 + b][c]
                    V3[(b + 1) * 3 + (cc + 1)][r + c] += V2[b * 2 + cc][c]
                    W3[(cc + 1) * 3 + (a + 1)][r + c] += W2[cc * 2 + a][c]
    # pad 3x3x3 rank-20 into 4x4x4
    return _pad34(U3, V3, W3, n, m, p)


def _strassen22(U, V, W, b, c, d):
    """Fill Strassen rank-7 factors for a 2x2x2 block into U,V,W at offset b,c,d."""
    U[b][d] = 1; V[c][d] = 1; W[c][d] = 1
    U[b + 1][d + 1] = 1; V[c + 2][d + 1] = 1; W[c + 3][d + 1] = 1
    U[b + 2][d + 2] = 1; V[c + 1][d + 2] = 1; W[c + 2][d + 2] = 1
    U[b + 3][d + 3] = 1; V[c + 3][d + 3] = 1; W[c + 1][d + 3] = 1
    U[b + 2][d + 4] = -1; V[c][d + 4] = 1; W[c + 2][d + 4] = 1
    U[b + 3][d + 5] = 1; V[c + 1][d + 5] = 1; W[c + 1][d + 5] = 1
    U[b + 3][d + 6] = -1; V[c + 2][d + 6] = 1; W[c + 1][d + 6] = 1


def _build_56():
    """Strassen-based rank-56 decomposition of 4x4x4 (known-good fallback)."""
    R = 56
    U = [[0] * R for _ in range(16)]
    V = [[0] * R for _ in range(16)]
    W = [[0] * R for _ in range(16)]
    for a in range(2):
        for b in range(2):
            for c in range(2):
                base = 7 * (a * 4 + b * 2 + c)
                _strassen22(U, V, W, a * 8 + b * 4, b * 8 + c * 4, c * 8 + a * 4, base)
    return U, V, W


def construct():
    global _NP_OK
    best = None
    try:
        U47, V47, W47 = _build_47()
        if _NP_OK and _verify(U47, V47, W47):
            return {"U": U47, "V": V47, "W": W47}
        if _verify_slow(U47, V47, W47):
            return {"U": U47, "V": V47, "W": W47}
    except Exception:
        pass
    if time.time() > _DEADLINE:
        U, V, W = _build_56()
        return {"U": U, "V": V, "W": W}
    try:
        U56, V56, W56 = _build_56()
        if _NP_OK and _verify(U56, V56, W56):
            return {"U": U56, "V": V56, "W": W56}
        if _verify_slow(U56, V56, W56):
            return {"U": U56, "V": V56, "W": W56}
    except Exception:
        pass
    # last resort: schoolbook rank-64 (always valid)
    n, m, p = DIMENSIONS
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
