"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 8]
import time
import numpy as np

# EVOLVE-BLOCK-START
def _verify(U, V, W, n, m, p):
    """Vectorized check that U,V,W decompose the matmul tensor."""
    Ua = np.asarray(U, dtype=float)
    Va = np.asarray(V, dtype=float)
    Wa = np.asarray(W, dtype=float)
    if Ua.shape[0] != n * m or Va.shape[0] != m * p or Wa.shape[0] != p * n:
        return False
    rank = Ua.shape[1]
    if Va.shape[1] != rank or Wa.shape[1] != rank:
        return False
    U3 = Ua.reshape(n, m, rank)
    V3 = Va.reshape(m, p, rank)
    W3 = Wa.reshape(p, n, rank)
    T = np.einsum('irj,krj,jri->ikr', U3, V3, W3)
    target = np.ones((n, p))
    return np.allclose(np.einsum('ikr,kr->ik', T, np.ones(rank)), target, atol=1e-7)


def _strassen():
    """Verified 2x2x2 Strassen factors. Indexing [row,col]: 00,01,10,11."""
    U2 = [[1,0,0,1],[0,0,1,1],[1,0,0,0],[0,0,0,1],[1,1,0,0],[-1,0,1,0],[0,1,0,-1]]
    V2 = [[1,0,0,1],[1,0,0,0],[0,1,0,-1],[-1,0,1,0],[0,0,0,1],[1,1,0,0],[0,0,1,1]]
    W2 = [[1,0,0,1],[0,0,1,-1],[0,1,0,1],[1,0,1,0],[-1,1,0,0],[0,0,0,1],[1,0,0,0]]
    return U2, V2, W2


def _kronecker_strassen(n, m, p):
    """Kronecker product of Strassen 2x2 over 2x2 blocks -> rank 98 for 4x4x8."""
    U2, V2, W2 = _strassen()
    rank = 2 * 7 * 7
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    for pb in range(2):
        for s in range(7):
            for t in range(7):
                col = pb * 49 + s * 7 + t
                U4 = np.zeros(16)
                V4 = np.zeros(16)
                W4 = np.zeros(16)
                for I in range(4):
                    for J in range(4):
                        U4[I * 4 + J] = U2[s][I // 2 * 2 + J // 2] * U2[t][I % 2 * 2 + J % 2]
                for J in range(4):
                    for K in range(4):
                        V4[J * 4 + K] = V2[s][J // 2 * 2 + K // 2] * V2[t][J % 2 * 2 + K % 2]
                for I in range(4):
                    for K in range(4):
                    # W matrix index is [k*n+i]; store into [I,K] then transpose below
                        W4[I * 4 + K] = W2[s][I // 2 * 2 + K // 2] * W2[t][I % 2 * 2 + K % 2]
                U[:, col] = U4
                V[:, col] = V4[:, pb * 4:(pb + 1) * 4].reshape(16)
                W[:, col] = W4.T.reshape(16)
    return {"U": U, "V": V, "W": W}


def _reduce(U, V, W, tol=1e-9, deadline=None):
    """Remove zero columns and linearly dependent columns (rank reduction)."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    R = U.shape[1]
    keep = [c for c in range(R) if (np.any(U[:, c] != 0) or np.any(V[:, c] != 0) or np.any(W[:, c] != 0))]
    U = U[:, keep]; V = V[:, keep]; W = W[:, keep]
    R = U.shape[1]
    M = np.vstack([U.reshape(-1, R), V.reshape(-1, R), W.reshape(-1, R)])
    try:
        _, pivots = np.linalg.qr(M, mode='reduced')
    except np.linalg.LinAlgError:
        return U, V, W
    piv = list(pivots)
    piv = [p for p in piv if p > tol]
    if len(piv) >= R:
        return U, V, W
    if len(piv) == 0:
        return U, V, W
    U = U[:, piv]; V = V[:, piv]; W = W[:, piv]
    return U, V, W


def construct():
    n, m, p = DIMENSIONS
    start = time.time()
    deadline = start + 130.0

    best = None
    best_rank = None

    # 1) Kronecker-Strassen rank-98, then reduce
    try:
        cand = _kronecker_strassen(n, m, p)
        U, V, W = cand["U"], cand["V"], cand["W"]
        U, V, W = _reduce(U, V, W, deadline=deadline)
        if _verify(U, V, W, n, m, p):
            best = {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
            best_rank = U.shape[1]
    except Exception:
        pass

    # 2) Fallback: schoolbook (always valid)
    if best is None:
        rank = n * m * p
        U = np.zeros((n * m, rank)); V = np.zeros((m * p, rank)); W = np.zeros((p * n, rank))
        c = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, c] = 1.0
                    V[j * p + k, c] = 1.0
                    W[k * n + i, c] = 1.0
                    c += 1
        best = {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
        best_rank = rank

    return best
# EVOLVE-BLOCK-END
