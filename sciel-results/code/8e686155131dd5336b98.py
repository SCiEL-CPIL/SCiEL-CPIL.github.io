"""Model-and-optimize: construct the known Strassen+schoolbook rank-44 decomposition and return it."""
import time
import sys

DIMENSIONS = [2, 3, 8]

# EVOLVE-BLOCK-START
def construct():
    t0 = time.time()
    deadline = t0 + 140.0

    def _ok():
        return (time.time() - t0) < deadline

    n, m, p = 2, 3, 8

    # Schoolbook rank-48 fallback (safe but weaker).
    def _schoolbook():
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        c = 0
        for i in range(n):
            for a in range(m):
                for k in range(p):
                    U[i * m + a][c] = 1
                    V[a * p + k][c] = 1
                    W[k * n + i][c] = 1
                    c += 1
        return {"U": U, "V": V, "W": W}

    # Strassen 2x2x2 core (rank 7)
    S_U = [
        [1, 0, 0, 1],
        [0, 0, 1, 1],
        [1, 0, 0, 0],
        [0, 0, 0, 1],
        [1, 1, 0, 0],
        [-1, 0, 1, 0],
        [0, 1, 0, -1]
    ]
    S_V = [
        [1, 0, 0, 1],
        [1, 0, 0, 0],
        [0, 1, 0, -1],
        [-1, 0, 1, 0],
        [0, 0, 0, 1],
        [1, 1, 0, 0],
        [0, 0, 1, 1]
    ]
    S_W = [
        [1, 0, 0, 1],
        [0, 1, 0, -1],
        [0, 0, 1, 1],
        [1, 1, 0, 0],
        [-1, 0, 1, 0],
        [0, 0, 0, 1],
        [1, 0, 0, 0]
    ]

    # Pre-compute non-zero entries for each Strassen vector
    def _nonzero(vec):
        return [(j, vec[j]) for j in range(len(vec)) if vec[j] != 0]

    S_U_nz = [_nonzero(row) for row in S_U]
    S_V_nz = [_nonzero(row) for row in S_V]
    S_W_nz = [_nonzero(row) for row in S_W]

    # Quick structural validation of Strassen core
    def _verify_strassen():
        for i in range(2):
            for a in range(2):
                for k in range(2):
                    s = 0
                    for r in range(7):
                        u = S_U[r][i * 2 + a]
                        v = S_V[r][a * 2 + k]
                        w = S_W[r][k * 2 + i]
                        s += u * v * w
                    if s != 1:
                        return False
        return True

    if not _verify_strassen():
        return _schoolbook()
    if not _ok():
        return _schoolbook()

    # Build rank-44 candidate
    rank = 44
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    col = 0
    # Part 1: A[0:2, 0:2] * B[0:2, :8] - 4 blocks of 2x2x2 Strassen
    for block_idx in range(4):
        k0 = 2 * block_idx
        for r in range(7):
            # U: local (li,la) -> global row = li*m + la
            for (j, uval) in S_U_nz[r]:
                li, la = j // 2, j % 2
                U[li * m + la][col] = uval
            # V: local (la,lk) -> global row = la*p + (k0 + lk)
            for (j, vval) in S_V_nz[r]:
                la, lk = j // 2, j % 2
                V[la * p + (k0 + lk)][col] = vval
            # W: local (lk,li) -> global row = (k0 + lk)*n + li
            for (j, wval) in S_W_nz[r]:
                lk, li = j // 2, j % 2
                W[(k0 + lk) * n + li][col] = wval
            col += 1
            if not _ok():
                return _schoolbook()

    # Part 2: A[0:2, 2] * B[2, :8] (schoolbook, 16 terms)
    for i in range(2):
        for k in range(8):
            U[i * m + 2][col] = 1
            V[2 * p + k][col] = 1
            W[k * n + i][col] = 1
            col += 1
            if not _ok():
                return _schoolbook()

    # Deterministic exact check using integers only
    def _validate_int(candidate):
        U = candidate["U"]
        V = candidate["V"]
        W = candidate["W"]
        R = len(U[0])
        for i in range(n):
            for a in range(m):
                for k in range(p):
                    s = 0
                    ur = U[i * m + a]
                    vr = V[a * p + k]
                    wr = W[k * n + i]
                    for c in range(R):
                        uv = ur[c]
                        if uv == 0:
                            continue
                        vv = vr[c]
                        if vv == 0:
                            continue
                        s += uv * vv * wr[c]
                    if s != 1:
                        return False
        return True

    cand = {"U": U, "V": V, "W": W}
    if _validate_int(cand):
        return cand

    # Fallback: try numpy-based verification as a cross-check
    try:
        import numpy as np
        U_np = np.array(U, dtype=np.float64)
        V_np = np.array(V, dtype=np.float64)
        W_np = np.array(W, dtype=np.float64)
        valid = True
        for i in range(n):
            for a in range(m):
                for k in range(p):
                    s = float(np.dot(np.dot(U_np[i*m+a], V_np[a*p+k]), W_np[k*n+i]))
                    if abs(s - 1.0) > 1e-9:
                        valid = False
                        break
                if not valid:
                    break
            if not valid:
                break
        if valid:
            return cand
    except Exception:
        pass

    return _schoolbook()
# EVOLVE-BLOCK-END
