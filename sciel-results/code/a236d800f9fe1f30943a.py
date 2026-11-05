"""Tensor decomposition for 2x3 * 3x10 matrix multiplication."""
import time
import numpy as np

DIMENSIONS = [2, 3, 10]

# EVOLVE-BLOCK-START


def _validate(n, m, p, U, V, W):
    """Verify exact tensor decomposition."""
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    U3 = U_arr.reshape(n, m, -1)
    V3 = V_arr.reshape(m, p, -1)
    W3 = W_arr.reshape(p, n, -1)
    T = np.einsum('ijr,jkr,kir->ijk', U3, V3, W3)
    return np.allclose(T, np.ones((n, m, p)), atol=1e-9)


def _build_strassen(n, m, p):
    """Build rank-55 decomposition using Strassen on 2x2 blocks."""
    assert n == 2 and m == 3 and p == 10
    rank = 55
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    r = 0
    strassen_u = {
        0: [(0, 1), (4, 1)], 1: [(3, 1), (4, 1)], 2: [(0, 1)],
        3: [(4, 1)], 4: [(0, 1), (1, 1)], 5: [(3, 1), (0, -1)], 6: [(1, 1), (4, -1)]
    }
    strassen_v = {
        0: [(0, 1), (11, 1)], 1: [(0, 1)], 2: [(1, 1), (11, -1)],
        3: [(10, 1), (0, -1)], 4: [(11, 1)], 5: [(0, 1), (1, 1)], 6: [(10, 1), (11, 1)]
    }
    strassen_w = {
        0: [(0, 1), (3, 1)], 1: [(1, 1), (3, -1)], 2: [(2, 1), (3, 1)],
        3: [(0, 1), (1, 1)], 4: [(0, -1), (2, 1)], 5: [(3, 1)], 6: [(0, 1)]
    }
    
    for b in range(5):
        col0, col1 = 2 * b, 2 * b + 1
        for s in range(7):
            for (u_row, coeff) in strassen_u[s]:
                U[u_row][r] += coeff
            for (v_off, coeff) in strassen_v[s]:
                v_row = col0 if v_off == 0 else col1 if v_off == 1 else 10 + col0 if v_off == 10 else 10 + col1
                V[v_row][r] += coeff
            for (w_off, coeff) in strassen_w[s]:
                w_row = col0 * 2 + w_off if w_off < 2 else col1 * 2 + (w_off - 2)
                W[w_row][r] += coeff
            r += 1
    
    for i in range(2):
        for k in range(10):
            U[i * 3 + 2][r] = 1
            V[20 + k][r] = 1
            W[k * 2 + i][r] = 1
            r += 1
    
    assert r == rank
    return U, V, W, rank


def _reduce_rank(n, m, p, U, V, W, deadline):
    """Aggressive rank reduction using systematic subset removal."""
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    rank = U_arr.shape[1]
    target = np.ones((n, m, p))
    tol = 1e-9
    
    col_contrib = np.zeros((rank, n, m, p))
    for r in range(rank):
        u_col = U_arr[:, r].reshape(n, m)
        v_col = V_arr[:, r].reshape(m, p)
        w_col = W_arr[:, r].reshape(p, n)
        col_contrib[r] = np.einsum('ij,jk,ki->ijk', u_col, v_col, w_col)
    
    T_current = col_contrib.sum(axis=0)
    keep = np.ones(rank, dtype=bool)
    
    # Phase 1: Remove singles
    changed = True
    while changed and time.time() < deadline:
        changed = False
        for r in range(rank):
            if not keep[r]:
                continue
            if np.allclose(T_current - col_contrib[r], target, atol=tol):
                keep[r] = False
                T_current -= col_contrib[r]
                changed = True
    
    # Phase 2: Remove pairs
    changed = True
    while changed and time.time() < deadline:
        changed = False
        kept = [r for r in range(rank) if keep[r]]
        for i in range(len(kept)):
            if time.time() > deadline:
                break
            for j in range(i + 1, len(kept)):
                r1, r2 = kept[i], kept[j]
                if not keep[r1] or not keep[r2]:
                    continue
                if np.allclose(T_current - col_contrib[r1] - col_contrib[r2], target, atol=tol):
                    keep[r1] = keep[r2] = False
                    T_current -= col_contrib[r1] + col_contrib[r2]
                    changed = True
                    break
            if changed:
                break
    
    # Phase 3: Remove triples
    changed = True
    while changed and time.time() < deadline:
        changed = False
        kept = [r for r in range(rank) if keep[r]]
        for i in range(len(kept)):
            if time.time() > deadline:
                break
            for j in range(i + 1, len(kept)):
                if time.time() > deadline:
                    break
                for k in range(j + 1, len(kept)):
                    r1, r2, r3 = kept[i], kept[j], kept[k]
                    if not keep[r1] or not keep[r2] or not keep[r3]:
                        continue
                    if np.allclose(T_current - col_contrib[r1] - col_contrib[r2] - col_contrib[r3], target, atol=tol):
                        keep[r1] = keep[r2] = keep[r3] = False
                        T_current -= col_contrib[r1] + col_contrib[r2] + col_contrib[r3]
                        changed = True
                        break
                if changed:
                    break
            if changed:
                break
    
    new_rank = int(np.sum(keep))
    if new_rank >= rank:
        return U, V, W, rank
    
    U_new = U_arr[:, keep]
    V_new = V_arr[:, keep]
    W_new = W_arr[:, keep]
    
    if _validate(n, m, p, U_new.tolist(), V_new.tolist(), W_new.tolist()):
        return U_new.tolist(), V_new.tolist(), W_new.tolist(), new_rank
    return U, V, W, rank


def construct():
    start_time = time.time()
    deadline = start_time + 140
    n, m, p = DIMENSIONS
    
    U, V, W, rank = _build_strassen(n, m, p)
    
    if not _validate(n, m, p, U, V, W):
        # Fallback: schoolbook
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    r = i * m * p + j * p + k
                    U[i * m + j][r] = 1
                    V[j * p + k][r] = 1
                    W[k * n + i][r] = 1
    
    U, V, W, rank = _reduce_rank(n, m, p, U, V, W, deadline)
    
    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
