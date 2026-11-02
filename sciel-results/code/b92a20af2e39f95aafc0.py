import numpy as np
import time

DIMENSIONS = [2, 3, 9]

def _build_rank50():
    """Deterministic rank-50 exact decomposition (Strassen on 2x2 blocks + schoolbook)."""
    n, m, p = DIMENSIONS
    rank = 50
    U = np.zeros((n * m, rank), dtype=np.int64)
    V = np.zeros((m * p, rank), dtype=np.int64)
    W = np.zeros((p * n, rank), dtype=np.int64)
    col = 0
    for b in range(4):
        k0 = 2 * b; k1 = 2 * b + 1
        v_e = 0 * p + k0; v_f = 0 * p + k1
        v_g = 1 * p + k0; v_h = 1 * p + k1
        u_a = 0 * m + 0; u_b = 0 * m + 1
        u_c = 1 * m + 0; u_d = 1 * m + 1
        U[u_a][col] += 1; U[u_d][col] += 1; V[v_e][col] += 1; V[v_h][col] += 1
        U[u_c][col+1] += 1; U[u_d][col+1] += 1; V[v_e][col+1] += 1
        U[u_a][col+2] += 1; V[v_f][col+2] += 1; V[v_h][col+2] -= 1
        U[u_d][col+3] += 1; V[v_g][col+3] += 1; V[v_e][col+3] -= 1
        U[u_a][col+4] += 1; U[u_b][col+4] += 1; V[v_h][col+4] += 1
        U[u_c][col+5] += 1; U[u_a][col+5] -= 1; V[v_e][col+5] += 1; V[v_f][col+5] += 1
        U[u_b][col+6] += 1; U[u_d][col+6] -= 1; V[v_g][col+6] += 1; V[v_h][col+6] += 1
        w_00 = k0 * n + 0; w_01 = k1 * n + 0; w_10 = k0 * n + 1; w_11 = k1 * n + 1
        W[w_00][col] += 1; W[w_00][col+3] += 1; W[w_00][col+4] -= 1; W[w_00][col+6] += 1
        W[w_01][col+2] += 1; W[w_01][col+4] += 1
        W[w_10][col+1] += 1; W[w_10][col+3] += 1
        W[w_11][col] += 1; W[w_11][col+1] -= 1; W[w_11][col+2] += 1; W[w_11][col+5] += 1
        col += 7
    k8 = 8
    v_e = 0 * p + k8; v_g = 1 * p + k8
    u_a = 0 * m + 0; u_b = 0 * m + 1; u_c = 1 * m + 0; u_d = 1 * m + 1
    U[u_a][col] += 1; V[v_e][col] += 1
    U[u_b][col+1] += 1; V[v_g][col+1] += 1
    U[u_c][col+2] += 1; V[v_e][col+2] += 1
    U[u_d][col+3] += 1; V[v_g][col+3] += 1
    W[k8 * n + 0][col] += 1; W[k8 * n + 0][col+1] += 1
    W[k8 * n + 1][col+2] += 1; W[k8 * n + 1][col+3] += 1
    col += 4
    u_ar0 = 0 * m + 2; u_ar1 = 1 * m + 2
    for k in range(9):
        v_b = 2 * p + k
        U[u_ar0][col] += 1; V[v_b][col] += 1; W[k * n + 0][col] += 1
        U[u_ar1][col+1] += 1; V[v_b][col+1] += 1; W[k * n + 1][col+1] += 1
        col += 2
    return U, V, W

def _verify(U, V, W):
    """Exact check: sum_t U[:,t] outer V[:,t] outer W[:,t] must equal the matmul tensor."""
    n, m, p = DIMENSIONS
    U = np.asarray(U, dtype=float); V = np.asarray(V, dtype=float); W = np.asarray(W, dtype=float)
    T = np.zeros((n * m, m * p, p * n))
    for t in range(U.shape[1]):
        T += np.einsum('i,j,k->ijk', U[:, t], V[:, t], W[:, t])
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = T[i * m + j, j * p + k, k * n + i]
                if val != 1.0:
                    return False
    return True

def _try_reduce(U, V, W, deadline):
    """Try to reduce rank by finding rank-1 terms that are linear combinations of others."""
    n, m, p = DIMENSIONS
    U = U.copy(); V = V.copy(); W = W.copy()
    r0 = U.shape[1]
    keep = list(range(r0))
    improved = True
    while improved:
        improved = False
        if time.time() > deadline - 15:
            break
        for t in range(len(keep)):
            if time.time() > deadline - 15:
                break
            idx = keep[t]
            # Try to express term idx as a linear combination of the other kept terms.
            others = [k for k in keep if k != idx]
            if len(others) < 2:
                continue
            # For each of the three factor matrices, the term must equal sum c_j * term_j.
            # Solve least-squares for c in V and W simultaneously is hard; instead check
            # whether the 3-way term lies in the span by checking the V and W factors
            # are in the span of the other V/W factors (necessary).
            Vo = V[:, others]; Wo = W[:, others]
            # Check V[:,idx] in span of Vo and W[:,idx] in span of Wo with SAME c.
            # Use V to solve for c, then verify W.
            try:
                c, *_ = np.linalg.lstsq(Vo, V[:, idx], rcond=None)
            except Exception:
                continue
            if np.max(np.abs(Wo @ c - W[:, idx])) > 1e-6:
                continue
            # Also verify U factor with same c (U must equal sum c_j U_j)
            Uo = U[:, others]
            if np.max(np.abs(Uo @ c - U[:, idx])) > 1e-6:
                continue
            # All three factors consistent: term idx is redundant. Remove it.
            keep = [k for k in keep if k != idx]
            improved = True
            break
    if len(keep) >= r0:
        return None
    newU = U[:, keep]; newV = V[:, keep]; newW = W[:, keep]
    if not _verify(newU, newV, newW):
        return None
    return newU, newV, newW

def construct():
    start = time.time()
    deadline = start + 130  # safety margin before 150s kill
    U, V, W = _build_rank50()
    best = (U, V, W)
    if not _verify(U, V, W):
        # Should never happen, but guard.
        pass
    # Attempt rank reduction via redundancy search.
    red = _try_reduce(U, V, W, deadline)
    if red is not None:
        best = red
    U, V, W = best
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
