"""Matrix multiplication 3x4 by 4x6: structured Strassen decomposition + rank-reduction search."""
import time
import itertools

import numpy as np

n, m, p = 3, 4, 6

_START_TIME = time.time()
_DEADLINE = 143.0

def _time_left():
    return _DEADLINE - (time.time() - _START_TIME)

def _time_ok():
    return _time_left() > 8.0

def _flatten_contrib(U, V, W, r):
    """Return the vectorized contribution of column r as a 1D array."""
    contrib = np.zeros(n * m * p, dtype=int)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                contrib[(i * m + j) * p + k] = U[i * m + j, r] * V[j * p + k, r] * W[k * n + i, r]
    return contrib


def _build_base():
    """Build the rank-66 base decomposition. Returns (U, V, W) as numpy int arrays."""
    rank = 66
    U = np.zeros((n * m, rank), dtype=int)   # 12 x 66
    V = np.zeros((m * p, rank), dtype=int)   # 24 x 66
    W = np.zeros((p * n, rank), dtype=int)   # 18 x 66

    col = 0

    def add_schoolbook(i, j, k):
        nonlocal col
        U[i * m + j, col] = 1
        V[j * p + k, col] = 1
        W[k * n + i, col] = 1
        col += 1

    def add_strassen(i_base, j_base, k_base):
        nonlocal col
        a = i_base * m + j_base
        b = i_base * m + (j_base + 1)
        c = (i_base + 1) * m + j_base
        d = (i_base + 1) * m + (j_base + 1)
        e = j_base * p + k_base
        f = j_base * p + (k_base + 1)
        g = (j_base + 1) * p + k_base
        h = (j_base + 1) * p + (k_base + 1)
        P = k_base * n + i_base
        Q = (k_base + 1) * n + i_base
        R = k_base * n + (i_base + 1)
        S = (k_base + 1) * n + (i_base + 1)

        # P1 = (a+d)(e+h)
        U[a, col] = 1; U[d, col] = 1
        V[e, col] = 1; V[h, col] = 1
        W[P, col] = 1; W[S, col] = 1
        col += 1
        # P2 = (c+d)e
        U[c, col] = 1; U[d, col] = 1
        V[e, col] = 1
        W[R, col] = 1; W[S, col] = -1
        col += 1
        # P3 = a(f-h)
        U[a, col] = 1
        V[f, col] = 1; V[h, col] = -1
        W[Q, col] = 1; W[S, col] = 1
        col += 1
        # P4 = d(g-e)
        U[d, col] = 1
        V[g, col] = 1; V[e, col] = -1
        W[P, col] = 1; W[R, col] = 1
        col += 1
        # P5 = (a+b)h
        U[a, col] = 1; U[b, col] = 1
        V[h, col] = 1
        W[P, col] = -1; W[Q, col] = 1
        col += 1
        # P6 = (c-a)(e+f)
        U[c, col] = 1; U[a, col] = -1
        V[e, col] = 1; V[f, col] = 1
        W[S, col] = 1
        col += 1
        # P7 = (b-d)(g+h)
        U[b, col] = 1; U[d, col] = -1
        V[g, col] = 1; V[h, col] = 1
        W[P, col] = 1
        col += 1

    for j in range(m):
        for k in range(p):
            add_schoolbook(0, j, k)

    for k_block in range(p // 2):
        for j_block in range(m // 2):
            add_strassen(1, 2 * j_block, 2 * k_block)

    return U, V, W


def _reconstruct(U, V, W):
    """Reconstruct the multiplication tensor C_{i,j,k} = sum_r U[i,j] V[j,k] W[k,i]."""
    C = np.zeros((n, m, p), dtype=int)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0
                for r in range(U.shape[1]):
                    s += U[i * m + j, r] * V[j * p + k, r] * W[k * n + i, r]
                C[i, j, k] = s
    return C


def _target_tensor():
    C = np.zeros((n, m, p), dtype=int)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                C[i, j, k] = 1
    return C


def _rank_of(U, V, W):
    """Return the effective (non-zero) rank: number of columns r with U[:,r],V[:,r],W[:,r] all nonzero."""
    cnt = 0
    for r in range(U.shape[1]):
        if np.any(U[:, r]) and np.any(V[:, r]) and np.any(W[:, r]):
            cnt += 1
    return cnt


def _verify(U, V, W):
    """Check exactness and return (ok, effective_rank)."""
    U = np.asarray(U)
    V = np.asarray(V)
    W = np.asarray(W)
    target = _target_tensor()
    recon = _reconstruct(U, V, W)
    ok = np.array_equal(recon, target)
    return ok, _rank_of(U, V, W)


def _try_reduce(base_U, base_V, base_W):
    """Search for a lower-rank exact decomposition using vectorized linear algebra."""
    U = base_U.copy()
    V = base_V.copy()
    W = base_W.copy()
    R = U.shape[1]

    # Precompute all contributions as a 2D array: (R, n*m*p)
    contribs = np.zeros((R, n * m * p), dtype=int)
    for r in range(R):
        contribs[r] = _flatten_contrib(U, V, W, r)

    # Identify active columns
    active_mask = np.array([
        np.any(U[:, r]) and np.any(V[:, r]) and np.any(W[:, r])
        for r in range(R)
    ])
    active = [r for r in range(R) if active_mask[r]]
    
    # First pass: remove columns that are exact integer linear combinations of others
    # Use vectorized approach: for each candidate column, check if it's in span of others
    changed = True
    while changed and _time_ok():
        changed = False
        active = [r for r in range(R) if np.any(U[:, r]) and np.any(V[:, r]) and np.any(W[:, r])]
        for idx in active:
            if not _time_ok():
                break
            # Get contribution of this column
            vec = contribs[idx].astype(float)
            nz = np.nonzero(vec)[0]
            if len(nz) == 0:
                U[:, idx] = 0; V[:, idx] = 0; W[:, idx] = 0
                contribs[idx] = 0
                changed = True
                continue
            
            # Build matrix from other active columns
            others = [t for t in active if t != idx]
            if len(others) == 0:
                continue
            
            A = np.column_stack([contribs[t][nz].astype(float) for t in others])
            b = vec[nz]
            
            try:
                sol, residuals, rank_A, sv = np.linalg.lstsq(A, b, rcond=None)
            except Exception:
                continue
            
            # Check if solution is approximately integer
            rounded = np.round(sol)
            if np.allclose(sol, rounded, atol=1e-6):
                # Verify exactly
                coeffs = rounded.astype(int)
                total = np.zeros(n * m * p, dtype=int)
                for i, t in enumerate(others):
                    if coeffs[i] != 0:
                        total += coeffs[i] * contribs[t]
                if np.array_equal(total, contribs[idx]):
                    U[:, idx] = 0; V[:, idx] = 0; W[:, idx] = 0
                    contribs[idx] = 0
                    changed = True
                    if not _time_ok():
                        break

    # Verify after first pass
    ok, r = _verify(U, V, W)
    if not ok:
        return base_U.copy(), base_V.copy(), base_W.copy()

    # Second pass: try dropping pairs of columns simultaneously
    active = [t for t in range(R) if np.any(U[:, t]) and np.any(V[:, t]) and np.any(W[:, t])]
    best_rank = len(active)
    rng = np.random.default_rng(12345)

    # Try dropping random single columns
    for _ in range(5000):
        if not _time_ok():
            break
        if len(active) <= 1:
            break
        a = active[rng.integers(len(active))]
        U2 = U.copy(); V2 = V.copy(); W2 = W.copy()
        U2[:, a] = 0; V2[:, a] = 0; W2[:, a] = 0
        ok2, r2 = _verify(U2, V2, W2)
        if ok2 and r2 < best_rank:
            U, V, W = U2, V2, W2
            active = [t for t in range(R) if np.any(U[:, t]) and np.any(V[:, t]) and np.any(W[:, t])]
            best_rank = r2

    # Try dropping pairs
    for _ in range(2000):
        if not _time_ok():
            break
        if len(active) <= 2:
            break
        idx1, idx2 = rng.choice(len(active), size=2, replace=False)
        a, b = active[idx1], active[idx2]
        U2 = U.copy(); V2 = V.copy(); W2 = W.copy()
        U2[:, a] = 0; V2[:, a] = 0; W2[:, a] = 0
        U2[:, b] = 0; V2[:, b] = 0; W2[:, b] = 0
        ok2, r2 = _verify(U2, V2, W2)
        if ok2 and r2 < best_rank:
            U, V, W = U2, V2, W2
            active = [t for t in range(R) if np.any(U[:, t]) and np.any(V[:, t]) and np.any(W[:, t])]
            best_rank = r2

    ok, r = _verify(U, V, W)
    if not ok:
        return base_U.copy(), base_V.copy(), base_W.copy()
    return U, V, W


def construct():
    base_U, base_V, base_W = _build_base()
    # quick sanity check
    ok, r = _verify(base_U, base_V, base_W)
    if not ok:
        # Should not happen; return base anyway
        return {"U": base_U.tolist(), "V": base_V.tolist(), "W": base_W.tolist()}

    U, V, W = _try_reduce(base_U, base_V, base_W)

    # Final verification; if broken, fall back to base
    ok, r = _verify(U, V, W)
    if not ok:
        U, V, W = base_U, base_V, base_W

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
