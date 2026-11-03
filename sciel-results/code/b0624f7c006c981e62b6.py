"""Matrix multiplication 3x4 by 4x6: structured Strassen decomposition + rank-reduction search."""
import time
import numpy as np

n, m, p = 3, 4, 6

_START_TIME = time.time()
_DEADLINE = 140.0

def _time_left():
    return _DEADLINE - (time.time() - _START_TIME)

def _time_ok():
    return _time_left() > 10.0


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


def _reconstruct_fast(U, V, W):
    """Vectorized reconstruction of the multiplication tensor."""
    rank = U.shape[1]
    C = np.zeros((n, m, p), dtype=int)
    for r in range(rank):
        u = U[:, r].reshape(n, m)
        v = V[:, r].reshape(m, p)
        w = W[:, r].reshape(p, n)
        if not (np.any(u) and np.any(v) and np.any(w)):
            continue
        # Contribution: C[i,j,k] += U[i,j,r] * V[j,k,r] * W[k,i,r]
        # This is a rank-1 tensor update: outer product structure
        # C += u[:, :, None] * v[None, :, :] * w[None, None, :]
        # But w is indexed by (k, i), so w[k, i]
        # C[i,j,k] += U[i,j,r] * V[j,k,r] * W[k,i,r]
        # = u[i,j] * v[j,k] * w[k,i]
        # We can compute this as: for each i, C[i,:,:] += u[i,:] * (v * w[:, i, None])^T? No.
        # Let's just do it element-wise but vectorized over the product
        # temp[i,j,k] = u[i,j] * v[j,k] * w[k,i]
        # This is: einsum('ij,jk,ki->ijk', u, v, w)
        C += np.einsum('ij,jk,ki->ijk', u, v, w)
    return C


def _target_tensor():
    C = np.zeros((n, m, p), dtype=int)
    C[:] = 1
    return C


def _rank_of(U, V, W):
    """Return the effective (non-zero) rank."""
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
    recon = _reconstruct_fast(U, V, W)
    ok = np.array_equal(recon, target)
    return ok, _rank_of(U, V, W)


def _try_reduce(base_U, base_V, base_W):
    """Search for a lower-rank exact decomposition using redundancy elimination."""
    U = base_U.copy()
    V = base_V.copy()
    W = base_W.copy()
    R = U.shape[1]
    target = _target_tensor()

    # Precompute tensor contributions for each column
    contribs = []
    for r in range(R):
        u = U[:, r].reshape(n, m)
        v = V[:, r].reshape(m, p)
        w = W[:, r].reshape(p, n)
        c = np.einsum('ij,jk,ki->ijk', u, v, w)
        contribs.append(c)

    def is_redundant(idx):
        """Check if column idx is an integer linear combination of the other columns."""
        contrib = contribs[idx]
        vec = contrib.reshape(-1)
        nz = np.nonzero(vec)[0]
        if len(nz) == 0:
            return False

        # Build matrix of other columns' contributions
        others = [t for t in range(R) if t != idx]
        A = np.zeros((len(nz), len(others)), dtype=float)
        for ci, t in enumerate(others):
            A[:, ci] = contribs[t].reshape(-1)[nz]
        b = vec[nz].astype(float)

        try:
            sol, *_ = np.linalg.lstsq(A, b, rcond=None)
        except Exception:
            return False

        # Check integer-ness with tighter tolerance
        rounded = np.round(sol)
        if not np.allclose(sol, rounded, atol=1e-8):
            return False

        # Verify exactly
        coeffs = rounded.astype(int)
        total = np.zeros((n, m, p), dtype=int)
        for ci, t in enumerate(others):
            if coeffs[ci] == 0:
                continue
            total += coeffs[ci] * contribs[t]

        return np.array_equal(total, contrib)

    # Iteratively remove redundant columns
    changed = True
    while changed and _time_ok():
        changed = False
        for idx in range(R):
            if not (np.any(U[:, idx]) and np.any(V[:, idx]) and np.any(W[:, idx])):
                U[:, idx] = 0; V[:, idx] = 0; W[:, idx] = 0
                contribs[idx] = np.zeros((n, m, p), dtype=int)
                changed = True
                continue
            if is_redundant(idx):
                U[:, idx] = 0; V[:, idx] = 0; W[:, idx] = 0
                contribs[idx] = np.zeros((n, m, p), dtype=int)
                changed = True
                if not _time_ok():
                    break

    # Verify
    ok, r = _verify(U, V, W)
    if not ok:
        return base_U.copy(), base_V.copy(), base_W.copy()

    # Try to remove more by checking random single active columns
    active = [t for t in range(R) if np.any(U[:, t]) and np.any(V[:, t]) and np.any(W[:, t])]
    best_rank = len(active)
    rng = np.random.default_rng(12345)

    for _ in range(5000):
        if not _time_ok():
            break
        if not active:
            break
        a = active[rng.integers(len(active))]
        U2 = U.copy(); V2 = V.copy(); W2 = W.copy()
        U2[:, a] = 0; V2[:, a] = 0; W2[:, a] = 0
        ok2, r2 = _verify(U2, V2, W2)
        if ok2 and r2 < best_rank:
            U, V, W = U2, V2, W2
            active = [t for t in range(R) if np.any(U[:, t]) and np.any(V[:, t]) and np.any(W[:, t])]
            best_rank = r2
            # Recompute contribs for active columns
            for t in active:
                u = U[:, t].reshape(n, m)
                v = V[:, t].reshape(m, p)
                w = W[:, t].reshape(p, n)
                contribs[t] = np.einsum('ij,jk,ki->ijk', u, v, w)

    ok, r = _verify(U, V, W)
    if not ok:
        return base_U.copy(), base_V.copy(), base_W.copy()
    return U, V, W


def construct():
    base_U, base_V, base_W = _build_base()
    ok, r = _verify(base_U, base_V, base_W)
    if not ok:
        return {"U": base_U.tolist(), "V": base_V.tolist(), "W": base_W.tolist()}

    U, V, W = _try_reduce(base_U, base_V, base_W)

    ok, r = _verify(U, V, W)
    if not ok:
        U, V, W = base_U, base_V, base_W

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
