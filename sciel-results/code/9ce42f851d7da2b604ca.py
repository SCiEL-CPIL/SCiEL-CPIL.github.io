"""Matrix multiplication 2x4x5 tensor rank minimization.

Entry point: construct() -> {'U','V','W'} with integer/half-integer entries
that exactly decompose the (2,4,5) matrix-multiplication tensor.
Score = seed / rank, so minimizing rank is the goal.
"""

import time
import random
import numpy as np

DIMENSIONS = [2, 4, 5]

# Deadline guard: finish well before the 150s wall-clock kill.
_START = time.time()
_DEADLINE = 125.0  # safety margin below 150s


def _time_left():
    return _DEADLINE - (time.time() - _START)


def _build_tensor():
    """Return the flattened multiplication tensor T of size n*m*p.

    T is indexed so that the decomposition
        sum_c U[a,c]*V[b,c]*W[d,c]  ==  T[a,b,d]
    with a = i*m+j, b = j*p+k, d = k*n+i.
    """
    n, m, p = DIMENSIONS
    T = np.zeros((n * m, m * p, p * n), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                d = k * n + i
                T[a, b, d] = 1.0
    return T


def _schoolbook():
    n, m, p = DIMENSIONS
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    c = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, c] = 1.0
                V[j * p + k, c] = 1.0
                W[k * n + i, c] = 1.0
                c += 1
    return U, V, W


def _verify(U, V, W, T, tol=1e-6):
    """Check that sum_c U[:,c] V[:,c] W[:,c] == T."""
    r = U.shape[1]
    rec = np.einsum('ac,bc,dc->abd', U, V, W)
    return np.allclose(rec, T, atol=tol, rtol=0)


def _to_halfint(x):
    """Round to nearest half-integer."""
    return np.round(x * 2.0) / 2.0


def _greedy_merge(U, V, W, T, max_iter=20000):
    """Try to reduce rank by merging columns whose triples are nearly
    linearly dependent. Returns a lower-rank (U,V,W) if found, else None."""
    n, m, p = DIMENSIONS
    U = U.copy(); V = V.copy(); W = W.copy()
    r = U.shape[1]
    improved = True
    iters = 0
    while improved and iters < max_iter and _time_left() > 5:
        improved = False
        iters += 1
        # Build the list of triples as vectors in R^{nm+mp+pn}
        nm, mp, pn = n * m, m * p, p * n
        vecs = []
        for c in range(r):
            v = np.concatenate([U[:, c], V[:, c], W[:, c]])
            vecs.append(v)
        vecs = np.array(vecs)  # shape (r, D)
        # Try to find a pair (c1,c2) that are linearly dependent (one ~ scalar*other)
        best = None
        best_ratio = None
        for c1 in range(r):
            for c2 in range(c1 + 1, r):
                # check if vecs[c2] is a scalar multiple of vecs[c1]
                nz = np.abs(vecs[c1]) > 1e-9
                if not np.any(nz):
                    continue
                # solve scalar a such that vecs[c2] ~= a * vecs[c1]
                a = np.dot(vecs[c2][nz], vecs[c1][nz]) / (np.dot(vecs[c1][nz], vecs[c1][nz]) + 1e-12)
                resid = np.linalg.norm(vecs[c2] - a * vecs[c1])
                scale = max(np.linalg.norm(vecs[c1]), np.linalg.norm(vecs[c2]), 1e-9)
                rel = resid / scale
                if rel < 1e-4:
                    if best is None or rel < best_ratio:
                        best = (c1, c2, a)
                        best_ratio = rel
        if best is not None:
            c1, c2, a = best
            # remove column c2 (it's ~ a * column c1)
            U = np.delete(U, c2, axis=1)
            V = np.delete(V, c2, axis=1)
            W = np.delete(W, c2, axis=1)
            r = U.shape[1]
            improved = True
            # re-verify
            if not _verify(U, V, W, T):
                # restore
                U = np.insert(U, c2, a * U[:, c1] if c1 < c2 else a * U[:, c1], axis=1)
                # careful: after delete, indices shift; simpler to abort this merge
                return None
    return U, V, W


def _random_search(T, target_rank, n_restarts=50, iters_per=300):
    """Random alternating-minimization search for a rank-target decomposition
    with small integer/half-integer entries."""
    n, m, p = DIMENSIONS
    nm, mp, pn = n * m, m * p, p * n
    best = None
    for rs in range(n_restarts):
        if _time_left() < 10:
            break
        rng = random.Random(12345 + rs)
        # Initialize with small random integer-ish values
        U = np.array([rng.choice([-2, -1, 0, 1, 2]) for _ in range(nm * target_rank)]).reshape(nm, target_rank)
        V = np.array([rng.choice([-2, -1, 0, 1, 2]) for _ in range(mp * target_rank)]).reshape(mp, target_rank)
        W = np.array([rng.choice([-2, -1, 0, 1, 2]) for _ in range(pn * target_rank)]).reshape(pn, target_rank)
        # Alternating least squares to fit T
        for it in range(iters_per):
            if _time_left() < 10:
                break
            # Fix V,W solve U:  for each a,b,d: T[a,b,d] = sum_c U[a,c] V[b,c] W[d,c]
            # U[a,:] = argmin sum_{b,d} (T[a,b,d] - U[a,:] . (V[b,:]*W[d,:]))^2
            # Build M_U[a, c] = sum_{b,d} V[b,c] W[d,c] * T[a,b,d]  and G_U[a,c,c'] = sum_{b,d} V[b,c]V[b,c'] W[d,c]W[d,c']
            # Simpler: use per-column projection
            # Solve U column by column using normal equations over (b,d)
            # We'll do a full ALS with numpy for speed.
            # U: minimize ||T - U V W||_F. For fixed V,W:
            #   U[a,c] = <T[a,:,:], V[:,c]*W[:,c]> / <V[:,c]*W[:,c], V[:,c]*W[:,c]>  (per column, decoupled)
            for c in range(target_rank):
                vw = V[:, c] * W[:, c]  # shape (mp*p? no) -> careful shapes
            # Shapes: V is (mp, r), W is (pn, r). The product V[b,c]*W[d,c] is over b in [0,mp), d in [0,pn).
            # T[a,b,d]. For column c, define q_c[b,d] = V[b,c]*W[d,c]. Then
            #   U[a,c] = sum_{b,d} T[a,b,d]*q_c[b,d] / sum_{b,d} q_c[b,d]^2
            for c in range(target_rank):
                qc = np.outer(V[:, c], W[:, c])  # (mp, pn)
                num = np.einsum('abd,bd->a', T, qc)  # (nm,)
                den = np.sum(qc * qc)
                U[:, c] = num / (den + 1e-12)
            # V
            for c in range(target_rank):
                qv = np.outer(U[:, c], W[:, c])  # (nm, pn) but need (nm, pn)? V[b,c] couples a and d
                # V[b,c] = sum_{a,d} T[a,b,d]*U[a,c]*W[d,c] / sum_{a,d} U[a,c]^2 W[d,c]^2
                qv = np.outer(U[:, c], W[:, c])  # (nm, pn)
                num = np.einsum('abd,ad->b', T, qv)
                den = np.sum(qv * qv)
                V[:, c] = num / (den + 1e-12)
            # W
            for c in range(target_rank):
                qw = np.outer(U[:, c], V[:, c])  # (nm, mp)
                num = np.einsum('abd,ab->d', T, qw)
                den = np.sum(qw * qw)
                W[:, c] = num / (den + 1e-12)
        # After ALS, round to half-integers and verify
        U2 = _to_halfint(U); V2 = _to_halfint(V); W2 = _to_halfint(W)
        if _verify(U2, V2, W2, T):
            best = (U2, V2, W2)
            break
    return best


def _strassen_block(T):
    """Use a Strassen-style block decomposition to get a candidate lower-rank
    factorization, then polish. Returns (U,V,W) or None."""
    n, m, p = DIMENSIONS
    nm, mp, pn = n * m, m * p, p * n
    # Start from schoolbook and attempt merges + small perturbations.
    U, V, W = _schoolbook()
    # Try greedy merge
    res = _greedy_merge(U, V, W, T)
    if res is not None and _verify(res[0], res[1], res[2], T):
        return res
    return None


def construct():
    T = _build_tensor()
    # Fallback: schoolbook always valid.
    U, V, W = _schoolbook()
    best_rank = U.shape[1]
    best = (U, V, W)

    # 1) Try Strassen-block + merge
    cand = _strassen_block(T)
    if cand is not None:
        U2, V2, W2 = cand
        if U2.shape[1] < best_rank and _verify(U2, V2, W2, T):
            best_rank = U2.shape[1]
            best = (U2, V2, W2)

    # 2) Random search for progressively lower ranks
    for target_rank in range(best_rank - 1, max(20, best_rank - 12), -1):
        if _time_left() < 15:
            break
        cand = _random_search(T, target_rank, n_restarts=40, iters_per=200)
        if cand is not None:
            U2, V2, W2 = cand
            if U2.shape[1] < best_rank:
                best_rank = U2.shape[1]
                best = (U2, V2, W2)
            break  # found one at this rank; try to go lower next loop

    Uf, Vf, Wf = best
    # Convert to Python lists of int/half-int
    def to_list(A):
        return [[float(x) for x in row] for row in A]
    return {"U": to_list(Uf), "V": to_list(Vf), "W": to_list(Wf)}
