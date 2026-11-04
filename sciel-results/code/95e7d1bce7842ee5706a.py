import time
import numpy as np
import random

DIMENSIONS = [2, 3, 6]


def _build_target(n, m, p):
    """Return list of (a, b, c) valid triples and their values (all 1)."""
    triples = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                triples.append((a, b, c))
    return triples


def _verify(U, V, W, n, m, p, tol=1e-6):
    """Check exact decomposition. U:(nm,R) V:(mp,R) W:(pn,R)."""
    num_A, num_B, num_C = n * m, m * p, p * n
    Uf = np.asarray(U, dtype=float)
    Vf = np.asarray(V, dtype=float)
    Wf = np.asarray(W, dtype=float)
    R = Uf.shape[1]
    # valid entries must be 1
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                val = float(np.dot(np.dot(Uf[a], Vf[b]), Wf[c]))
                if abs(val - 1.0) > tol:
                    return False
    # invalid entries must be 0
    valid = set()
    for i in range(n):
        for j in range(m):
            for k in range(p):
                valid.add((i * m + j, j * p + k, k * n + i))
    for a in range(num_A):
        i1, j1 = a // m, a % m
        for b in range(num_B):
            j2, k1 = b // p, b % p
            if j1 != j2:
                continue
            for c in range(num_C):
                k2, i2 = c // n, c % n
                if (i1 == i2 and k1 == k2):
                    continue
                val = float(np.dot(np.dot(Uf[a], Vf[b]), Wf[c]))
                if abs(val) > tol:
                    return False
    return True


def _to_small_ints(M, max_abs=4):
    """Round a matrix to small integers, zeroing tiny values."""
    M = np.asarray(M, dtype=float)
    out = np.round(M).astype(int)
    out[np.abs(out) > max_abs] = 0
    out[np.abs(M) < 1e-6] = 0
    return out


def _try_sparse_recovery(U, V, W, n, m, p, tol=1e-6):
    """Attempt to snap U,V,W to small integers preserving the decomposition."""
    if not _verify(U, V, W, n, m, p, tol=1e-3):
        return None
    Uf = np.asarray(U, dtype=float)
    Vf = np.asarray(V, dtype=float)
    Wf = np.asarray(W, dtype=float)
    best = None
    for scale in (1.0, 2.0, 0.5):
        Ui = _to_small_ints(Uf * scale)
        Vi = _to_small_ints(Vf * scale)
        Wi = _to_small_ints(Wf * scale)
        if _verify(Ui, Vi, Wi, n, m, p, tol=1e-6):
            if best is None:
                best = (Ui, Vi, Wi)
            elif (np.abs(Ui).sum() + np.abs(Vi).sum() + np.abs(Wi).sum()) < \
                 (np.abs(best[0]).sum() + np.abs(best[1]).sum() + np.abs(best[2]).sum()):
                best = (Ui, Vi, Wi)
    return best


def _random_candidate(rank, n, m, p, rng):
    """Generate a random rank-`rank` candidate and return (U,V,W, residual)."""
    num_A, num_B, num_C = n * m, m * p, p * n
    U = rng.standard_normal((num_A, rank))
    V = rng.standard_normal((num_B, rank))
    W = rng.standard_normal((num_C, rank))
    # normalize columns to keep magnitudes sane
    for r in range(rank):
        Un = np.linalg.norm(U[:, r])
        Vn = np.linalg.norm(V[:, r])
        Wn = np.linalg.norm(W[:, r])
        if Un > 0 and Vn > 0 and Wn > 0:
            U[:, r] /= Un
            V[:, r] /= Vn
            W[:, r] /= Wn
    # residual on valid entries
    triples = _build_target(n, m, p)
    res = 0.0
    for (a, b, c) in triples:
        val = float(np.dot(np.dot(U[a], V[b]), W[c]))
        res += (val - 1.0) ** 2
    return U, V, W, res


def construct():
    start_time = time.time()
    deadline = start_time + 130  # safety margin before 150 s limit

    n, m, p = DIMENSIONS
    num_A, num_B, num_C = n * m, m * p, p * n
    rng = random.Random(12345)
    np_rng = np.random.default_rng(12345)

    best = None  # (rank, U, V, W, score)
    best_residual = float('inf')

    # ---- Phase 1: randomized rank search from 30 upward ----
    for rank in range(30, 37):
        if time.time() > deadline:
            break
        # number of random trials scales with remaining budget
        remaining = max(0.0, deadline - time.time())
        trials = max(20, int(remaining * 3.0))
        for t in range(trials):
            if time.time() > deadline:
                break
            U, V, W, res = _random_candidate(rank, n, m, p, np_rng)
            if res < best_residual:
                best_residual = res
            # occasional sparse recovery on promising candidates
            if res < 0.5 and t % 5 == 0:
                rec = _try_sparse_recovery(U, V, W, n, m, p)
                if rec is not None:
                    Ui, Vi, Wi = rec
                    if best is None or rank < best[0]:
                        best = (rank, Ui, Vi, Wi, res)
                        if rank <= 30:
                            # found a very good rank; keep searching briefly
                            pass
        # if we found a valid low rank, stop early
        if best is not None and best[0] <= 30:
            break

    if best is not None:
        r, U, V, W, _ = best
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    # ---- Phase 2: fallback to schoolbook (guaranteed valid) ----
    rank_school = n * m * p
    U_school = np.zeros((num_A, rank_school), dtype=int)
    V_school = np.zeros((num_B, rank_school), dtype=int)
    W_school = np.zeros((num_C, rank_school), dtype=int)
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                U_school[a, col] = 1
                V_school[b, col] = 1
                W_school[c, col] = 1
                col += 1
    return {"U": U_school.tolist(), "V": V_school.tolist(), "W": W_school.tolist()}
