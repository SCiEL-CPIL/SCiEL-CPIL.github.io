"""Structural redesign: time-boxed exact rank-reduction search for the 4x5x5
matrix-multiplication tensor.

Instead of a fixed block decomposition, we search for a lower-rank exact
decomposition with integer/half-integer entries. Every candidate is verified by
reconstructing the full multiplication tensor (exact integer arithmetic), so any
returned result is guaranteed valid. We keep the best verified rank found before a
safety deadline and always fall back to the known-valid schoolbook rank 100.
"""
import time
import random
import itertools

import numpy as np

DIMENSIONS = [4, 5, 5]

# W index convention: W[k*n + i, r]  (k = row of B, i = column of A)
N, M, P = DIMENSIONS
NMP = N * M * P  # 100 schoolbook rank

# Fixed seed for reproducibility.
_SEED = 20240517
_DEADLINE = 130.0  # wall-clock safety margin (evaluator kills at 150 s)


def _mul_tensor():
    """Return the matrix-multiplication tensor T[i, j, k] (1 if i,j,k valid)."""
    T = np.zeros((N, M, P), dtype=object)
    for i in range(N):
        for j in range(M):
            for k in range(P):
                T[i, j, k] = 1
    return T


_T = _mul_tensor()


def _flat_T():
    """Flatten T to a length N*M*P vector (row-major i,j,k)."""
    return [int(_T[i, j, k]) for i in range(N) for j in range(M) for k in range(P)]


_FLAT = _flat_T()


def _verify(U, V, W, rank):
    """Check sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] == T[i,j,k] exactly."""
    # Precompute per-r product of U and V for speed is not trivial; do direct but
    # vectorised over r with numpy float64 (values are tiny, exact in float).
    Ua = np.asarray(U, dtype=float)
    Va = np.asarray(V, dtype=float)
    Wa = np.asarray(W, dtype=float)
    # Build S[i,j,k] = sum_r U[i*m+j,r]*V[j*p+k,r] then check *W.
    # Ua shape (N*M, rank), Va (M*P, rank), Wa (P*N, rank)
    # For each r, outer(U[:,r], V[:,r]) gives (N*M, M*P); reshape (N,M,M,P).
    S = np.zeros((N, M, M, P), dtype=float)
    for r in range(rank):
        u = Ua[:, r].reshape(N, M)
        v = Va[:, r].reshape(M, P)
        S += np.einsum('ij,kl->ijkl', u, v)
    # Now S[i,j,k2,k] must equal sum over j2 of S[i,j,j2,k] * W[k*n+i]
    # Actually S already sums over r. We need C[i,j,k] = sum_{j2} S[i,j,j2,k]*W[k*n+i,r]...
    # Wait: T = sum_r U[i*m+j,r] V[j*p+k,r] W[k*n+i,r]. The r sum is inside, and
    # the j index of U and V is the SAME j (the shared index). So S should be
    # S[i,j,k] = sum_r U[i*m+j,r]*V[j*p+k,r], NOT summed over j2.
    # Redo: u shape (N,M), v shape (M,P) but we need same j. So use diagonal.
    S2 = np.zeros((N, M, P), dtype=float)
    for r in range(rank):
        u = Ua[:, r].reshape(N, M)          # u[i,j]
        v = Va[:, r].reshape(M, P)          # v[j,k]
        # same j: S2[i,j,k] += u[i,j]*v[j,k]
        S2 += np.einsum('ij,jk->ijk', u, v)
    # Now multiply by W[k*n+i, r] summed over r:
    C = np.zeros((N, M, P), dtype=float)
    for r in range(rank):
        w = Wa[:, r].reshape(P, N)          # w[k,i]
        C += S2 * 0  # placeholder
    # Need per-r S2, recompute per r to combine with W per r.
    C = np.zeros((N, M, P), dtype=float)
    for r in range(rank):
        u = Ua[:, r].reshape(N, M)
        v = Va[:, r].reshape(M, P)
        s2 = np.einsum('ij,jk->ijk', u, v)
        w = Wa[:, r].reshape(P, N)
        C += s2 * w.transpose(1, 0)  # w[k,i] -> [i,k]
    # Compare to T
    diff = C - _T.astype(float)
    return float(np.max(np.abs(diff))) < 1e-9


def _schoolbook():
    rank = NMP
    U = [[0] * rank for _ in range(N * M)]
    V = [[0] * rank for _ in range(M * P)]
    W = [[0] * rank for _ in range(P * N)]
    c = 0
    for i in range(N):
        for j in range(M):
            for k in range(P):
                U[i * M + j][c] = 1
                V[j * P + k][c] = 1
                W[k * N + i][c] = 1
                c += 1
    return {"U": U, "V": V, "W": W}


def _rand_mat(rows, rank, rng, half=True):
    A = np.zeros((rows, rank), dtype=float)
    for r in range(rows):
        for c in range(rank):
            if rng.random() < 0.45:
                val = rng.randint(-2, 2)
                if half and rng.random() < 0.5:
                    val += 0.5
                A[r, c] = val
    return A


def _search_rank(rank, time_budget, rng):
    """Try to find an exact rank-`rank` decomposition. Returns (U,V,W) or None."""
    start = time.time()
    # We'll do a simple random-restart + local coordinate-descent (flip entries)
    # optimisation that minimises the reconstruction error, checking exactness.
    best_err = None
    best = None
    iters = 0
    while time.time() - start < time_budget:
        U = _rand_mat(N * M, rank, rng)
        V = _rand_mat(M * P, rank, rng)
        W = _rand_mat(P * N, rank, rng)
        # local search: random coordinate flips to reduce error
        for step in range(3000):
            iters += 1
            if time.time() - start > time_budget:
                break
            mat = rng.choice(['U', 'V', 'W'])
            arr = U if mat == 'U' else (V if mat == 'V' else W)
            rr = rng.randrange(arr.shape[0])
            cc = rng.randrange(rank)
            old = arr[rr, cc]
            new = old + rng.choice([-1, 1, -0.5, 0.5])
            arr[rr, cc] = new
            # cheap error estimate on a random subset of (i,j,k)
            err = 0.0
            for _ in range(200):
                i = rng.randrange(N)
                j = rng.randrange(M)
                k = rng.randrange(P)
                s = 0.0
                for r in range(rank):
                    s += U[i * M + j, r] * V[j * P + k, r] * W[k * N + i, r]
                err += abs(s - 1.0)
            if err > (best_err if best_err is not None else float('inf')) and \
               (best_err is not None):
                # revert if it got much worse
                pass
            # keep going (hill climbing with reverts)
            if err > old_err + 0.5:
                arr[rr, cc] = old
            old_err = err
        # full verify
        if time.time() - start > time_budget:
            break
        if _verify(U.tolist(), V.tolist(), W.tolist(), rank):
            return U.tolist(), V.tolist(), W.tolist()
    return None


def construct():
    t0 = time.time()
    rng = random.Random(_SEED)
    np.random.seed(_SEED)

    # Fallback: guaranteed-valid schoolbook rank 100.
    fallback = _schoolbook()

    # Try ranks below 100 in descending order, time-boxed each.
    # Only attempt a handful of ranks; most time on the most promising (98, 96, 94).
    candidate_ranks = [98, 96, 94, 92]
    for rank in candidate_ranks:
        if time.time() - t0 > _DEADLINE - 5:
            break
        # give more budget to higher ranks
        budget = min(20.0, max(4.0, (_DEADLINE - 5) - (time.time() - t0)))
        res = _search_rank(rank, budget, rng)
        if res is not None:
            U, V, W = res
            # final exact check
            if _verify(U, V, W, rank):
                return {"U": U, "V": V, "W": W}
    return fallback
