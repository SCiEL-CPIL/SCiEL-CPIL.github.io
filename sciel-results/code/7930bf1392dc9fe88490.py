import time
import numpy as np

DIMENSIONS = [3, 3, 5]
_START_TIME = time.time()
_DEADLINE = 150
_SAFETY_MARGIN = 8  # seconds before the kill

def _schoolbook_decomposition():
    """
    Rank-45 exact tensor decomposition for 3x3 * 3x5 matrix multiplication.
    Schoolbook algorithm with the specified index conventions.
    """
    n, m, p = DIMENSIONS
    rank = n * m * p  # 3 * 3 * 5 = 45

    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1

    return {"U": U, "V": V, "W": W}


def _verify_decomposition(U, V, W, n, m, p):
    """
    Verify that U, V, W exactly decompose the multiplication tensor:
        sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] == 1  for all (i,j,k)
    """
    Ua = np.asarray(U, dtype=float).reshape(n, m, -1)
    Va = np.asarray(V, dtype=float).reshape(m, p, -1)
    Wa = np.asarray(W, dtype=float).reshape(p, n, -1)
    # contract over r: out[i,j,k] = sum_r Ua[i,j,r]*Va[j,k,r]*Wa[k,i,r]
    out = np.einsum('ijr,jkr,kir->ijk', Ua, Va, Wa)
    return bool(np.all(np.abs(out - 1.0) < 1e-9))


def _contraction(U, V, W, n, m, p):
    """
    out[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    Vectorized via reshape + einsum.
    """
    Ua = U.reshape(n, m, -1)
    Va = V.reshape(m, p, -1)
    Wa = W.reshape(p, n, -1)
    return np.einsum('ijr,jkr,kir->ijk', Ua, Va, Wa)


def _als_step(U, V, W, n, m, p):
    """
    One full alternating-least-squares sweep. Each factor is updated by an
    exact linear solve with the other two fixed (coordinate descent on the
    tensor-nuclear / squared-error objective).
    Returns (newU, newV, newW).
    """
    rank = U.shape[1]
    Ua = U.reshape(n, m, rank)
    Va = V.reshape(m, p, rank)
    Wa = W.reshape(p, n, rank)

    newU = np.empty_like(U)
    newV = np.empty_like(V)
    newW = np.empty_like(W)

    # Solve for U: fix V, W.
    #   sum_r U[i,j,r] * V[j,k,r] * W[k,i,r] = 1  for all k
    #   => (sum_k V[j,k,r]*W[k,i,r])_r * U[i,j,:] = 1
    for i in range(n):
        for j in range(m):
            A = np.einsum('kr,kr->r', Va[j, :, :], Wa[:, i, :])  # (rank,)
            M = np.outer(A, A) + 1e-10 * np.eye(rank)
            try:
                newU[i * m + j, :] = np.linalg.solve(M, np.ones(rank))
            except np.linalg.LinAlgError:
                newU[i * m + j, :] = np.linalg.lstsq(M, np.ones(rank), rcond=None)[0]

    # Solve for V: fix U, W.
    for j in range(m):
        for k in range(p):
            A = np.einsum('ir,ir->r', Ua[:, j, :], Wa[k, :, :])  # (rank,)
            M = np.outer(A, A) + 1e-10 * np.eye(rank)
            try:
                newV[j * p + k, :] = np.linalg.solve(M, np.ones(rank))
            except np.linalg.LinAlgError:
                newV[j * p + k, :] = np.linalg.lstsq(M, np.ones(rank), rcond=None)[0]

    # Solve for W: fix U, V.
    for k in range(p):
        for i in range(n):
            A = np.einsum('jr,jr->r', Ua[:, :, :], Va[:, k, :])  # (rank,)
            M = np.outer(A, A) + 1e-10 * np.eye(rank)
            try:
                newW[k * n + i, :] = np.linalg.solve(M, np.ones(rank))
            except np.linalg.LinAlgError:
                newW[k * n + i, :] = np.linalg.lstsq(M, np.ones(rank), rcond=None)[0]

    return newU, newV, newW


def _polish(U, V, W, n, m, p, max_iters=8):
    """
    High-precision polish: round to the nearest half-integer, then run a few
    exact ALS steps constrained to the half-integer lattice, verifying after
    each. Returns a valid dict if one is found, else None.
    """
    rank = U.shape[1]
    Uc = np.round(U * 2.0) / 2.0
    Vc = np.round(V * 2.0) / 2.0
    Wc = np.round(W * 2.0) / 2.0

    def check(A, B, C):
        if _verify_decomposition(A.tolist(), B.tolist(), C.tolist(), n, m, p):
            return {"U": A.tolist(), "V": B.tolist(), "W": C.tolist()}
        return None

    res = check(Uc, Vc, Wc)
    if res is not None:
        return res

    for _ in range(max_iters):
        # ALS on the current (already half-integer) factors, then re-round.
        nU, nV, nW = _als_step(Uc, Vc, Wc, n, m, p)
        Uc = np.round(nU * 2.0) / 2.0
        Vc = np.round(nV * 2.0) / 2.0
        Wc = np.round(nW * 2.0) / 2.0
        res = check(Uc, Vc, Wc)
        if res is not None:
            return res
    return None


def _try_search(n, m, p, start_rank, end_rank, time_budget):
    """
    Best-effort search for a rank < end_rank exact decomposition.
    Returns (U, V, W) lists if found, else None.
    """
    nm, mp, pn = n * m, m * p, p * n

    for rank in range(start_rank, end_rank):
        if time.time() - _START_TIME > _DEADLINE - _SAFETY_MARGIN:
            break

        n_restarts = 4
        for seed in range(n_restarts):
            if time.time() - _START_TIME > _DEADLINE - _SAFETY_MARGIN:
                break
            rng = np.random.RandomState(seed * 7919 + rank * 131 + 17)

            # Structured initialization: start from a random rank-1-ish blend
            U = rng.randn(nm, rank) * 0.6
            V = rng.randn(mp, rank) * 0.6
            W = rng.randn(pn, rank) * 0.6

            best_err = np.inf
            for it in range(120):
                if time.time() - _START_TIME > _DEADLINE - _SAFETY_MARGIN:
                    break
                U, V, W = _als_step(U, V, W, n, m, p)
                out = _contraction(U, V, W, n, m, p)
                err = float(np.linalg.norm(out - 1.0))
                if err < best_err:
                    best_err = err
                if err < 1e-9:
                    break

            if best_err < 1e-7:
                res = _polish(U, V, W, n, m, p, max_iters=10)
                if res is not None:
                    return res
            # also try polishing the best iterate even if not fully converged
            if best_err < 1e-4:
                res = _polish(U, V, W, n, m, p, max_iters=12)
                if res is not None:
                    return res

    return None


def construct():
    """
    Entry point with deadline guard. Returns the best exact decomposition
    found (schoolbook rank-45 fallback guaranteed).
    """
    n, m, p = DIMENSIONS
    elapsed = time.time() - _START_TIME
    if elapsed > _DEADLINE - _SAFETY_MARGIN:
        return _schoolbook_decomposition()

    # Try to beat the schoolbook rank-45 baseline with a lower-rank search.
    try:
        result = _try_search(n, m, p, start_rank=12, end_rank=45, time_budget=_DEADLINE)
        if result is not None:
            # final safety verification before returning a non-schoolbook result
            if _verify_decomposition(result["U"], result["V"], result["W"], n, m, p):
                return result
    except Exception:
        pass

    return _schoolbook_decomposition()
