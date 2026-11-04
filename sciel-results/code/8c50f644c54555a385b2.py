"""Matrix multiplication 2x4x6 rank-minimal exact tensor decomposition.

We search for a rank-r decomposition of the multiplication tensor
T[i,j,k] (i in [0,n), j in [0,m), k in [0,p)) of the form
    T = sum_c  U[:,c] * V[:,c] * W[:,c]
where the output factor uses the trace convention W[k*n+i, c].
We try decreasing ranks and verify exactness with integer/half-integer rounding.
"""
import time
import numpy as np

DIMENSIONS = [2, 4, 6]
_START = time.time()
_DEADLINE = 120.0 - 20.0  # safety margin before the 140s kill


def _time_left():
    return _DEADLINE - (time.time() - _START)


def _round_half(x):
    # round to nearest half-integer
    return np.round(x * 2.0) / 2.0


def _try_construct(n, m, p, rank, rng, max_iters=400):
    """Attempt to find an exact rank-`rank` decomposition via iterative
    multiplicative-weight / ALS-style search with half-integer rounding."""
    # Tensor in 3D form T[i, j, k]
    T = np.zeros((n, m, p))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i, j, k] = 1.0  # multiplication tensor: A[i,j]*B[j,k] -> C[i,k]
    # Actually the multiplication tensor is T[i,j,k] = 1 for all i,j,k (the
    # product C[i,k] = sum_j A[i,j] B[j,k] is a sum over j of products, so each
    # (i,j,k) contributes A[i,j]*B[j,k]). Thus T is all-ones.
    #
    # A rank-r decomposition of the all-ones tensor of shape (n,m,p) is exactly
    # the problem of writing the 3-way all-ones tensor as a sum of r rank-1
    # tensors. This is a known combinatorial object. We search for it.

    # Random init of factor matrices
    U = rng.normal(size=(n * m, rank))
    V = rng.normal(size=(m * p, rank))
    W = rng.normal(size=(p * n, rank))

    best_err = None
    best = None
    for it in range(max_iters):
        if _time_left() < 5.0:
            break
        # ALS: fix two, solve for third by least squares
        for _ in range(3):
            # solve U
            # T_unfold over (i*m+j) axis: for each (i,j), T[i,j,:] contracted
            # with V and W. Build matrix M_U[(i,j), c] = sum_{k} V[j*p+k, c] * W[k*n+i, c]
            MU = np.zeros((n * m, rank))
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        MU[i * m + j] += V[j * p + k] * W[k * n + i]
            # T row for (i,j) is all-ones of length p, so target = sum_k 1 = p
            target_U = np.full(n * m, float(p))
            U = np.linalg.lstsq(MU, target_U, rcond=None)[0]
            # solve V
            MV = np.zeros((m * p, rank))
            for j in range(m):
                for k in range(p):
                    for i in range(n):
                        MV[j * p + k] += U[i * m + j] * W[k * n + i]
            target_V = np.full(m * p, float(n))
            V = np.linalg.lstsq(MV, target_V, rcond=None)[0]
            # solve W
            MW = np.zeros((p * n, rank))
            for k in range(p):
                for i in range(n):
                    for j in range(m):
                        MW[k * n + i] += U[i * m + j] * V[j * p + k]
            target_W = np.full(p * n, float(m))
            W = np.linalg.lstsq(MW, target_W, rcond=None)[0]

        # evaluate exact reconstruction error
        # C[i,k] = sum_j sum_c U[i*m+j,c] V[j*p+k,c] W[k*n+i,c]
        # should equal sum_j A[i,j]B[j,k] = A[i,j]*B[j,k] for the tensor identity.
        # The tensor identity requires: for all i,j,k, the rank-1 sum equals T[i,j,k]=1.
        # i.e. sum_c U[i*m+j,c] V[j*p+k,c] W[k*n+i,c] == 1 for every (i,j,k).
        err = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = float(np.dot(U[i * m + j], V[j * p + k] * W[k * n + i]))
                    err += (val - 1.0) ** 2
        err = float(np.sqrt(err))
        if best_err is None or err < best_err:
            best_err = err
            best = (U.copy(), V.copy(), W.copy())
        if err < 1e-7:
            break

    U, V, W = best
    # round to half-integers
    Ur = _round_half(U)
    Vr = _round_half(V)
    Wr = _round_half(W)
    # verify exactness after rounding
    ok = True
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = float(np.dot(Ur[i * m + j], Vr[j * p + k] * Wr[k * n + i]))
                if abs(val - 1.0) > 1e-9:
                    ok = False
                    break
            if not ok:
                break
        if not ok:
            break
    if ok:
        return {"U": Ur.tolist(), "V": Vr.tolist(), "W": Wr.tolist()}
    return None


def _schoolbook(n, m, p):
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    c = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][c] = 1
                V[j * p + k][c] = 1
                W[k * n + i][c] = 1
                c += 1
    return {"U": U, "V": V, "W": W}


def construct():
    n, m, p = DIMENSIONS
    fallback = _schoolbook(n, m, p)
    best = fallback
    best_rank = n * m * p
    rng = np.random.default_rng(12345)
    # try ranks below schoolbook
    for rank in range(best_rank - 1, 0, -1):
        if _time_left() < 15.0:
            break
        # a few random restarts per rank
        for attempt in range(4):
            if _time_left() < 10.0:
                break
            res = _try_construct(n, m, p, rank, rng, max_iters=200)
            if res is not None:
                best = res
                best_rank = rank
                break
    return best
