import time
import numpy as np
from scipy.optimize import minimize

DIMENSIONS = [2, 4, 8]


# EVOLVE-BLOCK-START
def _matmul_check(U, V, W, n, m, p):
    """Exact check that the tensor decomposition reconstructs the MM tensor.
    For every (i,j,k): sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] must equal 1.
    Returns (ok, max_abs_error).
    """
    Ua = np.array(U, dtype=float)
    Va = np.array(V, dtype=float)
    Wa = np.array(W, dtype=float)
    r = Ua.shape[1]
    Ur = Ua.reshape(n, m, r)
    Vr = Va.reshape(m, p, r)
    Wr = Wa.reshape(p, n, r)
    T = np.einsum('ijr,jkr,kin->ijk', Ur, Vr, Wr)
    err = np.max(np.abs(T - 1.0))
    return (err < 1e-9), float(err)


def _construct_rank32():
    """
    Rank-32 decomposition: 8 independent schoolbook 2x2 blocks.
    """
    n, m, p = DIMENSIONS
    rank = 32
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    col = 0
    for l in range(4):
        for rb in range(2):
            for i in range(2):
                for j_local in range(2):
                    for k_local in range(2):
                        r = col + (i * 4 + j_local * 2 + k_local)
                        gi = i
                        gj = (j_local if rb == 0 else 2 + j_local)
                        gk = 2 * l + k_local
                        U[gi * m + gj][r] = 1
                        V[gj * p + gk][r] = 1
                        W[gk * n + gi][r] = 1
            col += 4

    return {"U": U, "V": V, "W": W}


def _trim_zero_columns(U, V, W):
    """Remove rank columns that are all-zero in U, V, and W."""
    n_u = len(U)
    n_v = len(V)
    n_w = len(W)
    rank = len(U[0]) if n_u else 0
    keep = []
    for c in range(rank):
        if any(U[i][c] != 0 for i in range(n_u)) or \
           any(V[i][c] != 0 for i in range(n_v)) or \
           any(W[i][c] != 0 for i in range(n_w)):
            keep.append(c)
    if len(keep) == rank:
        return U, V, W, rank
    U2 = [[U[i][c] for c in keep] for i in range(n_u)]
    V2 = [[V[i][c] for c in keep] for i in range(n_v)]
    W2 = [[W[i][c] for c in keep] for i in range(n_w)]
    return U2, V2, W2, len(keep)


def _try_rank_search(n, m, p, target_rank, deadline):
    """
    Try to find a rank-`target_rank` decomposition using alternating minimization.
    Returns (U, V, W) as lists or None if not successful within deadline.
    """
    t0 = time.time()
    nU = n * m  # 8
    nV = m * p  # 32
    nW = p * n  # 16
    r = target_rank

    # Build target tensor T[i,j,k] = 1 for all i,j,k
    T = np.ones((n, m, p), dtype=float)

    best_err = np.inf
    best_params = None

    for attempt in range(20):
        if time.time() > deadline - 5:
            break

        # Random initialization
        Ua = np.random.randn(nU, r) * 0.5
        Va = np.random.randn(nV, r) * 0.5
        Wa = np.random.randn(nW, r) * 0.5

        # Alternating least squares iterations
        for it in range(200):
            if time.time() > deadline - 5:
                break

            # Update U: for each (i,j), solve U[i,j,:] to minimize ||T[i,j,:] - (V*W)[:,i,j]||
            # T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
            # For fixed V, W: T[i,j,:] = U[i*m+j,:] @ (V ⊙ W) where (V⊙W)[k,r] = V[j*p+k,r]*W[k*n+i,r]
            for i in range(n):
                for j in range(m):
                    # Build matrix M[k, r] = V[j*p+k, r] * W[k*n+i, r]
                    # Build matrix M[k, r] = V[j*p+k, r] * W[k*n+i, r]
                    M = np.zeros((p, r))
                    for k in range(p):
                        M[k, :] = Va[j*p+k, :] * Wa[k*n+i, :]
                    # Solve U[i*m+j, :] @ M.T = T[i,j,:]
                    # => M @ U[i*m+j, :].T = T[i,j,:]
                    # => U[i*m+j, :].T = M^{-1} @ T[i,j,:] (if M is square and invertible)
                    # Use lstsq
                    try:
                        sol, _, _, _ = np.linalg.lstsq(M.T, T[i, j, :], rcond=None)
                        Ua[i*m+j, :] = sol
                    except:
                        pass

            # Update V similarly
            for j in range(m):
                for k in range(p):
                    M = np.zeros((n, r))
                    for i in range(n):
                        M[i, :] = Ua[i*m+j, :] * Wa[k*n+i, :]
                    try:
                        sol, _, _, _ = np.linalg.lstsq(M.T, T[:, j, k], rcond=None)
                        Va[j*p+k, :] = sol
                    except:
                        pass

            # Update W similarly
            for k in range(p):
                for i in range(n):
                    M = np.zeros((m, r))
                    for j in range(m):
                        M[j, :] = Ua[i*m+j, :] * Va[j*p+k, :]
                    try:
                        sol, _, _, _ = np.linalg.lstsq(M.T, T[i, j, k], rcond=None)
                        Wa[k*n+i, :] = sol
                    except:
                        pass

            # Compute error
            Ur = Ua.reshape(n, m, r)
            Vr = Va.reshape(m, p, r)
            Wr = Wa.reshape(p, n, r)
            T_recon = np.einsum('ijr,jkr,kin->ijk', Ur, Vr, Wr)
            err = np.max(np.abs(T_recon - T))

            if err < best_err:
                best_err = err
                best_params = (Ua.copy(), Va.copy(), Wa.copy())

            if err < 1e-9:
                break

        if best_err < 1e-9:
            Ua, Va, Wa = best_params
            U_list = [[Ua[i, c] for c in range(r)] for i in range(nU)]
            V_list = [[Va[i, c] for c in range(r)] for i in range(nV)]
            W_list = [[Wa[i, c] for c in range(r)] for i in range(nW)]
            return U_list, V_list, W_list

    return None


def construct():
    n, m, p = DIMENSIONS
    t0 = time.time()
    deadline = t0 + 140  # safety margin

    # Try to find lower rank via search
    for target_rank in [24, 26, 28, 30]:
        if time.time() > deadline - 10:
            break
        try:
            result = _try_rank_search(n, m, p, target_rank, deadline)
            if result is not None:
                U_list, V_list, W_list = result
                ok, err = _matmul_check(U_list, V_list, W_list, n, m, p)
                if ok:
                    U2, V2, W2, r = _trim_zero_columns(U_list, V_list, W_list)
                    return {"U": U2, "V": V2, "W": W2}
        except Exception:
            pass

    # Fallback: rank-32 construction
    try:
        cand = _construct_rank32()
        ok, err = _matmul_check(cand["U"], cand["V"], cand["W"], n, m, p)
        if ok:
            U, V, W, r = _trim_zero_columns(cand["U"], cand["V"], cand["W"])
            return {"U": U, "V": V, "W": W}
    except Exception:
        pass

    # Final fallback: rank-64 schoolbook
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
