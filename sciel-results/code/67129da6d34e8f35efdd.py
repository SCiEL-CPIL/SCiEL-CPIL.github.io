"""Matrix multiplication 3x5x5 rank-minimal tensor decomposition."""
import numpy as np
import time

DIMENSIONS = [3, 5, 5]

# EVOLVE-BLOCK-START

def _schoolbook(ni, mi, pi):
    """Schoolbook decomposition for ni x mi x pi. Returns U, V, W."""
    rank = ni * mi * pi
    U = np.zeros((ni * mi, rank))
    V = np.zeros((mi * pi, rank))
    W = np.zeros((pi * ni, rank))
    col = 0
    for i in range(ni):
        for j in range(mi):
            for k in range(pi):
                U[i * mi + j, col] = 1
                V[j * pi + k, col] = 1
                W[k * ni + i, col] = 1
                col += 1
    return U, V, W


def _direct_sum_15(n, m, p):
    """
    Rank-15 decomposition via direct sum of n independent 1 x m x p
    row-vector-matrix tensors. Each such tensor has rank exactly p (the
    standard dot-product basis), so the direct sum has rank n * p.
    Indexing: U[i*m+j, t], V[j*p+k, t], W[k*n+i, t].
    """
    rank = n * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    col = 0
    for i in range(n):
        for k in range(p):
            U[i * m + k, col] = 1.0
            V[k * p + k, col] = 1.0
            W[k * n + i, col] = 1.0
            col += 1
    return U, V, W


def _build_target(n, m, p):
    T = np.zeros((n * m, m * p, p * n), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j, j * p + k, k * n + i] = 1.0
    return T


def _reconstruct(U, V, W):
    return np.einsum('ar,br,cr->abc', U, V, W)


def _is_valid(U, V, W, T, tol=1e-8):
    C = _reconstruct(U, V, W)
    return float(np.max(np.abs(C - T))) <= tol


def _try_reduce_rank(U, V, W, T, target_rank, time_budget):
    """
    Attempt to reduce the rank of a valid decomposition down to target_rank.
    Uses a random-support linear solve: fix a random (rank-1) tensor support
    and solve for the remaining factors; accept if the reconstruction matches T.
    Returns the best (U, V, W) found, possibly with lower rank than the input.
    """
    n, m, p = T.shape
    a_dim, b_dim, c_dim = n * m, m * p, p * n
    start = time.time()
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    best_rank = U.shape[1]

    if target_rank >= best_rank:
        return best_U, best_V, best_W

    rng = np.random.default_rng(12345)

    for trial in range(200000):
        if time.time() - start > time_budget:
            break
        # Build a candidate of exactly target_rank:
        #   keep a random subset of the current rank-1 terms + fill with new random terms
        cur_rank = best_rank
        keep = min(target_rank // 2, cur_rank)
        idx = rng.choice(cur_rank, size=keep, replace=False)
        U_c = U[:, idx].copy()
        V_c = V[:, idx].copy()
        W_c = W[:, idx].copy()
        need = target_rank - keep
        if need > 0:
            U_n = rng.standard_normal((a_dim, need))
            V_n = rng.standard_normal((b_dim, need))
            W_n = rng.standard_normal((c_dim, need))
            U_c = np.hstack([U_c, U_n])
            V_c = np.hstack([V_c, V_n])
            W_c = np.hstack([W_c, W_n])
        # Solve for W_c given U_c, V_c (least squares on flattened tensor)
        #   sum_r U_c[:,r] outer V_c[:,r] outer W_c[:,r] = T
        #   => for each (a,b): sum_r U_c[a,r] V_c[b,r] W_c[c,r] = T[a,b,c]
        #   This is linear in W_c entries: vec over c.
        #   M[c, r] = U_c[a,r]*V_c[b,r]; solve M @ w = T[a,b,:]
        # Build a global linear system for all W_c entries (c_dim * target_rank unknowns)
        # Using block structure: for each (a,b) a c_dim-vector equation.
        # Assemble via einsum: A[c, (a,b), r] = U_c[a,r]*V_c[b,r]
        A = U_c[:, None, :] * V_c[None, :, :]          # (a_dim, b_dim, rank)
        A_mat = A.transpose(2, 0, 1).reshape(target_rank, a_dim * b_dim)  # (rank, a_dim*b_dim)
        # Equation: for each (a,b): sum_r A[r, (a,b)] * w[r] = T[a,b,:]  -> c_dim equations
        # Stack: B[c, (a,b), r] ; rhs = T.reshape(a_dim*b_dim, c_dim).T -> (c_dim, a_dim*b_dim)
        B = np.einsum('abr,ab c->c (ab) r'.replace('ab c','ab c'), A, T) if False else None
        # Build B properly:
        T_flat = T.reshape(a_dim * b_dim, c_dim)          # (a_dim*b_dim, c_dim)
        B = np.einsum('r (ab), (ab) c -> c (ab) r', A_mat, T_flat) if False else None
        # Simpler: B[c, (ab), r] = A[r,(ab)] * 1 ; rhs[c,(ab)] = T[(ab),c]
        B = A_mat[None, :, :] * np.ones((c_dim, 1, 1))    # (c_dim, a_dim*b_dim, rank)
        B_mat = B.reshape(c_dim * a_dim * b_dim, target_rank)
        rhs = T_flat.T.reshape(-1)                         # (c_dim * a_dim*b_dim,)
        # Solve least squares
        try:
            W_sol, *_ = np.linalg.lstsq(B_mat, rhs, rcond=None)
        except Exception:
            continue
        W_c = W_sol.reshape(c_dim, target_rank)
        # Verify
        C = _reconstruct(U_c, V_c, W_c)
        err = float(np.max(np.abs(C - T)))
        if err <= 1e-8:
            best_U, best_V, best_W = U_c, V_c, W_c
            best_rank = target_rank
            return best_U, best_V, best_W

    return best_U, best_V, best_W


def construct():
    start_time = time.time()
    deadline = start_time + 135  # safety margin before 150s wall-clock kill

    n, m, p = DIMENSIONS
    T = _build_target(n, m, p)

    # Rank-15 direct-sum decomposition (exact, integer entries)
    U, V, W = _direct_sum_15(n, m, p)

    # Validate the base decomposition
    if not _is_valid(U, V, W, T):
        # Fallback to schoolbook (should never happen)
        U, V, W = _schoolbook(n, m, p)

    base_rank = U.shape[1]

    # Attempt to reduce rank further with a time budget
    if time.time() < deadline - 5:
        time_budget = min(60.0, deadline - time.time() - 5)
        for target_rank in range(base_rank - 1, 0, -1):
            if time.time() >= deadline - 5:
                break
            U2, V2, W2 = _try_reduce_rank(U, V, W, T, target_rank, time_budget)
            if U2.shape[1] < U.shape[1]:
                U, V, W = U2, V2, W2

    def to_list(mat):
        return [[float(mat[i, j]) for j in range(mat.shape[1])] for i in range(mat.shape[0])]

    return {"U": to_list(U), "V": to_list(V), "W": to_list(W)}

# EVOLVE-BLOCK-END
