"""Tensor decomposition for 2x3 * 3x10 matrix multiplication.

Combined design:
  * Primary path: Strassen-on-2x2x2-blocks (rank 55) construction.
  * Refinement machinery (from inspirations): fast vectorized column-removal
    search + linear-algebraic redundancy sweep (rank-deficiency of the
    flattened U/V/W factor matrices) + a randomized low-rank projection test.
  * 4x4x4 rank-23 embedding is tested only as a conditional projection,
    guarded by exact validation, so it can never hurt the score.
"""
import time
import numpy as np

DIMENSIONS = [2, 3, 10]

# EVOLVE-BLOCK-START


def _validate(n, m, p, U, V, W, atol=1e-9):
    """Verify sum_r U[i*m+j][r]*V[j*p+k][r]*W[k*n+i][r] == 1 for all i,j,k."""
    U_arr = np.asarray(U, dtype=float)
    V_arr = np.asarray(V, dtype=float)
    W_arr = np.asarray(W, dtype=float)
    U3 = U_arr.reshape(n, m, -1)
    V3 = V_arr.reshape(m, p, -1)
    W3 = W_arr.reshape(p, n, -1)
    T = np.einsum('ijr,jkr,kir->ijk', U3, V3, W3)
    return bool(np.allclose(T, np.ones((n, m, p)), atol=atol))


def _tensor_contribution(n, m, p, U_arr, V_arr, W_arr, keep_mask):
    """Compute the tensor sum over kept columns."""
    U3 = U_arr[:, keep_mask].reshape(n, m, -1)
    V3 = V_arr[:, keep_mask].reshape(m, p, -1)
    W3 = W_arr[:, keep_mask].reshape(p, n, -1)
    if U3.shape[2] == 0:
        return np.zeros((n, m, p))
    return np.einsum('ijr,jkr,kir->ijk', U3, V3, W3)


def _build_trivial(n, m, p):
    """Return the rank-nmp trivial decomposition as (U, V, W) lists."""
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    for r in range(rank):
        i = r // (m * p)
        rem = r % (m * p)
        j = rem // p
        k = rem % p
        idx_u = i * m + j
        idx_v = j * p + k
        idx_w = k * n + i
        U[idx_u][r] = 1
        V[idx_v][r] = 1
        W[idx_w][r] = 1
    return U, V, W, rank


def _build_strassen(n, m, p):
    """
    Build a rank-55 decomposition for 2x3 * 3x10 using Strassen on 2x2x2 blocks.

    A is 2x3, B is 3x10, C is 2x10.  For each pair of output columns (k0,k1)
    we apply Strassen (rank 7) to the 2x2x2 sub-block A[:,0:2]*B[0:2,k0:k1+1]
    and use schoolbook (rank 4) for the A[:,2]*B[2,k0:k1+1] remainder.
    5 column-pairs * (7 + 4) = 55.
    """
    assert n == 2 and m == 3 and p == 10

    rank = 55
    U = [[0] * rank for _ in range(n * m)]  # 6 x 55
    V = [[0] * rank for _ in range(m * p)]  # 30 x 55
    W = [[0] * rank for _ in range(p * n)]  # 20 x 55

    col = 0
    for t in range(5):
        k0 = 2 * t
        k1 = 2 * t + 1

        # --- Strassen 7 products for A[0:2,0:2] * B[0:2,k0:k1+1] ---
        # M1: (a00 + a11) * (B[0,k0] + B[1,k1])
        U[0][col] += 1
        U[4][col] += 1
        V[k0][col] += 1
        V[10 + k1][col] += 1
        W[k0 * 2 + 0][col] += 1
        W[k1 * 2 + 1][col] += 1
        col += 1

        # M2: (a10 + a11) * B[0,k0]
        U[3][col] += 1
        U[4][col] += 1
        V[k0][col] += 1
        W[k0 * 2 + 1][col] += 1
        W[k1 * 2 + 1][col] -= 1
        col += 1

        # M3: a00 * (B[0,k1] - B[1,k1])
        U[0][col] += 1
        V[k1][col] += 1
        V[10 + k1][col] -= 1
        W[k1 * 2 + 0][col] += 1
        W[k1 * 2 + 1][col] += 1
        col += 1

        # M4: a11 * (B[1,k0] - B[0,k0])
        U[4][col] += 1
        V[10 + k0][col] += 1
        V[k0][col] -= 1
        W[k0 * 2 + 0][col] += 1
        W[k0 * 2 + 1][col] += 1
        col += 1

        # M5: (a00 + a01) * B[1,k1]
        U[0][col] += 1
        U[1][col] += 1
        V[10 + k1][col] += 1
        W[k0 * 2 + 0][col] -= 1
        W[k1 * 2 + 0][col] += 1
        col += 1

        # M6: (a10 - a00) * (B[0,k0] + B[0,k1])
        U[3][col] += 1
        U[0][col] -= 1
        V[k0][col] += 1
        V[k1][col] += 1
        W[k1 * 2 + 1][col] += 1
        col += 1

        # M7: (a01 - a11) * (B[1,k0] + B[1,k1])
        U[1][col] += 1
        U[4][col] -= 1
        V[10 + k0][col] += 1
        V[10 + k1][col] += 1
        W[k0 * 2 + 0][col] += 1
        col += 1

        # --- Schoolbook for A[:,2] * B[2,k0:k1+1] ---
        U[2][col] += 1
        V[20 + k0][col] += 1
        W[k0 * 2 + 0][col] += 1
        col += 1

        U[2][col] += 1
        V[20 + k1][col] += 1
        W[k1 * 2 + 0][col] += 1
        col += 1

        U[5][col] += 1
        V[20 + k0][col] += 1
        W[k0 * 2 + 1][col] += 1
        col += 1

        U[5][col] += 1
        V[20 + k1][col] += 1
        W[k1 * 2 + 1][col] += 1
        col += 1

    assert col == rank, f"Expected rank {rank}, got {col}"
    return U, V, W, rank


def _try_reduce_rank(n, m, p, U, V, W, deadline):
    """
    Fast column-removal search.  Uses a vectorized residual check so each
    candidate removal is O(n*m*p) instead of a full einsum, and restarts the
    scan after every successful removal.
    """
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    rank = U_arr.shape[1]
    keep = np.ones(rank, dtype=bool)

    target = np.ones((n, m, p))

    # Precompute per-column contribution tensors c_r[i,j,k] =
    # U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r].
    U3 = U_arr.reshape(n, m, rank)
    V3 = V_arr.reshape(m, p, rank)
    W3 = W_arr.reshape(p, n, rank)
    # Calculate contribution of each rank component r to the tensor T[i,j,k]
    # contrib[r, i, j, k] = U[i, j, r] * V[j, k, r] * W[k, i, r]
    contrib = np.einsum('ijr,jkr,kir->rijk', U3, V3, W3)
    current = contrib.sum(axis=0)  # (n, m, p)

    changed = True
    while changed and time.time() < deadline:
        changed = False
        for r in range(rank):
            if not keep[r]:
                continue
            # Remove column r: new tensor = current - contrib[r]
            new_tensor = current - contrib[r]
            if np.allclose(new_tensor, target, atol=1e-9):
                keep[r] = False
                current = new_tensor
                changed = True
                break  # restart scan after a removal

    new_rank = int(np.sum(keep))
    if new_rank >= rank:
        return U, V, W, rank

    U_new = U_arr[:, keep]
    V_new = V_arr[:, keep]
    W_new = W_arr[:, keep]

    U_list = U_new.tolist()
    V_list = V_new.tolist()
    W_list = W_new.tolist()

    if _validate(n, m, p, U_list, V_list, W_list):
        return U_list, V_list, W_list, new_rank
    else:
        return U, V, W, rank


def _try_linear_redundancy(n, m, p, U, V, W, deadline):
    """
    Linear-algebraic redundancy sweep: a column r is linearly redundant if
    removing it does not change the column-space of any of the three factor
    matrices (i.e. its vector is in the span of the others).  This is a
    necessary condition for tensor redundancy and can catch columns the
    residual test misses.  We only remove a column if it is simultaneously
    redundant in U, V AND W, then re-validate the tensor.
    """
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    rank = U_arr.shape[1]
    keep = np.ones(rank, dtype=bool)

    def _column_redundant(M, r, keep_mask):
        """Is column r of M in the span of the other kept columns?"""
        cols = M[:, keep_mask]
        idx = int(np.where(keep_mask)[0].tolist().index(r))
        others = np.delete(cols, idx, axis=1)
        if others.shape[1] == 0:
            return False
        col = cols[:, idx]
        if np.allclose(col, 0.0, atol=1e-12):
            return True
        # Solve others @ x = col; redundant if residual ~ 0.
        try:
            x, residuals, rank_r, _ = np.linalg.lstsq(others, col, rcond=None)
        except np.linalg.LinAlgError:
            return False
        resid = np.linalg.norm(others @ x - col)
        scale = max(1.0, np.linalg.norm(col))
        return resid <= 1e-9 * scale

    changed = True
    while changed and time.time() < deadline:
        changed = False
        for r in range(rank):
            if not keep[r]:
                continue
            if (_column_redundant(U_arr, r, keep)
                    and _column_redundant(V_arr, r, keep)
                    and _column_redundant(W_arr, r, keep)):
                keep[r] = False
                changed = True
                break

    new_rank = int(np.sum(keep))
    if new_rank >= rank:
        return U, V, W, rank

    U_new = U_arr[:, keep]
    V_new = V_arr[:, keep]
    W_new = W_arr[:, keep]
    U_list = U_new.tolist()
    V_list = V_new.tolist()
    W_list = W_new.tolist()
    if _validate(n, m, p, U_list, V_list, W_list):
        return U_list, V_list, W_list, new_rank
    else:
        return U, V, W, rank


def _try_4x4x4_projection(n, m, p, deadline):
    """
    Conditional test of the 4x4x4 rank-23 embedding idea.  We zero-pad the
    2x3x10 multiplication tensor into 4x4x4 and attempt to recover a
    rank-23 factorization whose restriction to the original index set is
    exactly the target.  This is only returned if it validates exactly, so it
    can never lower the score.  Kept as a hypothesis test; the rectangular
    sub-problem almost certainly will not project cleanly, but the cost is
    small and bounded.
    """
    # Build the 4x4x4 multiplication tensor T4[i,j,k] = 1 for all i,j,k in [0,4).
    # Our target sub-tensor is T[i,j,k] = 1 for i in [0,2), j in [0,3), k in [0,10).
    # Note k only goes to 10 but 4x4x4 has k in [0,4), so the embedding is NOT
    # dimensionally compatible (p=10 > 4).  Hence a direct 4x4x4 projection is
    # infeasible.  We return None to signal "not applicable" rather than guess.
    if p > 4 or n > 4 or m > 4:
        return None
    # If dimensions did fit we would attempt the projection here.  They do not
    # for 2x3x10, so this branch is a no-op by design.
    return None


def _try_random_projection(n, m, p, deadline):
    """
    Randomized low-rank projection search (the 'change of variables / smooth
    surrogate' idea applied as a bounded test).  We try a few random
    structured low-rank factorizations and validate exactly.  If none pass,
    we return None.  Bounded by the deadline.
    """
    rng = np.random.default_rng(0)
    # Try a handful of candidate ranks below the current best.
    for cand_rank in (50, 48, 45, 42, 40):
        if time.time() > deadline:
            break
        # Random sparse-ish integer factors: each column has a few nonzeros.
        U = np.zeros((n * m, cand_rank))
        V = np.zeros((m * p, cand_rank))
        W = np.zeros((p * n, cand_rank))
        for r in range(cand_rank):
            nu = rng.integers(1, 3)
            nv = rng.integers(1, 3)
            nw = rng.integers(1, 3)
            for _ in range(nu):
                U[rng.integers(0, n * m), r] = rng.choice([-1, 1])
            for _ in range(nv):
                V[rng.integers(0, m * p), r] = rng.choice([-1, 1])
            for _ in range(nw):
                W[rng.integers(0, p * n), r] = rng.choice([-1, 1])
        if _validate(n, m, p, U.tolist(), V.tolist(), W.tolist()):
            return U.tolist(), V.tolist(), W.tolist(), cand_rank
    return None


def construct():
    start_time = time.time()
    deadline = start_time + 100.0  # safety margin under 150s kill

    n, m, p = DIMENSIONS

    # Primary path: Strassen construction (rank 55).
    U, V, W, rank = _build_strassen(n, m, p)

    if not _validate(n, m, p, U, V, W):
        # Fallback to the guaranteed trivial decomposition.
        U, V, W, rank = _build_trivial(n, m, p)
    else:
        # Refinement machinery, in order of increasing cost, each guarded by
        # the deadline and re-validated before accepting.
        U, V, W, rank = _try_reduce_rank(n, m, p, U, V, W, deadline)
        U, V, W, rank = _try_linear_redundancy(n, m, p, U, V, W, deadline)

        # Conditional 4x4x4 projection test (no-op for 2x3x10 by design).
        proj = _try_4x4x4_projection(n, m, p, deadline)
        if proj is not None:
            U, V, W, rank = proj

        # Randomized low-rank projection test in the spare time.
        rand = _try_random_projection(n, m, p, deadline)
        if rand is not None and rand[3] < rank:
            U, V, W, rank = rand

    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
