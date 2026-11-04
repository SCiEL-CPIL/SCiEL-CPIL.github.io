"""
Matrix multiplication 2x4x7 via tensor decomposition.
Attempts a search for a decomposition with rank < 50, falling back to
the known rank-50 block construction if the search does not succeed
within the time budget.

Score = seed / rank.  seed = 56 (schoolbook).  Rank 50 -> 1.12, rank 49 -> 1.1429, ...
"""

import time
import random
import numpy as np

# ---------------------------------------------------------------------------
# Deadline guard
# ---------------------------------------------------------------------------
_EVAL_DEADLINE = 150.0          # wall-clock seconds (hard limit)
_START = time.time()
TIME_BUDGET = _EVAL_DEADLINE - (time.time() - _START) - 5.0   # leave 5 s margin

def _time_left():
    return _EVAL_DEADLINE - 5.0 - time.time()

# ---------------------------------------------------------------------------
# Build the multiplication tensor T in [r,i,j] layout
#   r = k*n + i  (n*p values),  i is the row of A,  j = row of B / col of B
# ---------------------------------------------------------------------------
def _build_tensor(n, m, p):
    r_dim = p * n
    i_dim = n
    j_dim = m * p
    T = np.zeros((r_dim, i_dim, j_dim))
    for k in range(p):
        for i in range(n):
            for j in range(m):
                r = k * n + i
                jj = j * p + k
                T[r, i, jj] = 1.0
    return T

# ---------------------------------------------------------------------------
# Known rank-50 construction (Strassen blocks + direct rank-1)
# ---------------------------------------------------------------------------
def _construct_rank50(n=2, m=4, p=7):
    rank = 50
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]

    col = [0]

    def add_strassen(i0, j0, k0):
        c = col[0]
        # M1
        U[i0*m + j0][c] += 1
        U[(i0+1)*m + (j0+1)][c] += 1
        V[j0*p + k0][c] += 1
        V[(j0+1)*p + (k0+1)][c] += 1
        W[k0*n + i0][c] += 1
        W[(k0+1)*n + (i0+1)][c] += 1
        c += 1
        # M2
        U[(i0+1)*m + j0][c] += 1
        U[(i0+1)*m + (j0+1)][c] += 1
        V[j0*p + k0][c] += 1
        W[k0*n + (i0+1)][c] += 1
        W[(k0+1)*n + (i0+1)][c] -= 1
        c += 1
        # M3
        U[i0*m + j0][c] += 1
        V[j0*p + (k0+1)][c] += 1
        V[(j0+1)*p + (k0+1)][c] -= 1
        W[(k0+1)*n + i0][c] += 1
        W[(k0+1)*n + (i0+1)][c] += 1
        c += 1
        # M4
        U[(i0+1)*m + (j0+1)][c] += 1
        V[(j0+1)*p + k0][c] += 1
        V[j0*p + k0][c] -= 1
        W[k0*n + i0][c] += 1
        W[k0*n + (i0+1)][c] += 1
        c += 1
        # M5
        U[i0*m + j0][c] += 1
        U[i0*m + (j0+1)][c] += 1
        V[(j0+1)*p + (k0+1)][c] += 1
        W[k0*n + i0][c] -= 1
        W[(k0+1)*n + i0][c] += 1
        c += 1
        # M6
        U[(i0+1)*m + j0][c] += 1
        U[i0*m + j0][c] -= 1
        V[j0*p + k0][c] += 1
        V[j0*p + (k0+1)][c] += 1
        W[(k0+1)*n + (i0+1)][c] += 1
        c += 1
        # M7
        U[i0*m + (j0+1)][c] += 1
        U[(i0+1)*m + (j0+1)][c] -= 1
        V[(j0+1)*p + k0][c] += 1
        V[(j0+1)*p + (k0+1)][c] += 1
        W[k0*n + i0][c] += 1
        c += 1
        col[0] = c

    # 2x4x4 block: four 2x2x2 Strassen blocks
    add_strassen(0, 0, 0)
    add_strassen(0, 0, 2)
    add_strassen(0, 2, 0)
    add_strassen(0, 2, 2)
    # 2x4x2 block: two 2x2x2 Strassen blocks
    add_strassen(0, 0, 4)
    add_strassen(0, 2, 4)
    # 2x4x1 block: eight rank-1 terms
    for i in range(n):
        for j in range(m):
            U[i*m + j][col[0]] = 1.0
            V[j*p + 6][col[0]] = 1.0
            W[6*n + i][col[0]] = 1.0
            col[0] += 1

    assert col[0] == rank, f"Rank mismatch: used {col[0]}, expected {rank}"
    return {"U": U, "V": V, "W": W}

# ---------------------------------------------------------------------------
# Verification: check that the factor matrices exactly reproduce the tensor
# ---------------------------------------------------------------------------
def _verify(U, V, W, n, m, p, tol=1e-10):
    r_dim = p * n
    i_dim = n
    j_dim = m * p
    T = _build_tensor(n, m, p)
    # Reshape to matrix form: T_matrix[r, (i,j)] = T[r, i, j]
    T_mat = T.reshape(r_dim, i_dim * j_dim)
    # Build candidate
    rank = len(W[0])
    Ur = np.array(U).T   # (rank, i_dim)
    Vr = np.array(V).T   # (rank, j_dim)
    Wr = np.array(W).T   # (rank, r_dim)
    # Candidate[r, i, j] = sum_t Wr[r,t] * Ur[t,i] * Vr[t,j]
    # Efficient: for each t, add outer(Wr[:,t], Ur[t,:]) * Vr[t,:]
    cand = np.zeros_like(T_mat)
    Ur_flat = Ur  # (rank, n)
    for t in range(rank):
        wt = Wr[t, :]          # (r_dim,)
        ut = Ur[t, :]          # (n,)
        vt = Vr[t, :]          # (j_dim,)
        # cand += outer(wt, ut) * vt   -> (r_dim, n * j_dim)
        # ut * vt reshaped to (n, j_dim) then flattened
        outer = ut[:, None] * vt[None, :]   # (n, j_dim)
        cand += np.outer(wt, outer.ravel())
    max_err = np.max(np.abs(cand - T_mat))
    return max_err < tol

# ---------------------------------------------------------------------------
# Search: try random integer/half-integer matrices of rank r and verify
# ---------------------------------------------------------------------------
def _random_search(T, T_mat, n, m, p, max_rank, time_budget):
    """Try random factor matrices of rank max_rank. Return (U, V, W) or None."""
    r_dim = p * n
    i_dim = n
    j_dim = m * p
    rank = max_rank

    # Allowed entries: integers and half-integers in a small range
    # For small ranks, the entries are typically small.
    # Use a set of candidate values.
    allowed_vals = []
    for v in range(-10, 11):
        allowed_vals.append(float(v))
        if v < 10:
            allowed_vals.append(v + 0.5)
            allowed_vals.append(-(v + 0.5))
    # Remove duplicates
    allowed_vals = list(set(allowed_vals))
    nv = len(allowed_vals)

    # Precompute random index arrays for speed
    # We'll use numpy random for vectorised generation

    rng = np.random.RandomState(12345)

    # Sparsity patterns: for each rank-1 term, which positions are nonzero
    # In the rank-50 solution, each term has at most ~4 nonzeros in each factor.
    # For rank 49, let's allow up to 6 nonzeros per factor per term.

    max_nz = 6
    attempts = 0

    while _time_left() > time_budget:
        attempts += 1
        # Generate random U (i_dim x rank), V (j_dim x rank), W (r_dim x rank)
        U = np.zeros((i_dim, rank))
        V = np.zeros((j_dim, rank))
        W = np.zeros((r_dim, rank))

        for t in range(rank):
            # Random positions
            nzu = rng.randint(1, max_nz + 1, size=1)[0]
            nzv = rng.randint(1, max_nz + 1, size=1)[0]
            nzw = rng.randint(1, max_nz + 1, size=1)[0]

            pos_u = rng.randint(0, i_dim, size=nzu)
            pos_v = rng.randint(0, j_dim, size=nzv)
            pos_w = rng.randint(0, r_dim, size=nzw)

            vals_u = rng.randint(0, nv, size=nzu)
            vals_v = rng.randint(0, nv, size=nzv)
            vals_w = rng.randint(0, nv, size=nzw)

            U[pos_u, t] = np.array(allowed_vals)[vals_u]
            V[pos_v, t] = np.array(allowed_vals)[vals_v]
            W[pos_w, t] = np.array(allowed_vals)[vals_w]

        # Check
        Ur = U.T   # (rank, i_dim)
        Vr = V.T   # (rank, j_dim)
        Wr = W.T   # (rank, r_dim)

        # Build candidate efficiently
        cand = np.zeros_like(T_mat)
        for t in range(rank):
            wt = Wr[t, :]
            ut = Ur[t, :]
            vt = Vr[t, :]
            outer = ut[:, None] * vt[None, :]
            cand += np.outer(wt, outer.ravel())

        if np.max(np.abs(cand - T_mat)) < 1e-10:
            # Found! Convert to lists
            U_list = [[0.0] * rank for _ in range(i_dim)]
            V_list = [[0.0] * rank for _ in range(j_dim)]
            W_list = [[0.0] * rank for _ in range(r_dim)]
            for i in range(i_dim):
                for t in range(rank):
                    U_list[i][t] = U[i, t]
            for j in range(j_dim):
                for t in range(rank):
                    V_list[j][t] = V[j, t]
            for r in range(r_dim):
                for t in range(rank):
                    W_list[r][t] = W[r, t]
            return {"U": U_list, "V": V_list, "W": W_list}

    return None

# ---------------------------------------------------------------------------
# Greedy search: build rank-1 terms one at a time
# ---------------------------------------------------------------------------
def _greedy_search(T, T_mat, n, m, p, target_rank, time_budget):
    """Greedy: build rank-1 terms one at a time, trying to reach target_rank."""
    r_dim = p * n
    i_dim = n
    j_dim = m * p

    residual = T_mat.copy()
    U_rows = []
    V_rows = []
    W_rows = []

    # Candidate vectors: all possible small integer/half-integer vectors
    # with at most 3 nonzeros (for efficiency)
    def gen_candidates(dim, max_nz=3, max_val=4):
        cands = []
        vals = [float(v) for v in range(-max_val, max_val+1)] + \
               [float(v+0.5) for v in range(-max_val, max_val)] + \
               [float(-(v+0.5)) for v in range(-max_val, max_val)]
        vals = list(set(vals))
        nv = len(vals)
        for nz in range(1, max_nz+1):
            for pos in range(1, dim+1):
                # combinations of positions
                pass
        # Simpler: random generation in the loop instead
        return None

    # Use random candidate vectors with sparsity
    rng = np.random.RandomState(54321)
    allowed_vals = [float(v) for v in range(-8, 9)] + \
                   [float(v+0.5) for v in range(-8, 8)] + \
                   [float(-(v+0.5)) for v in range(-8, 8)]
    allowed_vals = list(set(allowed_vals))
    nv = len(allowed_vals)
    av = np.array(allowed_vals)

    max_nz = 5  # max nonzeros per factor per term

    while len(W_rows) < target_rank:
        # Try to find a good rank-1 term for the residual
        best_err = np.max(np.abs(residual))
        best_u = best_v = best_w = None

        for trial in range(2000):
            # Random u, v, w
            nzu = rng.randint(1, max_nz+1)
            nzv = rng.randint(1, max_nz+1)
            nzw = rng.randint(1, max_nz+1)
            pos_u = rng.randint(0, i_dim, size=nzu)
            pos_v = rng.randint(0, j_dim, size=nzv)
            pos_w = rng.randint(0, r_dim, size=nzw)
            u = np.zeros(i_dim); u[pos_u] = av[rng.randint(0, nv, size=nzu)]
            v = np.zeros(j_dim); v[pos_v] = av[rng.randint(0, nv, size=nzv)]
            w = np.zeros(r_dim); w[pos_w] = av[rng.randint(0, nv, size=nzw)]

            # Compute the rank-1 term as a matrix
            # term[r, (i,j)] = w[r] * u[i] * v[j]
            term_mat = np.outer(w, np.outer(u, v).flatten())

            new_residual = residual - term_mat
            err = np.max(np.abs(new_residual))

            if err < best_err:
                best_err = err
                best_u, best_v, best_w = u.copy(), v.copy(), w.copy()
                if err < 1e-12:
                    break

        if best_u is None:
            break

        # Accept the term
        residual -= np.outer(best_w, np.outer(best_u, best_v).flatten())
        W_rows.append(best_w.tolist())
        U_rows.append(best_u.tolist())
        V_rows.append(best_v.tolist())

        if _time_left() < 5:
            break

    if len(W_rows) <= target_rank:
        # Verify full decomposition
        rank = len(W_rows)
        U_list = [[0.0] * rank for _ in range(i_dim)]
        V_list = [[0.0] * rank for _ in range(j_dim)]
        W_list = [[0.0] * rank for _ in range(r_dim)]
        for i in range(i_dim):
            for t in range(rank):
                U_list[i][t] = U_rows[t][i]
        for j in range(j_dim):
            for t in range(rank):
                V_list[j][t] = V_rows[t][j]
        for r in range(r_dim):
            for t in range(rank):
                W_list[r][t] = W_rows[t][r]

        # Check all entries are integer or half-integer
        def check_half(x):
            r = 2 * x
            return abs(r - round(r)) < 1e-9

        ok = True
        for row in U_list + V_list + W_list:
            for x in row:
                if abs(x) > 1e-9 and not check_half(x):
                    ok = False
                    break
            if not ok:
                break

        if ok:
            return {"U": U_list, "V": V_list, "W": W_list}

    return None

# ---------------------------------------------------------------------------
# Main construct
# ---------------------------------------------------------------------------
def construct():
    n, m, p = 2, 4, 7

    # Build the tensor for verification
    T = _build_tensor(n, m, p)
    r_dim = p * n
    T_mat = T.reshape(r_dim, n * (m * p))

    # Known rank-50 solution as fallback
    fallback = _construct_rank50(n, m, p)

    # Try greedy search for rank 49
    time_budget = min(TIME_BUDGET, 60.0)
    result = _greedy_search(T, T_mat, n, m, p, 49, time_budget)
    if result is not None:
        if _verify(result["U"], result["V"], result["W"], n, m, p):
            return result

    # Try greedy search for rank 48
    if _time_left() > 30:
        result = _greedy_search(T, T_mat, n, m, p, 48, min(_time_left(), 40.0))
        if result is not None:
            if _verify(result["U"], result["V"], result["W"], n, m, p):
                return result

    # Try random search for rank 49
    if _time_left() > 20:
        result = _random_search(T, T_mat, n, m, p, 49, min(_time_left(), 25.0))
        if result is not None:
            return result

    # Fallback to rank 50
    return fallback
