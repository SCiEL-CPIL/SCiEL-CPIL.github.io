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
    # 2x4x1 block: try to use fewer terms by combining
    # For the last column (k=6), we have 2*4=8 entries in W, 4 entries in U, 1 entry in V
    # Can we do better than 8 rank-1 terms?
    # Each term contributes to at most one W entry (since V has only 1 nonzero per term for k=6)
    # So we need at least 8 terms... unless we use V entries for other k values
    # But that would mess up the other blocks.
    # Let's try a different approach: use 2x2 Strassen for the last 2x4x2 block
    # and handle the last 2x4x1 separately with a clever construction
    
    # Actually, let's try: use Strassen for 2x4x4 (4 blocks * 7 = 28),
    # then for 2x4x2, use 2x2x2 Strassen twice (2*7=14), total 42
    # Then for 2x4x1, we need 8 terms, total 50. That's what we have.
    
    # Alternative: can we share terms between the 2x4x2 and 2x4x1 parts?
    # The 2x4x2 part uses k=4,5 and the 2x4x1 part uses k=6.
    # If we use a term that has V entries for k=4,5,6, it might help.
    
    # For now, keep the 8 terms but see if we can reduce by 1
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
# Structured search: try to find rank-49 by modifying rank-50 base
# ---------------------------------------------------------------------------
def _structured_search_rank49(n=2, m=4, p=7, time_budget=30.0):
    """
    Try to find a rank-49 decomposition by:
    1. Starting from rank-50 and trying to merge/eliminate one term
    2. Using structured perturbations
    """
    r_dim = p * n
    i_dim = n
    j_dim = m * p
    
    # Get the rank-50 base
    base = _construct_rank50(n, m, p)
    U_base = np.array(base["U"])  # (i_dim, 50)
    V_base = np.array(base["V"])  # (j_dim, 50)
    W_base = np.array(base["W"])  # (r_dim, 50)
    
    T = _build_tensor(n, m, p)
    T_mat = T.reshape(r_dim, i_dim * j_dim)
    
    # Try eliminating each term and seeing if we can redistribute
    # For rank-49, we need to combine 50 terms into 49
    # Strategy: for each pair of terms, try to merge them into one
    
    allowed_vals = set()
    for v in range(-8, 9):
        allowed_vals.add(float(v))
        if v < 8:
            allowed_vals.add(v + 0.5)
            allowed_vals.add(-(v + 0.5))
    allowed_vals = list(allowed_vals)
    av = np.array(allowed_vals)
    
    rng = np.random.RandomState(42)
    
    # Try merging pairs of terms
    for t1 in range(50):
        if _time_left() < 5:
            break
        for t2 in range(t1 + 1, 50):
            if _time_left() < 5:
                break
            
            # Try to find a single term that replaces terms t1 and t2
            # The contribution of terms t1 and t2 to the tensor is:
            # sum over (r,i,j): W[r,t1]*U[i,t1]*V[j,t2] + W[r,t2]*U[i,t2]*V[j,t2]
            # We want to find u_new, v_new, w_new such that this equals
            # W_new[r]*U_new[i]*V_new[j] for all r,i,j
            
            # This is generally not possible, but we can try random searches
            # for a term that matches the combined contribution
            
            u1 = U_base[:, t1]
            v1 = V_base[:, t1]
            w1 = W_base[:, t1]
            u2 = U_base[:, t2]
            v2 = V_base[:, t2]
            w2 = W_base[:, t2]
            
            # Combined contribution matrix
            combined = np.outer(w1, np.outer(u1, v1).flatten()) + \
                       np.outer(w2, np.outer(u2, v2).flatten())
            
            # Try to find a rank-1 approximation
            # Use SVD to get the best rank-1 approximation
            U_svd, s, Vt_svd = np.linalg.svd(combined, full_matrices=False)
            u_new = U_svd[:, 0] * np.sqrt(s[0])
            v_new_flat = Vt_svd[0, :] * np.sqrt(s[0])
            v_new = v_new_flat.reshape(i_dim, j_dim).flatten() if False else v_new_flat
            # Wait, v_new should be of length j_dim, not i_dim*j_dim
            # Let me fix: v_new_flat is already (i_dim*j_dim,) but we need (j_dim,)
            # Actually no: np.outer(u1, v1) is (i_dim, j_dim), so .flatten() is (i_dim*j_dim,)
            # So v_new should be (j_dim,), but we have (i_dim*j_dim,) from SVD
            # This approach doesn't directly work. Let me try a different approach.
            
            # Alternative: try random rank-1 terms and check if they match
            for attempt in range(100):
                if _time_left() < 5:
                    break
                
                nzu = rng.randint(1, 4)
                nzv = rng.randint(1, 6)
                nzw = rng.randint(1, 8)
                
                pos_u = rng.randint(0, i_dim, size=nzu)
                pos_v = rng.randint(0, j_dim, size=nzv)
                pos_w = rng.randint(0, r_dim, size=nzw)
                
                u_new = np.zeros(i_dim)
                v_new = np.zeros(j_dim)
                w_new = np.zeros(r_dim)
                
                u_new[pos_u] = av[rng.randint(0, len(av), size=nzu)]
                v_new[pos_v] = av[rng.randint(0, len(av), size=nzv)]
                w_new[pos_w] = av[rng.randint(0, len(av), size=nzw)]
                
                # Check if this matches the combined contribution
                candidate = np.outer(w_new, np.outer(u_new, v_new).flatten())
                if np.max(np.abs(candidate - combined)) < 1e-10:
                    # Found a merge! Create the rank-49 solution
                    U_new = np.delete(U_base, [t1, t2], axis=1)
                    V_new = np.delete(V_base, [t1, t2], axis=1)
                    W_new = np.delete(W_base, [t1, t2], axis=1)
                    
                    U_new = np.column_stack([U_new, u_new])
                    V_new = np.column_stack([V_new, v_new])
                    W_new = np.column_stack([W_new, w_new])
                    
                    # Verify
                    cand = np.zeros_like(T_mat)
                    for t in range(49):
                        wt = W_new[:, t]
                        ut = U_new[:, t]
                        vt = V_new[:, t]
                        cand += np.outer(wt, np.outer(ut, vt).flatten())
                    
                    if np.max(np.abs(cand - T_mat)) < 1e-10:
                        U_list = [[0.0] * 49 for _ in range(i_dim)]
                        V_list = [[0.0] * 49 for _ in range(j_dim)]
                        W_list = [[0.0] * 49 for _ in range(r_dim)]
                        for i in range(i_dim):
                            for t in range(49):
                                U_list[i][t] = U_new[i, t]
                        for j in range(j_dim):
                            for t in range(49):
                                V_list[j][t] = V_new[j, t]
                        for r in range(r_dim):
                            for t in range(49):
                                W_list[r][t] = W_new[r, t]
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

    # Try structured search for rank 49 (most promising)
    if _time_left() > 10:
        result = _structured_search_rank49(n, m, p, min(_time_left(), 40.0))
        if result is not None:
            if _verify(result["U"], result["V"], result["W"], n, m, p):
                return result

    # Try greedy search for rank 49
    if _time_left() > 15:
        result = _greedy_search(T, T_mat, n, m, p, 49, min(_time_left(), 30.0))
        if result is not None:
            if _verify(result["U"], result["V"], result["W"], n, m, p):
                return result

    # Fallback to rank 50
    return fallback
