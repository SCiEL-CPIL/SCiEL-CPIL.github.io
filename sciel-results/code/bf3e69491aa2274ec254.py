"""2x5x6 matrix multiplication tensor decomposition (rank-minimal search).

Strategy:
  * Build a structured block decomposition (Strassen 2x2x2 cores for the two
    2x2 middle blocks, direct rank-1 for the 2x1 block) -> candidate rank 45.
  * Refine via Gauss-Seidel alternating minimization: the tensor is multilinear,
    so with two factors fixed the constraints on the third are LINEAR.  Solve
    each free factor exactly (integer least squares), snap to integers, and
    verify EXACTLY with pure integer arithmetic.
  * Additionally, use a set-cover heuristic to identify and merge redundant
    rank-1 terms that share common index projections, attempting to reduce
    the rank below the structured candidate.
  * If the structured/refined candidate is not exact, fall back to the
    guaranteed schoolbook rank 60 (always valid).
"""
import time
import numpy as np
from collections import defaultdict

DIMENSIONS = [2, 5, 6]


def _build_strassen_candidate(n, m, p):
    """Build a rank-45 candidate using block partitioning + Strassen cores.

    Middle (m=5) split into column groups [0,1],[2,3],[4] (sizes 2,2,1).
    Output (p=6) split into column pairs [0,1],[2,3],[4,5].
    """
    mid_groups = [(0, 2), (2, 4), (4, 5)]
    out_pairs = [(0, 2), (2, 4), (4, 6)]

    # Strassen 2x2x2 core: returns (Uf, Vf, Wf) each a dict keyed by
    # (local_a_row, local_b_row, out_col) -> coefficient.
    def strassen_core():
        Uf = {}
        Vf = {}
        Wf = {}

        def u_add(r0, c0, r1, c1, s):
            Uf[(r0, c0)] = Uf.get((r0, c0), 0) + s
            Uf[(r1, c1)] = Uf.get((r1, c1), 0) + s

        def v_add(r0, c0, r1, c1, s):
            Vf[(r0, c0)] = Vf.get((r0, c0), 0) + s
            Vf[(r1, c1)] = Vf.get((r1, c1), 0) + s

        def w_add(r0, c0, r1, c1, s):
            Wf[(r0, c0)] = Wf.get((r0, c0), 0) + s
            Wf[(r1, c1)] = Wf.get((r1, c1), 0) + s

        # M1 = (A00+A11)*(B00+B11)
        u_add(0, 0, 1, 1, 1)
        v_add(0, 0, 1, 1, 1)
        w_add(0, 0, 0, 1, 1)
        w_add(1, 1, 0, 1, 1)
        # M2 = (A10+A11)*B00
        u_add(1, 0, 1, 1, 1)
        v_add(0, 0, 0, 0, 1)
        w_add(1, 0, 0, 1, 1)
        # M3 = A00*(B10-B11)
        u_add(0, 0, 0, 0, 1)
        v_add(1, 0, 1, 1, -1)
        w_add(0, 1, 0, 1, 1)
        # M4 = A11*(B01-B10)
        u_add(1, 1, 1, 1, 1)
        v_add(0, 1, 1, 0, -1)
        w_add(0, 0, 0, 1, 1)
        w_add(1, 0, 0, 1, 1)
        # M5 = (A00+A01)*B11
        u_add(0, 0, 0, 1, 1)
        v_add(1, 1, 1, 1, 1)
        w_add(0, 0, 0, 1, -1)
        w_add(0, 1, 0, 1, 1)
        # M6 = (A10-A00)*(B00+B10)
        u_add(1, 0, 0, 0, -1)
        v_add(0, 0, 1, 0, 1)
        w_add(1, 1, 0, 1, 1)
        # M7 = (A01-A11)*(B01+B11)
        u_add(0, 1, 1, 1, -1)
        v_add(0, 1, 1, 1, 1)
        w_add(0, 0, 0, 1, 1)
        return Uf, Vf, Wf

    core_U, core_V, core_W = strassen_core()

    total_rank = 0
    for ms, me in mid_groups:
        for os_, oe in out_pairs:
            if me - ms == 2:
                total_rank += 7
            else:
                total_rank += (oe - os_)  # rank 1 per output column
    # = 2*(3*7) + 1*3 = 42+3 = 45.

    U = np.zeros((n * m, total_rank))
    V = np.zeros((m * p, total_rank))
    W = np.zeros((p * n, total_rank))

    col = 0
    for ms, me in mid_groups:
        if me - ms == 2:
            for os_, oe in out_pairs:
                for (lr, lc), c in core_U.items():
                    U[(lr) * m + (ms + lc), col] += c
                for (lr, lc), c in core_V.items():
                    V[(ms + lr) * p + (os_ + lc), col] += c
                for (lr, lc), c in core_W.items():
                    W[(os_ + lc) * n + lr, col] += c
                col += 1
        else:
            # 2x1 (A rows 0,1 col ms) x 1x2 (B row ms cols os_,oe)
            for oc in range(os_, oe):
                U[0 * m + ms, col] = 1
                U[1 * m + ms, col] = 1
                V[ms * p + oc, col] = 1
                W[oc * n + 0, col] = 1
                W[oc * n + 1, col] = 1
                col += 1
    return U, V, W, col


def _build_schoolbook(n, m, p):
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    return U, V, W, rank


def _verify_exact(n, m, p, U, V, W):
    """Exact integer verification. Returns (ok, max_abs_error)."""
    U = np.asarray(U, dtype=object)
    V = np.asarray(V, dtype=object)
    W = np.asarray(W, dtype=object)
    rank = U.shape[1]
    T = np.zeros((n, m, p), dtype=object)
    for r in range(rank):
        Ur = U[:, r]
        Vr = V[:, r]
        Wr = W[:, r]
        for i in range(n):
            for j in range(m):
                a = Ur[i * m + j]
                if a == 0:
                    continue
                for k in range(p):
                    b = Vr[j * p + k]
                    if b == 0:
                        continue
                    T[i, j, k] = T[i, j, k] + a * b * Wr[k * n + i]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if T[i, j, k] != 1:
                    return False, abs(int(T[i, j, k]) - 1)
    return True, 0


def _snap_round(U, V, W):
    return (np.rint(U), np.rint(V), np.rint(W))


def _gauss_seidel(n, m, p, U, V, W, max_iters=40, tol=1e-11):
    """Alternating exact least-squares refinement exploiting multilinearity."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    rank = U.shape[1]
    prev_err = np.inf
    for it in range(max_iters):
        # --- Solve U (rows indexed by (i,j)) given V,W ---
        for i in range(n):
            for j in range(m):
                b = np.zeros(rank)
                for k in range(p):
                    b = b + V[j * p + k, :] * W[k * n + i, :]
                if np.allclose(b, 0):
                    continue
                nb2 = float(np.dot(b, b))
                if nb2 > 1e-30:
                    U[i * m + j, :] = b / nb2
        # --- Solve V (rows indexed by (j,k)) given U,W ---
        for j in range(m):
            for k in range(p):
                b = np.zeros(rank)
                for i in range(n):
                    b = b + U[i * m + j, :] * W[k * n + i, :]
                nb2 = float(np.dot(b, b))
                if nb2 > 1e-30:
                    V[j * p + k, :] = b / nb2
        # --- Solve W (rows indexed by (k,i)) given U,V ---
        for k in range(p):
            for i in range(n):
                b = np.zeros(rank)
                for j in range(m):
                    b = b + U[i * m + j, :] * V[j * p + k, :]
                nb2 = float(np.dot(b, b))
                if nb2 > 1e-30:
                    W[k * n + i, :] = b / nb2
        # --- Evaluate error (float) ---
        err = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for r in range(rank):
                        s += U[i * m + j, r] * V[j * p + k, r] * W[k * n + i, r]
                    err += (s - 1.0) ** 2
        if abs(prev_err - err) < tol and err < 1e-6:
            break
        prev_err = err
    return U, V, W


def _set_cover_merge(n, m, p, U, V, W):
    """
    Heuristic set-cover approach to merge redundant rank-1 terms.
    
    Model the 60 schoolbook terms as edges in a tripartite graph.
    Use a greedy heuristic to identify terms that can be merged
    by sharing common index projections (i, j, or k).
    
    This is an experimental approach; if it doesn't improve rank,
    we return the original matrices.
    """
    U_orig = np.array(U, dtype=float)
    V_orig = np.array(V, dtype=float)
    W_orig = np.array(W, dtype=float)
    rank_orig = U_orig.shape[1]
    
    # Try to find a lower rank decomposition by identifying redundant terms
    # We'll use a simple heuristic: try to remove terms one by one and check
    # if the decomposition still holds (approximately)
    
    best_rank = rank_orig
    best_U, best_V, best_W = U_orig.copy(), V_orig.copy(), W_orig.copy()
    
    # For each term, try to see if it can be removed
    # This is O(rank^2 * n*m*p) which is manageable for small ranks
    for r_remove in range(rank_orig):
        # Skip this term
        U_test = np.delete(U_orig, r_remove, axis=1)
        V_test = np.delete(V_orig, r_remove, axis=1)
        W_test = np.delete(W_orig, r_remove, axis=1)
        
        # Check if the decomposition still holds (with some tolerance)
        err = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for r in range(U_test.shape[1]):
                        s += U_test[i * m + j, r] * V_test[j * p + k, r] * W_test[k * n + i, r]
                    err += (s - 1.0) ** 2
        
        if err < 1e-8:
            # This term is redundant
            best_rank = U_test.shape[1]
            best_U, best_V, best_W = U_test, V_test, W_test
            # Restart the search with the new reduced matrices
            U_orig, V_orig, W_orig = U_test, V_test, W_test
            rank_orig = U_test.shape[1]
            break  # Restart with the new reduced set
    
    if best_rank < rank_orig:
        return best_U, best_V, best_W, best_rank
    
    return U_orig, V_orig, W_orig, rank_orig


def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # safety margin under 120s

    n, m, p = DIMENSIONS

    # Fallback: guaranteed valid schoolbook rank 60.
    fb_U, fb_V, fb_W, fb_rank = _build_schoolbook(n, m, p)

    # Try the structured rank-45 Strassen candidate.
    try:
        U, V, W, rank = _build_strassen_candidate(n, m, p)
        # Refine with alternating exact least-squares.
        U, V, W = _gauss_seidel(n, m, p, U, V, W)
        U, V, W = _snap_round(U, V, W)
        ok, _ = _verify_exact(n, m, p, U, V, W)
        if ok and rank < fb_rank:
            # Try to further reduce rank using set-cover heuristic
            if time.time() < deadline - 10.0:
                U2, V2, W2, rank2 = _set_cover_merge(n, m, p, U, V, W)
                if rank2 < rank:
                    U2, V2, W2 = _snap_round(U2, V2, W2)
                    ok2, _ = _verify_exact(n, m, p, U2, V2, W2)
                    if ok2:
                        return {"U": U2.tolist(), "V": V2.tolist(), "W": W2.tolist()}
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    except Exception:
        pass

    if time.time() > deadline:
        return {"U": fb_U.tolist(), "V": fb_V.tolist(), "W": fb_W.tolist()}

    # Fallback (always valid).
    return {"U": fb_U.tolist(), "V": fb_V.tolist(), "W": fb_W.tolist()}


# EVOLVE-BLOCK-START
# Primary entry point: construct() -> {'U','V','W'} (shapes (n*m,rank),(m*p,rank),(p*n,rank)).
# Score = rank (lower is better).  Structured Strassen block decomposition with
# alternating exact least-squares refinement + exact integer verification;
# set-cover heuristic for further rank reduction; schoolbook rank 60 as guaranteed fallback.
# EVOLVE-BLOCK-END
