"""Combined program: block-structured Strassen search with ALS refinement."""
import time
import numpy as np
from itertools import product

DIMENSIONS = [2, 4, 5]

# EVOLVE-BLOCK-START

def _schoolbook_decomposition(n, m, p):
    """Standard O(nmp) decomposition."""
    rank = n * m * p
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1.0
                V[j * p + k, col] = 1.0
                W[k * n + i, col] = 1.0
                col += 1
    return U, V, W


def _check_decomposition(U, V, W, n, m, p, tol=1e-9):
    """Check if the tensor decomposition is exact."""
    nm = n * m
    mp = m * p
    np_ = n * p
    rank = U.shape[1]
    U = np.asarray(U, dtype=float)
    V = np.asarray(V, dtype=float)
    W = np.asarray(W, dtype=float)

    # Check the 40 "one" positions
    for i in range(n):
        for j in range(m):
            for k in range(p):
                idx_a = i * m + j
                idx_b = j * p + k
                idx_c = k * n + i
                val = np.sum(U[idx_a, :] * V[idx_b, :] * W[idx_c, :])
                if abs(val - 1.0) > tol:
                    return False

    # Check that all other positions are zero
    # Build set of "one" positions
    one_positions = set()
    for i in range(n):
        for j in range(m):
            for k in range(p):
                one_positions.add((i * m + j, j * p + k, k * n + i))

    for a in range(nm):
        for b in range(mp):
            for c in range(np_):
                if (a, b, c) in one_positions:
                    continue
                val = np.sum(U[a, :] * V[b, :] * W[c, :])
                if abs(val) > tol:
                    return False
    return True


def _check_half_integer(U, V, W, tol=1e-9):
    """Check if entries are half-integers."""
    for mat in [U, V, W]:
        mat = np.asarray(mat, dtype=float)
        if not np.allclose(mat * 2, np.round(mat * 2), atol=tol):
            return False
    return True


def _snap_to_half_integer(U, V, W):
    """Snap entries to nearest half-integer."""
    U_s = np.round(np.asarray(U, dtype=float) * 2) / 2.0
    V_s = np.round(np.asarray(V, dtype=float) * 2) / 2.0
    W_s = np.round(np.asarray(W, dtype=float) * 2) / 2.0
    return U_s, V_s, W_s


def _build_target_tensor(n, m, p):
    """Build the target matmul tensor T[a,b,c]."""
    nm = n * m
    mp = m * p
    np_ = n * p
    T = np.zeros((nm, mp, np_), dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j, j * p + k, k * n + i] = 1.0
    return T


def _try_rank_als(T, r, n, m, p, max_restarts=30, max_iters=3000, time_budget=None, start_time=None):
    """Try to find a rank-r decomposition using ALS with random restarts."""
    if start_time is None:
        start_time = time.time()
    nm = n * m
    mp = m * p
    np_ = n * p

    best_result = None
    best_err = np.inf

    for restart in range(max_restarts):
        if time_budget is not None and time.time() > time_budget:
            break

        # Random initialization
        U = np.random.randn(nm, r) * 0.5
        V = np.random.randn(mp, r) * 0.5
        W = np.random.randn(np_, r) * 0.5

        prev_err = np.inf
        for it in range(max_iters):
            if time_budget is not None and time.time() > time_budget:
                break

            # Coordinate descent: update one column at a time
            for idx in range(r):
                # Compute residual excluding current term
                T_res = T.copy()
                for rr in range(r):
                    if rr == idx:
                        continue
                    T_res -= np.einsum('a,b,c->abc', U[:, rr], V[:, rr], W[:, rr])

                # Find best rank-1 approximation of T_res
                # Flatten to (nm, mp*np_) and SVD
                Tf = T_res.reshape(nm, mp * np_)
                try:
                    U_svd, S, Vt_svd = np.linalg.svd(Tf, full_matrices=False)
                except np.linalg.LinAlgError:
                    continue
                if S[0] < 1e-15:
                    continue
                u_vec = U_svd[:, 0] * np.sqrt(S[0])
                v_full = Vt_svd[0, :] * np.sqrt(S[0])
                V_mat = v_full.reshape(mp, np_)
                try:
                    V2_svd, S2, Wt_svd = np.linalg.svd(V_mat, full_matrices=False)
                except np.linalg.LinAlgError:
                    continue
                if S2[0] < 1e-15:
                    continue
                v_vec = V2_svd[:, 0] * np.sqrt(S2[0])
                w_vec = Wt_svd[0, :] * np.sqrt(S2[0])

                U[:, idx] = u_vec
                V[:, idx] = v_vec
                W[:, idx] = w_vec

            # Compute error
            T_recon = np.zeros_like(T)
            for rr in range(r):
                T_recon += np.einsum('a,b,c->abc', U[:, rr], V[:, rr], W[:, rr])
            err = np.linalg.norm(T - T_recon)

            if err < 1e-10:
                best_result = (U.copy(), V.copy(), W.copy())
                return best_result
            if err > prev_err * (1 + 1e-6):
                break
            prev_err = err

        # Check if this restart gave the best so far
        T_recon = np.zeros_like(T)
        for rr in range(r):
            T_recon += np.einsum('a,b,c->abc', U[:, rr], V[:, rr], W[:, rr])
        err = np.linalg.norm(T - T_recon)
        if err < best_err:
            best_err = err
            best_result = (U.copy(), V.copy(), W.copy())

    return best_result


def _try_strassen_block(n, m, p, time_budget=None, start_time=None):
    """
    Try to exploit block structure: split the 4-dim inner product into two 2-dim blocks.
    For 2x4 * 4x5: A is 2x4, B is 4x5.
    Split A into [A1 | A2] where A1 is 2x2, A2 is 2x2.
    Split B into [B1; B2] where B1 is 2x5, B2 is 2x5.
    C = A1*B1 + A2*B2.
    
    Each 2x2 * 2x5 can be done with rank at most 2*2*5=20 schoolbook,
    but using Strassen-like: 2x2*2x2 is rank 3, so 2x2*2x5 needs... 
    Actually, let's think differently.
    
    The key insight: if we can decompose the 4-dim space into two independent 2-dim subspaces,
    we get two independent 2x2*2x5 problems. But 2x2*2x5 isn't standard Strassen.
    
    Alternative: use the fact that for 2x2*2x2, Strassen gives rank 3.
    If we can reduce 2x4*4x5 to a combination of 2x2*2x2 blocks...
    
    Actually, let's try a more direct approach: use known low-rank constructions.
    For 2x2*2x2, rank 3 (Strassen).
    For 2x2*2x3, we can use rank 6 (two Strassen blocks sharing the 2x2 input).
    
    Let's just try the ALS approach with various ranks and see what works.
    """
    if start_time is None:
        start_time = time.time()
    T = _build_target_tensor(n, m, p)
    nm = n * m
    mp = m * p
    np_ = n * p

    # Try ranks from a reasonable lower bound
    # Lower bound: max(n*m, m*p, n*p) / something
    # For 2x4*4x5: nm=8, mp=20, np_=10
    # Trivial lower bound: max(nm, mp, np_) = 20? No, that's not right.
    # The rank must be at least max(n*m, m*p, n*p) / max(n, m, p) ... not quite.
    # A better lower bound: rank >= max(n*m, m*p, n*p) / max(n, m, p) is not standard.
    # Let's just try from rank 12 up to 40.
    
    best_result = None
    for r in range(12, 41):
        if time_budget is not None and time.time() > time_budget - 5:
            break
        
        result = _try_rank_als(T, r, n, m, p, max_restarts=15, max_iters=2000,
                               time_budget=time_budget, start_time=start_time)
        if result is not None:
            U, V, W = result
            # Snap to half-integer and verify
            U_s, V_s, W_s = _snap_to_half_integer(U, V, W)
            if _check_decomposition(U_s, V_s, W_s, n, m, p):
                best_result = (U_s, V_s, W_s)
                break
            # Also check without snapping
            if _check_decomposition(U, V, W, n, m, p):
                if _check_half_integer(U, V, W):
                    best_result = (U, V, W)
                    break
                # If not half-integer, try to find a half-integer solution
                # by checking if rounding works
                if _check_decomposition(U_s, V_s, W_s, n, m, p):
                    best_result = (U_s, V_s, W_s)
                    break

    return best_result


def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin before 120s limit

    # Start with schoolbook as fallback
    U_sb, V_sb, W_sb = _schoolbook_decomposition(n, m, p)
    best_U, best_V, best_W = U_sb.copy(), V_sb.copy(), W_sb.copy()
    best_rank = n * m * p

    # Try to find lower rank decompositions
    result = _try_strassen_block(n, m, p, time_budget=deadline, start_time=start_time)
    if result is not None:
        U, V, W = result
        best_U, best_V, best_W = U, V, W
        best_rank = U.shape[1]

    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}

# EVOLVE-BLOCK-END
