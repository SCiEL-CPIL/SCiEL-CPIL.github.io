"""Matrix multiplication 2,2,11 - rank-minimal decomposition.

Strategy: build several structured (Strassen-style) decompositions with
different column groupings, then aggressively reduce rank using SVD-based
matricization rank detection plus greedy exact column elimination, and
finally a direct low-rank polish. Keep the public interface construct().
"""
import time
import numpy as np

DIMENSIONS = [2, 2, 11]


def _build_target_tensor():
    """Build the multiplication tensor T of shape (n*m, m*p, p*n)."""
    n, m, p = 2, 2, 11
    T = np.zeros((n * m, m * p, p * n), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1.0
    return T


def _verify(U, V, W, T=None):
    """Verify that the decomposition exactly reproduces the multiplication tensor."""
    if T is None:
        T = _build_target_tensor()
    U = np.asarray(U, dtype=np.float64)
    V = np.asarray(V, dtype=np.float64)
    W = np.asarray(W, dtype=np.float64)
    T_decomp = np.einsum('ar,br,cr->abc', U, V, W)
    return np.allclose(T, T_decomp, atol=1e-9)


def _matricize(U, V, W):
    """Return M with columns vec(U_r ⊗ V_r ⊗ W_r)."""
    dim_a, dim_b, dim_c = U.shape[0], V.shape[0], W.shape[0]
    r = U.shape[1]
    M = np.zeros((dim_a * dim_b * dim_c, r), dtype=np.float64)
    for col in range(r):
        M[:, col] = np.einsum('a,b,c->abc', U[:, col], V[:, col], W[:, col]).flatten()
    return M


def _build_strassen_for_group(k0, k1, col, U, V, W, p, n):
    """Add a Strassen 2x2x2x2 block for output columns k0,k1 at offset col."""
    v_e = k0
    v_f = k1
    v_g = p + k0
    v_h = p + k1
    w_c00 = k0 * n + 0
    w_c01 = k1 * n + 0
    w_c10 = k0 * n + 1
    w_c11 = k1 * n + 1
    u_a, u_b, u_c, u_d = 0, 1, 2, 3

    # m1 = (a+d)(e+h)
    U[u_a, col] = 1.0; U[u_d, col] = 1.0
    V[v_e, col] = 1.0; V[v_h, col] = 1.0
    W[w_c00, col] = 1.0; W[w_c11, col] = 1.0
    col += 1
    # m2 = (c+d)e
    U[u_c, col] = 1.0; U[u_d, col] = 1.0
    V[v_e, col] = 1.0
    W[w_c10, col] = 1.0; W[w_c11, col] = -1.0
    col += 1
    # m3 = a(f-h)
    U[u_a, col] = 1.0
    V[v_f, col] = 1.0; V[v_h, col] = -1.0
    W[w_c01, col] = 1.0; W[w_c11, col] = 1.0
    col += 1
    # m4 = d(g-e)
    U[u_d, col] = 1.0
    V[v_g, col] = 1.0; V[v_e, col] = -1.0
    W[w_c00, col] = 1.0; W[w_c10, col] = 1.0
    col += 1
    # m5 = (a+b)h
    U[u_a, col] = 1.0; U[u_b, col] = 1.0
    V[v_h, col] = 1.0
    W[w_c00, col] = -1.0; W[w_c01, col] = 1.0
    col += 1
    # m6 = (c-a)(e+f)
    U[u_c, col] = 1.0; U[u_a, col] = -1.0
    V[v_e, col] = 1.0; V[v_f, col] = 1.0
    W[w_c11, col] = 1.0
    col += 1
    # m7 = (b-d)(g+h)
    U[u_b, col] = 1.0; U[u_d, col] = -1.0
    V[v_g, col] = 1.0; V[v_h, col] = 1.0
    W[w_c00, col] = 1.0
    col += 1
    return col


def _build_single_for_column(k2, col, U, V, W, p, n):
    """Add a schoolbook 2x2x2x1 block (rank 4) for output column k2."""
    v_e2 = k2
    v_g2 = p + k2
    w_c02 = k2 * n + 0
    w_c12 = k2 * n + 1
    U[0, col] = 1.0; V[v_e2, col] = 1.0; W[w_c02, col] = 1.0; col += 1
    U[1, col] = 1.0; V[v_g2, col] = 1.0; W[w_c02, col] = 1.0; col += 1
    U[2, col] = 1.0; V[v_e2, col] = 1.0; W[w_c12, col] = 1.0; col += 1
    U[3, col] = 1.0; V[v_g2, col] = 1.0; W[w_c12, col] = 1.0; col += 1
    return col


def _build_strassen_decomposition(pairing=None):
    """Build a Strassen-based decomposition. pairing: list of column groups,
    each group is a tuple of 2 columns (Strassen) or 1 column (schoolbook)."""
    n, m, p = 2, 2, 11
    if pairing is None:
        pairing = [(0,1),(2,3),(4,5),(6,7),(8,9),(10,)]
    # compute rank
    rank = sum(7 if len(g) == 2 else 4 for g in pairing)
    U = np.zeros((n * m, rank), dtype=np.float64)
    V = np.zeros((m * p, rank), dtype=np.float64)
    W = np.zeros((p * n, rank), dtype=np.float64)
    col = 0
    for g in pairing:
        if len(g) == 2:
            col = _build_strassen_for_group(g[0], g[1], col, U, V, W, p, n)
        else:
            col = _build_single_for_column(g[0], col, U, V, W, p, n)
    return U, V, W


def _build_schoolbook():
    """Build the schoolbook rank-44 decomposition."""
    n, m, p = 2, 2, 11
    rank = n * m * p
    U = np.zeros((n * m, rank), dtype=np.float64)
    V = np.zeros((m * p, rank), dtype=np.float64)
    W = np.zeros((p * n, rank), dtype=np.float64)
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1.0
                V[j * p + k, col] = 1.0
                W[k * n + i, col] = 1.0
                col += 1
    return U, V, W


def _svd_numerical_rank(M, tol_factor=1e-9):
    """Estimate numerical rank of M via SVD."""
    try:
        s = np.linalg.svd(M, compute_uv=False)
        if s.size == 0:
            return 0
        tol = max(M.shape) * tol_factor * s[0]
        return int(np.sum(s > tol))
    except Exception:
        return M.shape[1]


def _greedy_reduce(U, V, W, T, M, max_time=60, tol=1e-9):
    """Greedy backward elimination: remove columns that are in the span of others."""
    start_time = time.time()
    current_rank = U.shape[1]
    T_flat = T.flatten()
    active = list(range(current_rank))

    def residual_without(idx_list):
        others = [a for a in active if a not in idx_list]
        if len(others) == 0:
            return np.linalg.norm(T_flat)
        M_sub = M[:, others]
        try:
            sol = np.linalg.lstsq(M_sub, T_flat, rcond=1e-12)[0]
            return np.linalg.norm(M_sub @ sol - T_flat)
        except Exception:
            return np.inf

    removed_any = False
    # single removals, repeated
    progress = True
    while progress and (time.time() - start_time) < max_time:
        progress = False
        if len(active) <= 1:
            break
        for idx in range(len(active)):
            if (time.time() - start_time) >= max_time:
                break
            r = active[idx]
            if residual_without([r]) < tol:
                active.remove(r)
                removed_any = True
                progress = True
                break
    if not removed_any:
        return None
    keep = sorted(active)
    U_red = U[:, keep]; V_red = V[:, keep]; W_red = W[:, keep]
    if _verify(U_red, V_red, W_red, T):
        return U_red, V_red, W_red
    return None


def _double_removal(U, V, W, T, M, max_time=30, tol=1e-9):
    """Try removing pairs of columns simultaneously."""
    start_time = time.time()
    current_rank = U.shape[1]
    if current_rank <= 2:
        return None
    T_flat = T.flatten()
    active = list(range(current_rank))
    removed_any = False

    for _ in range(60):
        if (time.time() - start_time) >= max_time:
            break
        if len(active) <= 2:
            break
        found = False
        for i in range(len(active)):
            if (time.time() - start_time) >= max_time:
                break
            for j in range(i + 1, len(active)):
                if (time.time() - start_time) >= max_time:
                    break
                r1, r2 = active[i], active[j]
                others = [a for a in active if a != r1 and a != r2]
                if len(others) == 0:
                    continue
                M_sub = M[:, others]
                try:
                    sol = np.linalg.lstsq(M_sub, T_flat, rcond=1e-12)[0]
                    resid = np.linalg.norm(M_sub @ sol - T_flat)
                except Exception:
                    continue
                if resid < tol:
                    active.remove(r2); active.remove(r1)
                    removed_any = True
                    found = True
                    break
            if found:
                break
        if not found:
            break
    if removed_any:
        keep = sorted(active)
        U_red = U[:, keep]; V_red = V[:, keep]; W_red = W[:, keep]
        if _verify(U_red, V_red, W_red, T):
            return U_red, V_red, W_red
    return None


def _try_direct_low_rank(T, target_rank, max_time=20):
    """Attempt a direct rank-`target_rank` decomposition via randomized
    initialization + alternating least squares on the matricization.
    Returns (U,V,W) or None."""
    start_time = time.time()
    n, m, p = 2, 2, 11
    dim_a, dim_b, dim_c = n * m, m * p, p * n
    T_flat = T.flatten()

    rng = np.random.default_rng(12345)
    best = None
    best_resid = np.inf

    for trial in range(40):
        if (time.time() - start_time) >= max_time:
            break
        U = rng.standard_normal((dim_a, target_rank))
        V = rng.standard_normal((dim_b, target_rank))
        W = rng.standard_normal((dim_c, target_rank))
        for it in range(300):
            if (time.time() - start_time) >= max_time:
                break
            # update U: for each r, U[:,r] = argmin ||T - sum_r U_r ⊗ V_r ⊗ W_r||
            # Build the linear system: T_flat = M x where M[:,r] = vec(V_r ⊗ W_r)
            M_vw = np.zeros((dim_a * dim_b * dim_c, target_rank))
            for r in range(target_rank):
                M_vw[:, r] = np.einsum('a,b->ab', V[:, r], W[:, r]).flatten()
            # We need U such that M_vw @ u = T_flat where u stacks U columns in order
            # vec(U_r) ordering: M_vw column r corresponds to U[:,r]
            try:
                U_flat = np.linalg.lstsq(M_vw, T_flat, rcond=None)[0]
            except Exception:
                break
            U = U_flat.reshape(dim_a, target_rank)
            # update V
            M_uw = np.zeros((dim_a * dim_b * dim_c, target_rank))
            for r in range(target_rank):
                M_uw[:, r] = np.einsum('a,b->ab', U[:, r], W[:, r]).flatten()
            try:
                V_flat = np.linalg.lstsq(M_uw, T_flat, rcond=None)[0]
            except Exception:
                break
            V = V_flat.reshape(dim_b, target_rank)
            # update W
            M_uv = np.zeros((dim_a * dim_b * dim_c, target_rank))
            for r in range(target_rank):
                M_uv[:, r] = np.einsum('a,b->ab', U[:, r], V[:, r]).flatten()
            try:
                W_flat = np.linalg.lstsq(M_uv, T_flat, rcond=None)[0]
            except Exception:
                break
            W = W_flat.reshape(dim_c, target_rank)
            # compute residual
            T_decomp = np.einsum('ar,br,cr->abc', U, V, W)
            resid = np.linalg.norm(T_decomp - T)
            if resid < best_resid:
                best_resid = resid
                best = (U.copy(), V.copy(), W.copy())
            if resid < 1e-10:
                break

    if best is not None and best_resid < 1e-9:
        U, V, W = best
        if _verify(U, V, W, T):
            return U, V, W
    return None


def construct():
    start_time = time.time()
    DEADLINE = 145  # safety margin before 150s kill

    T = _build_target_tensor()

    # Candidate structured constructions with different column groupings.
    candidates = []
    # Standard: 5 pairs + 1 single (rank 39)
    candidates.append(_build_strassen_decomposition([(0,1),(2,3),(4,5),(6,7),(8,9),(10,)]))
    # Different pairing: leave a different single column out
    for single in range(11):
        pairs = []
        used = set([single])
        remaining = [k for k in range(11) if k != single]
        for idx in range(0, len(remaining), 2):
            if idx + 1 < len(remaining):
                pairs.append((remaining[idx], remaining[idx+1]))
        pairing = pairs + [(single,)]
        candidates.append(_build_strassen_decomposition(pairing))
    # Schoolbook
    candidates.append(_build_schoolbook())

    # Filter to valid, keep minimum rank
    valid = []
    for cand in candidates:
        U, V, W = cand
        if U.shape[1] > 0 and _verify(U, V, W, T):
            valid.append(cand)
    if not valid:
        # last resort
        U, V, W = _build_schoolbook()
        return {'U': U.tolist(), 'V': V.tolist(), 'W': W.tolist()}

    # Pick the one with smallest rank (tie: first)
    valid.sort(key=lambda c: c[0].shape[1])
    best_U, best_V, best_W = valid[0]
    best_rank = best_U.shape[1]

    # Aggressive rank reduction on the best candidate
    elapsed = time.time() - start_time
    remaining = DEADLINE - elapsed - 4
    if remaining > 3 and best_rank > 1:
        M = _matricize(best_U, best_V, best_W)
        # SVD-based rank detection
        num_rank = _svd_numerical_rank(M)
        if num_rank < best_rank:
            # keep first num_rank columns after SVD ordering (approx)
            try:
                Q, R, P = np.linalg.qr(M, mode='complete')
                diag_abs = np.abs(np.diag(R))
                tol = max(M.shape) * np.finfo(float).eps * (diag_abs[0] if diag_abs.size else 1.0)
                keep = np.sort(np.where(diag_abs > tol)[0])
                if len(keep) < best_rank:
                    U_r, V_r, W_r = best_U[:, keep], best_V[:, keep], best_W[:, keep]
                    if _verify(U_r, V_r, W_r, T) and U_r.shape[1] < best_rank:
                        best_U, best_V, best_W = U_r, V_r, W_r
                        best_rank = U_r.shape[1]
            except Exception:
                pass

        # Greedy single removal
        elapsed = time.time() - start_time
        remaining = DEADLINE - elapsed - 4
        if remaining > 3 and best_rank > 1:
            M = _matricize(best_U, best_V, best_W)
            red = _greedy_reduce(best_U, best_V, best_W, T, M, max_time=remaining)
            if red is not None and red[0].shape[1] < best_rank:
                best_U, best_V, best_W = red
                best_rank = best_U.shape[1]

        # Double removal
        elapsed = time.time() - start_time
        remaining = DEADLINE - elapsed - 4
        if remaining > 5 and best_rank > 2:
            M = _matricize(best_U, best_V, best_W)
            red2 = _double_removal(best_U, best_V, best_W, T, M, max_time=remaining)
            if red2 is not None and red2[0].shape[1] < best_rank:
                best_U, best_V, best_W = red2
                best_rank = red2[0].shape[1]

    # Try direct low-rank solvers for ranks below current best
    elapsed = time.time() - start_time
    remaining = DEADLINE - elapsed - 4
    if remaining > 5 and best_rank > 2:
        for target_rank in range(best_rank - 1, 2, -1):
            elapsed = time.time() - start_time
            remaining = DEADLINE - elapsed - 3
            if remaining < 5:
                break
            res = _try_direct_low_rank(T, target_rank, max_time=min(remaining, 20))
            if res is not None and res[0].shape[1] < best_rank:
                best_U, best_V, best_W = res
                best_rank = best_U.shape[1]
                if best_rank <= 3:
                    break

    # Final verification
    if not _verify(best_U, best_V, best_W, T):
        U, V, W = _build_strassen_decomposition([(0,1),(2,3),(4,5),(6,7),(8,9),(10,)])
        if _verify(U, V, W, T):
            return {'U': U.tolist(), 'V': V.tolist(), 'W': W.tolist()}
        U, V, W = _build_schoolbook()
        return {'U': U.tolist(), 'V': V.tolist(), 'W': W.tolist()}

    return {'U': best_U.tolist(), 'V': best_V.tolist(), 'W': best_W.tolist()}
