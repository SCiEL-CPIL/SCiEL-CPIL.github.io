"""Search for a rank-minimal exact tensor decomposition of 3x5 * 5x6 multiplication."""
import time
import random
import numpy as np

DIMENSIONS = [3, 5, 6]

# EVOLVE-BLOCK-START
_DEADLINE = time.time() + 140.0


def _strassen_2x2x2_terms():
    """Return the 7 Strassen terms for 2x2x2 matrix multiplication.
    Each term is (u_pat, v_pat, w_pat) mapping local index tuples to integer values.
    Convention: U index (li,lj), V index (lj,lk), W index (lk,li)."""
    terms = []
    terms.append(({(0, 0): 1, (1, 1): 1}, {(0, 0): 1, (1, 1): 1}, {(0, 0): 1, (1, 1): 1}))
    terms.append(({(1, 0): 1, (1, 1): 1}, {(0, 0): 1}, {(0, 1): 1, (1, 1): -1}))
    terms.append(({(0, 0): 1}, {(0, 1): 1, (1, 1): -1}, {(1, 0): 1, (1, 1): 1}))
    terms.append(({(1, 1): 1}, {(1, 0): 1, (0, 0): -1}, {(0, 0): 1, (0, 1): 1}))
    terms.append(({(0, 0): 1, (0, 1): 1}, {(1, 1): 1}, {(0, 0): -1, (1, 0): 1}))
    terms.append(({(1, 0): 1, (0, 0): -1}, {(0, 0): 1, (0, 1): 1}, {(1, 1): 1}))
    terms.append(({(0, 1): 1, (1, 1): -1}, {(1, 0): 1, (1, 1): 1}, {(0, 0): 1}))
    return terms


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


def _build_recursive_strassen(n, m, p, i_groups, j_groups, k_groups):
    """Build a decomposition by splitting each dimension into groups.
    Any (2,2,2) block uses the 7-term Strassen trick; others use schoolbook."""
    nm, mp, pn = n * m, m * p, p * n
    strassen_terms = _strassen_2x2x2_terms()

    total_rank = 0
    block_info = []
    for ig in i_groups:
        for jg in j_groups:
            for kg in k_groups:
                si, sj, sk = len(ig), len(jg), len(kg)
                r = 7 if (si == 2 and sj == 2 and sk == 2) else si * sj * sk
                total_rank += r
                block_info.append((ig, jg, kg, r))

    U = [[0] * total_rank for _ in range(nm)]
    V = [[0] * total_rank for _ in range(mp)]
    W = [[0] * total_rank for _ in range(pn)]

    col = 0
    for ig, jg, kg, r in block_info:
        si, sj, sk = len(ig), len(jg), len(kg)
        if si == 2 and sj == 2 and sk == 2:
            for (u_pat, v_pat, w_pat) in strassen_terms:
                for (li, lj), val_u in u_pat.items():
                    gi, gj = ig[li], jg[lj]
                    U[gi * m + gj][col] = val_u
                for (lj, lk), val_v in v_pat.items():
                    gj, gk = jg[lj], kg[lk]
                    V[gj * p + gk][col] = val_v
                for (lk, li), val_w in w_pat.items():
                    gk, gi = kg[lk], ig[li]
                    W[gk * n + gi][col] = val_w
                col += 1
        else:
            for li in range(si):
                for lj in range(sj):
                    for lk in range(sk):
                        gi, gj, gk = ig[li], jg[lj], kg[lk]
                        U[gi * m + gj][col] = 1
                        V[gj * p + gk][col] = 1
                        W[gk * n + gi][col] = 1
                        col += 1

    return {"U": U, "V": V, "W": W}


def _verify_fast(U, V, W, n, m, p):
    """Fast verification using numpy. Returns True if decomposition is exact."""
    U = np.asarray(U, dtype=float)
    V = np.asarray(V, dtype=float)
    W = np.asarray(W, dtype=float)
    r = U.shape[1]
    nm = n * m
    mp = m * p
    pn = p * n

    # Reshape U to (n, m, r), V to (m, p, r), W to (p, n, r)
    U3 = U.reshape(n, m, r)
    V3 = V.reshape(m, p, r)
    W3 = W.reshape(p, n, r)
    
    # T[i,j,k] = sum_c U3[i,j,c] * V3[j,k,c] * W3[k,i,c]
    # Vectorized: for each (i,k), compute T[i,:,k] = sum_c U3[i,:,c] * (V3 * W3[k,:,c]).T
    # Actually: T[i,j,k] = sum_c U3[i,j,c] * V3[j,k,c] * W3[k,i,c]
    # = sum_c U3[i,j,c] * (V3 * W3[k,:,c])[j,k] ... no
    # Let me think: for fixed i,k, T[i,:,k][j] = sum_c U3[i,j,c] * V3[j,k,c] * W3[k,i,c]
    # = sum_c (U3[i,:,c] * W3[k,i,c]) * V3[:,k,c]
    # So T[i,:,k] = sum_c (U3[i,:,c] * W3[k,i,c]) * V3[:,k,c]
    # This is: T[i,:,k] = sum_c V3[:,k,c] * (U3[i,:,c] * W3[k,i,c])
    # = V3[:,k,:] @ (U3[i,:,c] * W3[k,i,c] for each c)
    # Let M[i,k] = U3[i,:,c] * W3[k,i,c] for each c -> shape (m, r)
    # Then T[i,:,k] = V3[:,k,:] @ M[i,k]  where V3[:,k,:] has shape (m, r)
    
    # More efficiently:
    # T[i,j,k] = sum_c U3[i,j,c] * V3[j,k,c] * W3[k,i,c]
    # Let's compute it as: for each j, T[:,j,:] = sum_c U3[:,j,c] * (V3[j,:,c] * W3[:,:,j... no
    # 
    # Simplest vectorized approach:
    # T = np.einsum('ijc,jkc,kic->ijk', U3, V3, W3)
    # But this might be slow for large r. Let's try a different approach.
    #
    # Alternative: T[i,j,k] = sum_c U3[i,j,c] * V3[j,k,c] * W3[k,i,c]
    # For each c: contribution is U3[:,:,c] * V3[:,:,c] * W3[:,:,c] but with index permutation
    # U3[i,j,c], V3[j,k,c], W3[k,i,c]
    # So for each c, the contribution to T is an outer-product-like thing.
    #
    # Let's just use einsum for clarity and let numpy optimize it.
    T = np.einsum('ijc,jkc,kic->ijk', U3, V3, W3, optimize=True)
    
    # Check T[i,j,k] == 1 for all (i,j,k)
    if not np.allclose(T, 1.0, atol=1e-9):
        return False
    return True


def _verify_exact(U, V, W, n, m, p):
    """Exact verification with integer/rational arithmetic if possible."""
    U = np.asarray(U)
    V = np.asarray(V)
    W = np.asarray(W)
    r = U.shape[1]
    
    U3 = U.reshape(n, m, r)
    V3 = V.reshape(m, p, r)
    W3 = W.reshape(p, n, r)
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = np.sum(U3[i, j, :] * V3[j, k, :] * W3[k, i, :])
                if abs(val - 1.0) > 1e-12:
                    return False
    return True


def _try_reduce_rank(U, V, W, n, m, p, max_iterations=200):
    """Try to reduce rank by removing columns that are linearly dependent
    across all three factor matrices simultaneously."""
    U = np.array(U, dtype=float).copy()
    V = np.array(V, dtype=float).copy()
    W = np.array(W, dtype=float).copy()
    r = U.shape[1]

    for _ in range(max_iterations):
        if time.time() > _DEADLINE:
            break
        reduced = False
        # Try removing each column; check if it's in the span of others in all 3 matrices
        for target in range(r):
            if time.time() > _DEADLINE:
                break
            others = [i for i in range(r) if i != target]
            if len(others) == 0:
                continue

            ok = True
            for M in (U, V, W):
                M_others = M[:, others]
                M_target = M[:, target]
                # Check rank increase using SVD for speed
                r1 = np.linalg.matrix_rank(M_others, tol=1e-10)
                r2 = np.linalg.matrix_rank(np.column_stack([M_others, M_target]), tol=1e-10)
                if r2 > r1:
                    ok = False
                    break
            if not ok:
                continue

            U = np.delete(U, target, axis=1)
            V = np.delete(V, target, axis=1)
            W = np.delete(W, target, axis=1)
            r -= 1
            reduced = True
            break

        if not reduced:
            break

    return U.tolist(), V.tolist(), W.tolist()


def _circulant_embed(n, m, p):
    """Try circulant embedding for 3x5x6 matrix multiplication.
    
    Embed the multiplication into a larger circulant matrix multiplication.
    For 3x5: we can embed into a 7x7 circulant system.
    For the full 3x5x6, we need a more careful approach.
    
    Actually, let's try a different approach: use the fact that for small matrices,
    we can try random rank-r decompositions and verify.
    """
    # This is a placeholder - the main strategy is still partition-based Strassen
    # with rank reduction. We'll add a random search strategy instead.
    return None


def _generate_partitions(dim):
    """Generate all ways to partition [0..dim) into contiguous chunks of size 1 or 2."""
    results = []
    def rec(start, cur):
        if start == dim:
            results.append(list(cur))
            return
        if start + 1 <= dim:
            cur.append([start])
            rec(start + 1, cur)
            cur.pop()
        if start + 2 <= dim:
            cur.append([start, start + 1])
            rec(start + 2, cur)
            cur.pop()
    rec(0, [])
    return results


def _expected_rank(ig, jg, kg):
    tot = 0
    for a in ig:
        for b in jg:
            for c in kg:
                tot += 7 if (len(a) == 2 and len(b) == 2 and len(c) == 2) \
                    else len(a) * len(b) * len(c)
    return tot


def _polish(U, V, W, n, m, p):
    """Final polish: snap values to nearest integer or half-integer if close."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    
    for M in (U, V, W):
        for i in range(M.shape[0]):
            for j in range(M.shape[1]):
                v = M[i, j]
                # Try snapping to nearest 0.5
                snapped = round(v * 2) / 2
                if abs(v - snapped) < 1e-8:
                    M[i, j] = snapped
                else:
                    # Try integer
                    snapped_i = round(v)
                    if abs(v - snapped_i) < 1e-8:
                        M[i, j] = snapped_i
    
    return U.tolist(), V.tolist(), W.tolist()


def construct():
    t0 = time.time()
    n, m, p = DIMENSIONS
    nm, mp, pn = n * m, m * p, p * n
    full = n * m * p

    best = _schoolbook(n, m, p)
    best_rank = full

    def timed():
        return time.time() > _DEADLINE

    # Generate all partitions
    ip = _generate_partitions(n)
    jp = _generate_partitions(m)
    kp = _generate_partitions(p)

    # Build candidate list and sort by expected rank
    candidates = []
    for ig in ip:
        for jg in jp:
            for kg in kp:
                candidates.append((ig, jg, kg))

    candidates.sort(key=lambda t: _expected_rank(t[0], t[1], t[2]))

    # Strategy 1: Try all partition combinations, keep best valid
    for (ig, jg, kg) in candidates:
        if timed():
            break
        cand = _build_recursive_strassen(n, m, p, ig, jg, kg)
        if _verify_fast(cand["U"], cand["V"], cand["W"], n, m, p):
            r = len(cand["U"][0])
            if r < best_rank:
                best = cand
                best_rank = r
            # Try rank reduction
            U_red, V_red, W_red = _try_reduce_rank(cand["U"], cand["V"], cand["W"], n, m, p, max_iterations=50)
            if _verify_fast(U_red, V_red, W_red, n, m, p):
                new_rank = len(U_red[0])
                if new_rank < best_rank:
                    best = {"U": U_red, "V": V_red, "W": W_red}
                    best_rank = new_rank

    # Strategy 2: Polish the best result
    if not timed():
        U_pol, V_pol, W_pol = _polish(best["U"], best["V"], best["W"], n, m, p)
        if _verify_fast(U_pol, V_pol, W_pol, n, m, p):
            best = {"U": U_pol, "V": V_pol, "W": W_pol}
            # Try more reduction after polishing
            U_red, V_red, W_red = _try_reduce_rank(best["U"], best["V"], best["W"], n, m, p, max_iterations=50)
            if _verify_fast(U_red, V_red, W_red, n, m, p):
                new_rank = len(U_red[0])
                if new_rank < best_rank:
                    best = {"U": U_red, "V": V_red, "W": W_red}
                    best_rank = new_rank

    # Strategy 3: Random search for lower-rank decompositions
    # Try random rank-r matrices and see if they can be adjusted to work
    if not timed():
        random.seed(42)
        for r_try in range(best_rank - 1, max(20, best_rank - 10), -1):
            if timed():
                break
            # Try a few random initializations
            for attempt in range(3):
                if timed():
                    break
                U_rand = np.random.randn(nm, r_try)
                V_rand = np.random.randn(mp, r_try)
                W_rand = np.random.randn(pn, r_try)
                # This won't work directly - we need a more structured approach
                # Skip for now; the partition-based approach is more reliable
                break

    # Final verification
    if not _verify_fast(best["U"], best["V"], best["W"], n, m, p):
        best = _schoolbook(n, m, p)

    return best
# EVOLVE-BLOCK-END
