"""Search for a rank-minimal exact tensor decomposition of 3x5 * 5x6 multiplication."""
import time
import random
import numpy as np

DIMENSIONS = [3, 5, 6]

# EVOLVE-BLOCK-START
_DEADLINE = time.time() + 110.0   # hard wall-clock guard (evaluator kills at 150s)


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
    """Fast verification: check that sum_c U[a,c]*V[b,c]*W[c'] = delta for all (a,b,c')."""
    U = np.asarray(U, dtype=float)
    V = np.asarray(V, dtype=float)
    W = np.asarray(W, dtype=float)
    nm = n * m
    mp = m * p
    pn = p * n
    r = U.shape[1]
    VT = V.T  # (r, mp)

    for c in range(pn):
        k0 = c // n
        i0 = c % n
        U_scaled = U * W[c, :]  # (nm, r)
        M_c = U_scaled @ VT     # (nm, mp)

        # Check that M_c[i0*m+j, j*p+k0] == 1 for all j
        for j in range(m):
            a = i0 * m + j
            b = j * p + k0
            if abs(M_c[a, b] - 1.0) > 1e-9:
                return False

        # Check that all other entries are 0
        M_check = M_c.copy()
        for j in range(m):
            M_check[i0 * m + j, j * p + k0] = 0.0
        if np.max(np.abs(M_check)) > 1e-9:
            return False

    return True


def _col_in_span(M_others, M_target, tol=1e-10):
    """Check if M_target is in the column span of M_others using QR."""
    if M_others.shape[1] == 0:
        return np.allclose(M_target, 0, atol=tol)
    # Use SVD for robustness
    U_s, s, Vt = np.linalg.svd(M_others, full_matrices=False)
    rank = np.sum(s > tol * max(M_others.shape))
    if rank == 0:
        return np.allclose(M_target, 0, atol=tol)
    # Project target onto span
    U_r = U_s[:, :rank]
    proj = U_r @ (U_r.T @ M_target)
    residual = M_target - proj
    return np.linalg.norm(residual) <= tol * max(1.0, np.linalg.norm(M_target))


def _try_reduce_rank(U, V, W, n, m, p, max_iterations=100):
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
        for target in range(r):
            others = [i for i in range(r) if i != target]
            if len(others) == 0:
                continue

            # Check each factor's target column is in the span of its others.
            ok = True
            for M in (U, V, W):
                M_others = M[:, others]
                M_target = M[:, target]
                if not _col_in_span(M_others, M_target):
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


def _generate_partitions(n, m, p):
    """Generate candidate groupings for each dimension. Each dimension is split
    into contiguous groups of size 1 or 2."""
    def _partitions(dim):
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

    ip = _partitions(n)
    jp = _partitions(m)
    kp = _partitions(p)
    return ip, jp, kp


def _circulant_embed(n, m, p):
    """Embed the n×m×p multiplication tensor into a circulant structure.
    
    Key idea: embed A (n×m) and B (m×p) into larger matrices where
    the product can be computed with fewer multiplications using
    circulant/cyclic convolution tricks, then extract the result.
    
    For 3×5×6: embed into a structure where we can exploit cyclic symmetry.
    We try to find a rank-r decomposition by solving a linear system.
    """
    nm, mp, pn = n * m, m * p, p * n
    full_rank = n * m * p
    
    # Build the multiplication tensor as a matrix: T is (nm*mp) x pn
    # T[(i*m+j)*(mp) + (j*p+k), k*n+i] = 1
    T = np.zeros((nm * mp, pn))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a * mp + b, c] = 1.0
    
    # Try to find a low-rank factorization: T ≈ U_flat @ V_flat^T
    # where U_flat is (nm*mp, r) and V_flat is (pn, r)
    # Then reshape U_flat into U (nm, r) and V_flat into V (mp, r)
    # and W is separate (pn, r)
    
    # Actually, the tensor decomposition is:
    # T[a,b,c] = sum_r U[a,r] * V[b,r] * W[c,r]
    # This is NOT a standard matrix rank factorization of T.
    
    # Alternative: use the fact that for fixed c, T[:,:,c] = outer(U_c, V_c) * W[c,:]
    # where U_c = U * W[c,:] (element-wise scaling of rows)
    # So T[:,:,c] = (U * W[c,:]) @ V.T
    # This means T[:,:,c] has rank <= r for each c.
    
    # Let's try: for each c, T[:,:,c] should be a rank-1 matrix (if r=1 per c)
    # but we need a shared r across all c.
    
    # Strategy: use SVD to find approximate low-rank structure
    # For each c, compute SVD of T[:,:,c] and check if a small number of
    # components suffice.
    
    # Better strategy: formulate as an optimization problem
    # Minimize sum_{a,b,c} (T[a,b,c] - sum_r U[a,r]*V[b,r]*W[c,r])^2
    # subject to U,V,W having the right shapes.
    
    # This is a non-convex tensor factorization. Use alternating least squares.
    
    # Initialize with schoolbook
    U = np.zeros((nm, full_rank))
    V = np.zeros((mp, full_rank))
    W = np.zeros((pn, full_rank))
    c = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, c] = 1
                V[j * p + k, c] = 1
                W[k * n + i, c] = 1
                c += 1
    
    # Try to reduce rank by projecting onto lower-rank subspace
    # and checking if the tensor is preserved.
    
    # For each target rank r_target, try to find U,V,W of that rank.
    # Use the fact that if we fix two factors, the third is determined.
    
    best_rank = full_rank
    best_U, best_V, best_W = U, V, W
    
    for r_target in range(max(20, full_rank - 20), full_rank):
        if time.time() > _DEADLINE:
            break
        # Try to find rank-r_target decomposition
        # Initialize by truncating SVD of the tensor
        # Reshape T into a 3D tensor
        T3 = T.reshape(nm, mp, pn)
        
        # Use tensor SVD (mode-1, mode-2, mode-3)
        # Mode-1 unfolding: (nm, mp*pn)
        T1 = T3.reshape(nm, -1)
        U1, s1, Vt1 = np.linalg.svd(T1, full_matrices=False)
        
        # Mode-2 unfolding: (mp, nm*pn)
        T2 = T3.transpose(1, 0, 2).reshape(mp, -1)
        U2, s2, Vt2 = np.linalg.svd(T2, full_matrices=False)
        
        # Mode-3 unfolding: (pn, nm*mp)
        T3u = T3.transpose(2, 0, 1).reshape(pn, -1)
        U3, s3, Vt3 = np.linalg.svd(T3u, full_matrices=False)
        
        # Initialize factors from SVD
        U_init = U1[:, :r_target] * np.sqrt(s1[:r_target])
        V_init = U2[:, :r_target] * np.sqrt(s2[:r_target])
        W_init = U3[:, :r_target] * np.sqrt(s3[:r_target])
        
        # Alternating least squares
        for iteration in range(50):
            if time.time() > _DEADLINE:
                break
            # Update U: for each a, U[a,:] = argmin sum_{b,c} (T[a,b,c] - sum_r U[a,r]*V[b,r]*W[c,r])^2
            # This is: U[a,:] @ (V[b,:] * W[c,:]) = T[a,b,c] for each (b,c)
            # Stack: for fixed a, the equation is U[a,:] @ M_a = T[a,:,:].flatten()
            # where M_a[r, (b,c)] = V[b,r] * W[c,r]
            
            # This is expensive. Instead, use a simpler update.
            # For each a: U[a,:] = (sum_{b,c} T[a,b,c] * (V[b,:]*W[c,:])^T) / (sum_{b,c} (V[b,:]*W[c,:])^2)
            
            # Actually, let's use a different approach: 
            # Fix V and W, solve for U via least squares.
            # T[a,b,c] = sum_r U[a,r] * V[b,r] * W[c,r]
            # For fixed a: T[a,:,:] = U[a,:] @ (V * W_expand)
            # where V * W_expand has shape (r, mp*pn) with [V*W_expand][r, b*pn+c] = V[b,r]*W[c,r]
            
            # Build the combined matrix
            # VW[r, b*pn+c] = V[b,r] * W[c,r]
            VW = np.einsum('br,cr->bcr', V_init, W_init).reshape(r_target, mp * pn)
            # T_a[a, b*pn+c] = T[a,b,c]
            T_flat = T3.reshape(nm, mp * pn)
            
            # Solve: U_init @ VW^T = T_flat
            # U_init = T_flat @ (VW^T)^+ 
            # Using lstsq
            try:
                U_new, _, _, _ = np.linalg.lstsq(VW.T, T_flat.T, rcond=None)
                U_init = U_new.T
            except:
                break
            
            # Update V similarly
            UW = np.einsum('ar,cr->acr', U_init, W_init).reshape(r_target, nm * pn)
            T2_flat = T3.transpose(1, 0, 2).reshape(mp, nm * pn)
            try:
                V_new, _, _, _ = np.linalg.lstsq(UW.T, T2_flat.T, rcond=None)
                V_init = V_new.T
            except:
                break
            
            # Update W
            UV = np.einsum('ar,br->abr', U_init, V_init).reshape(r_target, nm * mp)
            T3_flat = T3.transpose(2, 0, 1).reshape(pn, nm * mp)
            try:
                W_new, _, _, _ = np.linalg.lstsq(UV.T, T3_flat.T, rcond=None)
                W_init = W_new.T
            except:
                break
        
        # Check if this decomposition is exact
        if _verify_fast(U_init, V_init, W_init, n, m, p):
            if r_target < best_rank:
                best_rank = r_target
                best_U, best_V, best_W = U_init, V_init, W_init
    
    if best_rank < full_rank:
        return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
    return None


def construct():
    t0 = time.time()
    n, m, p = DIMENSIONS
    nm, mp, pn = n * m, m * p, p * n
    full = n * m * p

    best = _schoolbook(n, m, p)
    best_rank = full

    def timed():
        return time.time() > _DEADLINE

    # Strategy 1: recursive Strassen over many partitions, keep the best valid one.
    ip, jp, kp = _generate_partitions(n, m, p)
    candidates = []
    for ig in ip:
        for jg in jp:
            for kg in kp:
                candidates.append((ig, jg, kg))

    # Order candidates by expected rank ascending (prefer more 2x2x2 blocks).
    def expected_rank(ig, jg, kg):
        tot = 0
        for a in ig:
            for b in jg:
                for c in kg:
                    tot += 7 if (len(a) == 2 and len(b) == 2 and len(c) == 2) \
                        else len(a) * len(b) * len(c)
        return tot
    candidates.sort(key=lambda t: expected_rank(t[0], t[1], t[2]))

    for (ig, jg, kg) in candidates:
        if timed():
            break
        cand = _build_recursive_strassen(n, m, p, ig, jg, kg)
        if _verify_fast(cand["U"], cand["V"], cand["W"], n, m, p):
            r = len(cand["U"][0])
            if r < best_rank:
                best = cand
                best_rank = r
            # try to shave further via linear-dependency reduction
            U, V, W = _try_reduce_rank(cand["U"], cand["V"], cand["W"], n, m, p, max_iterations=80)
            if _verify_fast(U, V, W, n, m, p):
                new_rank = len(U[0])
                if new_rank < best_rank:
                    best = {"U": U, "V": V, "W": W}
                    best_rank = new_rank

    # Strategy 2: More aggressive rank reduction on the best candidate found.
    if not timed():
        U_b = np.array(best["U"], dtype=float)
        V_b = np.array(best["V"], dtype=float)
        W_b = np.array(best["W"], dtype=float)
        
        # Try multiple rounds of reduction with different orderings
        for attempt in range(5):
            if timed():
                break
            rng = random.Random(42 + attempt)
            cols = list(range(U_b.shape[1]))
            rng.shuffle(cols)
            U_sh = U_b[:, cols]
            V_sh = V_b[:, cols]
            W_sh = W_b[:, cols]
            U_red, V_red, W_red = _try_reduce_rank(U_sh.tolist(), V_sh.tolist(), W_sh.tolist(),
                                                   n, m, p, max_iterations=30)
            if _verify_fast(U_red, V_red, W_red, n, m, p):
                new_rank = len(U_red[0])
                if new_rank < best_rank:
                    best = {"U": U_red, "V": V_red, "W": W_red}
                    best_rank = new_rank

    # Strategy 3: Try circulant embedding / tensor SVD approach
    if not timed():
        circ_result = _circulant_embed(n, m, p)
        if circ_result is not None:
            if _verify_fast(circ_result["U"], circ_result["V"], circ_result["W"], n, m, p):
                new_rank = len(circ_result["U"][0])
                if new_rank < best_rank:
                    best = circ_result
                    best_rank = new_rank

    # Final verification; fall back to schoolbook if anything went wrong.
    if not _verify_fast(best["U"], best["V"], best["W"], n, m, p):
        best = _schoolbook(n, m, p)

    return best


# EVOLVE-BLOCK-END
