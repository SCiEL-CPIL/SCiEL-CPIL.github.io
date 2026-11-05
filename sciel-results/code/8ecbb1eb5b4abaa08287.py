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
    """Vectorized verification: compute all M_c at once and check against expected pattern."""
    U = np.asarray(U, dtype=float)
    V = np.asarray(V, dtype=float)
    W = np.asarray(W, dtype=float)
    nm = n * m
    mp = m * p
    pn = p * n
    r = U.shape[1]
    
    # T[a,b,c] = sum_r U[a,r]*V[b,r]*W[c,r]
    # For each c: M_c = (U * W[c,:]) @ V.T
    # Vectorize: M_all[c, a, b] = sum_r U[a,r]*V[b,r]*W[c,r]
    # = einsum('ar,br,cr->cab', U, V, W)
    # But that's O(pn*nm*mp*r) which is fine for small dims.
    # Alternative: M_all = np.einsum('cr,ar,br->cab', W, U, V)
    M_all = np.einsum('cr,ar,br->cab', W, U, V)  # (pn, nm, mp)
    
    # Build expected tensor
    # Expected: T[k*n+i, i*m+j, j*p+k] = 1, else 0
    expected = np.zeros((pn, nm, mp), dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                c_idx = k * n + i
                a_idx = i * m + j
                b_idx = j * p + k
                expected[c_idx, a_idx, b_idx] = 1.0
    
    if np.max(np.abs(M_all - expected)) > 1e-9:
        return False
    return True


def _try_reduce_rank(U, V, W, n, m, p, max_iterations=100):
    """Try to reduce rank by removing columns that are linearly dependent
    across all three factor matrices simultaneously. Uses rank-increment via
    augmented matrix rank check (faster than separate SVDs)."""
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
            # Use: rank([M_others | M_target]) == rank(M_others)
            # Equivalent: rank(M_others) == rank(augmented)
            ok = True
            for M in (U, V, W):
                M_others = M[:, others]
                M_target = M[:, target]
                # Compute rank of M_others and augmented
                # Use SVD on the smaller matrix
                rank_others = np.linalg.matrix_rank(M_others)
                # Check if target is in span: solve M_others @ x = M_target
                # If residual is ~0, it's in the span
                try:
                    # Use lstsq for numerical stability
                    sol, residuals, rank_aug, sv = np.linalg.lstsq(M_others, M_target, rcond=1e-10)
                    residual = M_others @ sol - M_target
                    if np.linalg.norm(residual) > 1e-8 * np.linalg.norm(M_target) + 1e-10:
                        ok = False
                        break
                except np.linalg.LinAlgError:
                    # Fall back to rank check
                    rank_together = np.linalg.matrix_rank(np.column_stack([M_others, M_target]))
                    if rank_together > rank_others:
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
    into contiguous groups of size 1 or 2 (we only exploit the 2x2x2 Strassen
    block, so groups larger than 2 give no benefit and are treated as schoolbook)."""
    def _partitions(dim):
        # all ways to partition [0..dim) into contiguous chunks of size 1 or 2
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
    # Only evaluate the top candidates to save time for reduction passes
    # For (3,5,6): ip has F(3)=3 ways, jp has F(5)=8 ways, kp has F(6)=13 ways
    # Total: 3*8*13 = 312 candidates. With vectorized verify, this is fast.
    # But reduction is expensive, so limit to top ~50 candidates.
    candidates = candidates[:80]

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
    # The key insight: after Strassen-based construction, there may be additional
    # linear dependencies across the three factor matrices that weren't exploited.
    if not timed():
        U_b = np.array(best["U"], dtype=float)
        V_b = np.array(best["V"], dtype=float)
        W_b = np.array(best["W"], dtype=float)
        
        # Try multiple rounds of reduction with different orderings
        for attempt in range(5):
            if timed():
                break
            # Shuffle column order to find different dependencies
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

    # Strategy 3: Circulant embedding exploiting 3|6 structure.
    # The 3x5x6 multiplication: C = A(3x5) * B(5x6).
    # Since 6 = 2*3, we can think of B as two 5x3 blocks: B = [B1 | B2].
    # C = [A*B1 | A*B2]. The sub-multiplications A(3x5)*B1(5x3) and A(3x5)*B2(5x3)
    # each produce a 3x3 result. We can use a circulant/FFT trick for the 3x3 part.
    #
    # Alternative: use the fact that matrix multiplication (3,5,6) can be embedded
    # into a larger tensor with circulant structure. Specifically, embed the 3x3
    # multiplication into a circulant of size 3 (rank 3 via FFT over Z_3) but
    # this doesn't help directly. Instead, try a direct circulant construction.
    #
    # Key idea: The tensor T[i,j,k] for (3,5,6) has structure T[i,j,k] = 1 iff
    # i in [0,3), j in [0,5), k in [0,6), and the multiplication is standard.
    # We can try to find a low-rank decomposition by exploiting symmetries.
    #
    # For now, try: use the existing best and do more aggressive reduction passes.
    if not timed():
        U_b = np.array(best["U"], dtype=float)
        V_b = np.array(best["V"], dtype=float)
        W_b = np.array(best["W"], dtype=float)
        
        # Multiple passes with different column orderings to find more dependencies
        for attempt in range(10):
            if timed():
                break
            rng = random.Random(1000 + attempt)
            cols = list(range(U_b.shape[1]))
            rng.shuffle(cols)
            U_sh = U_b[:, cols]
            V_sh = V_b[:, cols]
            W_sh = W_b[:, cols]
            U_red, V_red, W_red = _try_reduce_rank(U_sh.tolist(), V_sh.tolist(), W_sh.tolist(),
                                                   n, m, p, max_iterations=50)
            if _verify_fast(U_red, V_red, W_red, n, m, p):
                new_rank = len(U_red[0])
                if new_rank < best_rank:
                    best = {"U": U_red, "V": V_red, "W": W_red}
                    best_rank = new_rank
                    U_b = np.array(U_red, dtype=float)
                    V_b = np.array(V_red, dtype=float)
                    W_b = np.array(W_red, dtype=float)
    
    # Strategy 4: Try a direct circulant/FFT-based construction for (3,5,6).
    # Embed the problem: since p=6=2*n, we can write the multiplication as
    # C[k,i] = sum_j A[i,j] * B[j,k] for k=0..5, i=0..2, j=0..4.
    # Split k into two groups of 3: k1 = k mod 3, k2 = k // 3 (k2 in {0,1}).
    # Then C[k1 + 3*k2, i] = sum_j A[i,j] * B[j, k1 + 3*k2].
    # This is two independent (3,5,3) multiplications (for k2=0 and k2=1).
    # Each (3,5,3) multiplication has schoolbook rank 45. Two of them: 90.
    # But we can potentially share structure. Use Strassen on the (3,3,3) sub-blocks.
    #
    # Actually, let's try: decompose into (3,5,3) + (3,5,3) and apply
    # Strassen-style tricks to each. The (3,3,3) part can use rank-23 (known
    # optimal for 3x3x3 is 23). But we need (3,5,3), not (3,3,3).
    #
    # Simpler: just do more reduction passes on the current best.
    
    # Final verification; fall back to schoolbook if anything went wrong.
    if not _verify_fast(best["U"], best["V"], best["W"], n, m, p):
        best = _schoolbook(n, m, p)

    return best


# EVOLVE-BLOCK-END
