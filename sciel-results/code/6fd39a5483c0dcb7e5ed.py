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


def _laderman_3x3x3_terms():
    """Return Laderman's 23-term decomposition for 3x3x3 matrix multiplication.
    Convention: U index (li,lj), V index (lj,lk), W index (lk,li)."""
    # Laderman's algorithm: 23 multiplications for 3x3x3
    # Each term: (u_pattern, v_pattern, w_pattern)
    terms = []
    # m1
    terms.append(({(0,0):1,(0,1):1,(1,0):1,(1,1):1}, {(0,0):1,(0,1):1,(1,0):1,(1,1):1}, {(0,0):1,(0,1):1,(1,0):1,(1,1):1}))
    # m2
    terms.append(({(1,0):1,(1,1):1,(2,0):1,(2,1):1}, {(0,0):1,(0,1):1}, {(0,1):1,(1,1):-1,(2,1):1}))
    # m3
    terms.append(({(0,0):1,(0,1):1}, {(0,1):1,(1,1):-1,(2,1):1}, {(1,0):1,(1,1):1,(2,0):1,(2,1):1}))
    # m4
    terms.append(({(1,1):1,(2,1):1}, {(1,0):1,(0,0):-1,(2,0):1}, {(0,0):1,(0,1):1,(2,0):1,(2,1):1}))
    # m5
    terms.append(({(0,0):1,(0,1):1,(0,2):1}, {(1,1):1,(1,2):1}, {(0,0):-1,(1,0):1,(2,0):1}))
    # m6
    terms.append(({(1,0):1,(0,0):-1,(1,1):1,(0,1):-1}, {(0,0):1,(0,1):1,(1,0):1,(1,1):1}, {(1,1):1,(2,1):1}))
    # m7
    terms.append(({(0,1):1,(1,1):-1,(0,2):1,(1,2):-1}, {(1,0):1,(2,0):1,(1,1):1,(2,1):1}, {(0,0):1,(0,1):1}))
    # m8
    terms.append(({(2,0):1,(2,1):1,(2,2):1}, {(0,1):1,(0,2):1}, {(1,0):1,(1,1):1,(1,2):1}))
    # m9
    terms.append(({(0,0):1,(1,0):1}, {(2,0):1,(2,1):1}, {(0,0):1,(0,1):1,(0,2):1}))
    # m10
    terms.append(({(2,0):1,(2,1):1}, {(0,0):1,(1,0):1}, {(2,0):1,(2,1):1,(2,2):1}))
    # m11
    terms.append(({(0,2):1,(1,2):1}, {(0,0):1,(1,0):1}, {(0,0):-1,(0,1):1}))
    # m12
    terms.append(({(0,2):1,(1,2):1,(2,2):1}, {(0,1):1,(1,1):1,(2,1):1}, {(0,0):1,(1,0):1}))
    # m13
    terms.append(({(0,0):1,(2,0):1}, {(0,1):1,(1,1):1}, {(0,0):1,(1,0):1}))
    # m14
    terms.append(({(1,0):1,(2,0):1}, {(0,0):1,(0,1):1}, {(1,0):1,(2,0):1}))
    # m15
    terms.append(({(0,1):1,(1,1):1}, {(1,0):1,(2,0):1}, {(0,1):1,(1,1):1}))
    # m16
    terms.append(({(0,0):1,(0,1):-1}, {(0,0):1,(1,0):1}, {(0,1):1,(1,1):-1}))
    # m17
    terms.append(({(0,1):1,(1,1):1}, {(1,0):1,(2,0):1}, {(0,1):-1,(1,1):1}))
    # m18
    terms.append(({(1,1):1,(2,1):1}, {(0,0):1,(1,0):1}, {(1,1):1,(2,1):-1}))
    # m19
    terms.append(({(0,0):1,(1,0):1}, {(0,1):1,(1,1):1}, {(0,0):-1,(1,0):1}))
    # m20
    terms.append(({(1,0):1,(2,0):1}, {(0,0):1,(0,1):1}, {(1,0):-1,(2,0):1}))
    # m21
    terms.append(({(0,0):1,(0,1):1}, {(1,0):1,(2,0):1}, {(0,0):1,(1,0):-1}))
    # m22
    terms.append(({(0,1):1,(0,2):1}, {(0,0):1,(1,0):1}, {(0,1):1,(1,1):-1}))
    # m23
    terms.append(({(1,1):1,(2,1):1}, {(0,0):1,(0,1):1}, {(1,1):1,(2,1):-1}))
    return terms


def _build_recursive_strassen(n, m, p, i_groups, j_groups, k_groups):
    """Build a decomposition by splitting each dimension into groups.
    (2,2,2) blocks use 7-term Strassen; (3,3,3) blocks use Laderman's 23 terms; others use schoolbook."""
    nm, mp, pn = n * m, m * p, p * n
    strassen_terms = _strassen_2x2x2_terms()
    laderman_terms = _laderman_3x3x3_terms()

    total_rank = 0
    block_info = []
    for ig in i_groups:
        for jg in j_groups:
            for kg in k_groups:
                si, sj, sk = len(ig), len(jg), len(kg)
                if si == 2 and sj == 2 and sk == 2:
                    r = 7
                elif si == 3 and sj == 3 and sk == 3:
                    r = 23
                else:
                    r = si * sj * sk
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
        elif si == 3 and sj == 3 and sk == 3:
            for (u_pat, v_pat, w_pat) in laderman_terms:
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
    """Fast verification using numpy matrix multiplication."""
    U = np.asarray(U, dtype=float)
    V = np.asarray(V, dtype=float)
    W = np.asarray(W, dtype=float)
    r = U.shape[1]
    nm = n * m
    mp = m * p
    pn = p * n

    # Build the target multiplication tensor as a 3D array
    # T[i,j,k] = 1 for all i,j,k
    # The decomposition gives: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
    
    # Vectorized check: for each c, compute outer product contribution
    # M[i,j,k] = sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c]
    
    # Build M as (n, m, p) tensor
    M = np.zeros((n, m, p), dtype=float)
    for c in range(r):
        u = U[:, c].reshape(n, m)
        v = V[:, c].reshape(m, p)
        w = W[:, c].reshape(p, n)
        # M[i,j,k] += u[i,j] * v[j,k] * w[k,i]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    M[i, j, k] += u[i, j] * v[j, k] * w[k, i]
    
    if not np.allclose(M, 1.0, atol=1e-9):
        return False
    return True


def _try_reduce_rank_aggressive(U, V, W, n, m, p, max_rounds=200):
    """Aggressive rank reduction using multiple strategies:
    1. Single column elimination (original)
    2. Pairwise column combination detection
    3. Systematic elimination with different orderings
    """
    U = np.array(U, dtype=float).copy()
    V = np.array(V, dtype=float).copy()
    W = np.array(W, dtype=float).copy()
    r = U.shape[1]
    nm, mp, pn = n * m, m * p, p * n

    for round_num in range(max_rounds):
        if time.time() > _DEADLINE:
            break
        reduced = False

        # Strategy 1: Try removing each column if it's in span of others
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
                # Use SVD-based rank check
                rank_others = np.linalg.matrix_rank(M_others, tol=1e-10)
                rank_together = np.linalg.matrix_rank(np.column_stack([M_others, M_target]), tol=1e-10)
                if rank_together > rank_others:
                    ok = False
                    break
            if ok:
                U = np.delete(U, target, axis=1)
                V = np.delete(V, target, axis=1)
                W = np.delete(W, target, axis=1)
                r -= 1
                reduced = True
                break

        if reduced:
            continue

        # Strategy 2: Try to find if two columns can be combined
        # Check if column c1 = alpha * column c2 in all three matrices simultaneously
        for c1 in range(r):
            if time.time() > _DEADLINE:
                break
            for c2 in range(c1 + 1, r):
                if time.time() > _DEADLINE:
                    break
                # Check if U[:,c1] is a scalar multiple of U[:,c2]
                # and same for V and W
                u1, u2 = U[:, c1], U[:, c2]
                v1, v2 = V[:, c1], V[:, c2]
                w1, w2 = W[:, c1], W[:, c2]

                # Find alpha from U
                nonzero_u2 = np.where(np.abs(u2) > 1e-10)[0]
                if len(nonzero_u2) == 0:
                    continue
                alpha_candidates = set()
                for idx in nonzero_u2[:5]:  # Check up to 5 nonzeros
                    if np.abs(u2[idx]) > 1e-10:
                        alpha_candidates.add(u1[idx] / u2[idx])

                # Check if any alpha works for all three matrices
                for alpha in alpha_candidates:
                    if (np.allclose(u1, alpha * u2, atol=1e-10) and
                        np.allclose(v1, alpha * v2, atol=1e-10) and
                        np.allclose(w1, alpha * w2, atol=1e-10)):
                        # Remove c2 (keep c1, which is alpha*c2)
                        U = np.delete(U, c2, axis=1)
                        V = np.delete(V, c2, axis=1)
                        W = np.delete(W, c2, axis=1)
                        r -= 1
                        reduced = True
                        break
                if reduced:
                    break
            if reduced:
                break

        if not reduced:
            break

    return U.tolist(), V.tolist(), W.tolist()


def _try_reduce_rank_with_ordering(U, V, W, n, m, p, orderings=5):
    """Try rank reduction with different column orderings to find more dependencies."""
    best_U, best_V, best_W = U, V, W
    best_rank = len(U[0])

    for attempt in range(orderings):
        if time.time() > _DEADLINE:
            break
        rng = random.Random(42 + attempt)
        cols = list(range(len(U[0])))
        rng.shuffle(cols)

        U_sh = np.array(U)[:, cols]
        V_sh = np.array(V)[:, cols]
        W_sh = np.array(W)[:, cols]

        U_red, V_red, W_red = _try_reduce_rank_aggressive(
            U_sh.tolist(), V_sh.tolist(), W_sh.tolist(), n, m, p, max_rounds=100
        )

        if len(U_red[0]) < best_rank:
            best_U, best_V, best_W = U_red, V_red, W_red
            best_rank = len(U_red[0])

    return best_U, best_V, best_W


def _generate_partitions(n, m, p):
    """Generate candidate groupings for each dimension (groups of size 1, 2, or 3)."""
    def _partitions(dim):
        results = []
        def rec(start, cur):
            if start == dim:
                results.append(list(cur))
                return
            for size in range(1, min(4, dim - start + 1)):
                cur.append(list(range(start, start + size)))
                rec(start + size, cur)
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

    # Strategy 1: recursive Strassen over many partitions
    ip, jp, kp = _generate_partitions(n, m, p)
    candidates = []
    for ig in ip:
        for jg in jp:
            for kg in kp:
                candidates.append((ig, jg, kg))

    def expected_rank(ig, jg, kg):
        tot = 0
        for a in ig:
            for b in jg:
                for c in kg:
                    if len(a) == 2 and len(b) == 2 and len(c) == 2:
                        tot += 7
                    elif len(a) == 3 and len(b) == 3 and len(c) == 3:
                        tot += 23
                    else:
                        tot += len(a) * len(b) * len(c)
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

            # Aggressive rank reduction with multiple orderings
            U_red, V_red, W_red = _try_reduce_rank_with_ordering(
                cand["U"], cand["V"], cand["W"], n, m, p, orderings=8
            )
            if _verify_fast(U_red, V_red, W_red, n, m, p):
                new_rank = len(U_red[0])
                if new_rank < best_rank:
                    best = {"U": U_red, "V": V_red, "W": W_red}
                    best_rank = new_rank

    # Final verification; fall back to schoolbook if anything went wrong
    if not _verify_fast(best["U"], best["V"], best["W"], n, m, p):
        best = _schoolbook(n, m, p)

    return best


# EVOLVE-BLOCK-END
