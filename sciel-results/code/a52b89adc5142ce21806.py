"""Search for a rank-minimal exact tensor decomposition of 3x5 * 5x6 multiplication."""
import time
import random
import numpy as np
from fractions import Fraction

DIMENSIONS = [3, 5, 6]

# EVOLVE-BLOCK-START
_DEADLINE = time.time() + 110.0  # hard wall-clock guard (evaluator kills at 150s)


def _strassen_2x2x2_terms():
    """Return the 7 Strassen terms for 2x2x2 matrix multiplication."""
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
    """Build a decomposition by splitting each dimension into groups."""
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
    """Fast verification using numpy matrix multiplication."""
    U = np.asarray(U, dtype=float)
    V = np.asarray(V, dtype=float)
    W = np.asarray(W, dtype=float)
    r = U.shape[1]
    nm = n * m
    mp = m * p
    pn = p * n

    for i in range(n):
        for k in range(p):
            for j in range(m):
                s = 0.0
                for c in range(r):
                    s += U[i * m + j, c] * V[j * p + k, c] * W[k * n + i, c]
                if abs(s - 1.0) > 1e-9:
                    return False
    return True


def _verify_full(U, V, W, n, m, p):
    """Full verification: check all (i,j,k) triplets."""
    U = np.asarray(U, dtype=float)
    V = np.asarray(V, dtype=float)
    W = np.asarray(W, dtype=float)
    r = U.shape[1]

    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for c in range(r):
                    s += U[i * m + j, c] * V[j * p + k, c] * W[k * n + i, c]
                if abs(s - 1.0) > 1e-9:
                    return False
    return True


def _try_reduce_rank_aggressive(U, V, W, n, m, p, max_rounds=200):
    """Aggressive rank reduction using multiple strategies."""
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
        for c1 in range(r):
            if time.time() > _DEADLINE:
                break
            for c2 in range(c1 + 1, r):
                if time.time() > _DEADLINE:
                    break
                u1, u2 = U[:, c1], U[:, c2]
                v1, v2 = V[:, c1], V[:, c2]
                w1, w2 = W[:, c1], W[:, c2]

                nonzero_u2 = np.where(np.abs(u2) > 1e-10)[0]
                if len(nonzero_u2) == 0:
                    continue
                alpha_candidates = set()
                for idx in nonzero_u2[:5]:
                    if np.abs(u2[idx]) > 1e-10:
                        alpha_candidates.add(u1[idx] / u2[idx])

                for alpha in alpha_candidates:
                    if (np.allclose(u1, alpha * u2, atol=1e-10) and
                            np.allclose(v1, alpha * v2, atol=1e-10) and
                            np.allclose(w1, alpha * w2, atol=1e-10)):
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
    """Try rank reduction with different column orderings."""
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
    """Generate candidate groupings for each dimension."""
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


def _rational_polish(U, V, W, n, m, p, tol=1e-6):
    """Try to find rational (integer/half-integer) values close to the current solution."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    r = U.shape[1]
    nm, mp, pn = n * m, m * p, p * n

    def _snap(val):
        if abs(val) < tol:
            return 0
        for denom in [1, 2, 3, 4, 6]:
            num = round(val * denom)
            if abs(val - num / denom) < tol:
                return Fraction(num, denom)
        return Fraction(val).limit_denominator(1000)

    U_rat = [[_snap(U[i, c]) for c in range(r)] for i in range(nm)]
    V_rat = [[_snap(V[i, c]) for c in range(r)] for i in range(mp)]
    W_rat = [[_snap(W[i, c]) for c in range(r)] for i in range(pn)]

    if _verify_fast(U_rat, V_rat, W_rat, n, m, p):
        return U_rat, V_rat, W_rat

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

                # Aggressive rank reduction with multiple orderings
                U_red, V_red, W_red = _try_reduce_rank_with_ordering(
                    cand["U"], cand["V"], cand["W"], n, m, p, orderings=8
                )
                if _verify_fast(U_red, V_red, W_red, n, m, p):
                    new_rank = len(U_red[0])
                    if new_rank < best_rank:
                        best = {"U": U_red, "V": V_red, "W": W_red}
                        best_rank = new_rank

    # Strategy 2: Try additional Strassen-based constructions with different groupings
    special_groupings = [
        ([[0], [1, 2]], [[0, 1], [2, 3, 4]], [[0, 1], [2, 3], [4, 5]]),
        ([[0], [1, 2]], [[0, 1], [2, 3, 4]], [[0, 1, 2], [3, 4, 5]]),
        ([[0, 1], [2]], [[0, 1], [2, 3, 4]], [[0, 1], [2, 3], [4, 5]]),
        ([[0], [1, 2]], [[0], [1, 2], [3, 4]], [[0, 1], [2, 3], [4, 5]]),
        ([[0], [1, 2]], [[0, 1], [2, 3], [4]], [[0, 1], [2, 3], [4, 5]]),
    ]

    for (ig, jg, kg) in special_groupings:
        if timed():
            break
        cand = _build_recursive_strassen(n, m, p, ig, jg, kg)
        if _verify_fast(cand["U"], cand["V"], cand["W"], n, m, p):
            r = len(cand["U"][0])
            if r < best_rank:
                best = cand
                best_rank = r

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
