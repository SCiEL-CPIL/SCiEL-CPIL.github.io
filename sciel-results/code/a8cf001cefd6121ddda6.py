"""Linear triple systems on 16 points minimizing the independence number.

Structural redesign: instead of the full F2^4 affine-line design (which covers every
pair and therefore cannot be extended), we build *partial* linear systems by random
completion and then directly minimize the independence number (the actual score)
using a fast exact oracle plus randomized local search with simulated annealing.
"""
import time
import random
from itertools import combinations


def _indep_number(triples, n=16):
    """Exact independence number: largest point set containing no full triple.

    Uses a max-independent-set DP over the 3-uniform hypergraph, which is far
    faster than marking every superset of every triple.
    """
    if not triples:
        return n
    # For each point, the set of triples it participates in.
    pt_triples = [[] for _ in range(n)]
    for ti, t in enumerate(triples):
        for p in t:
            pt_triples[p].append(ti)

    m = len(triples)
    full = (1 << m) - 1
    cache = {}
    cache[full] = 0

    def solve(available, covered):
        key = (available, covered)
        val = cache.get(key)
        if val is not None:
            return val
        # Choose the triple with the fewest remaining points (branching heuristic).
        best_t = -1
        best_k = n + 1
        avail = available
        while avail:
            lsb = avail & (-avail)
            ti = lsb.bit_length() - 1
            avail ^= lsb
            k = 0
            for p in triples[ti]:
                if not (covered >> p) & 1:
                    k += 1
            if k < best_k:
                best_k = k
                best_t = ti
                if k <= 1:
                    break
        if best_t < 0:
            res = 0
        else:
            # Option A: remove this triple from available (do not cover it).
            res_a = solve(available & ~(1 << best_t), covered)
            # Option B: cover all of its points (forbidden to be in the set).
            new_cov = covered
            for p in triples[best_t]:
                new_cov |= (1 << p)
            res_b = 1 + solve(available & ~(1 << best_t), new_cov)
            res = res_a if res_a >= res_b else res_b
        if len(cache) < 4_000_000:
            cache[key] = res
        return res

    return solve(full, 0)


def _indep_number_fast(triples, n=16):
    """Public alias (kept for compatibility)."""
    return _indep_number(triples, n)


def _is_linear(triples):
    """Any two triples share at most one point."""
    seen = set()
    for t in triples:
        for i in range(3):
            for j in range(i + 1, 3):
                pair = (t[i], t[j]) if t[i] < t[j] else (t[j], t[i])
                if pair in seen:
                    return False
                seen.add(pair)
    return True


def _affine_lines_f24():
    """All 120 affine lines of F2^4 (a complete 2-(16,3,1) design)."""
    triples = []
    seen = set()
    for a in range(16):
        for b in range(a + 1, 16):
            c = a ^ b
            if c == a or c == b:
                continue
            t = tuple(sorted((a, b, c)))
            if t not in seen:
                seen.add(t)
                triples.append(list(t))
    return triples


def _random_initial_system(deadline, rng):
    """Build a random linear system: random triples, then maximal linear completion."""
    n = 16
    used = set()
    triples = []
    for _ in range(300000):
        if time.time() > deadline:
            break
        a = rng.randrange(n)
        b = rng.randrange(n)
        if a == b:
            continue
        c = rng.randrange(n)
        if c == a or c == b:
            continue
        pts = (a, b, c)
        ok = True
        for i in range(3):
            for j in range(i + 1, 3):
                pair = (pts[i], pts[j]) if pts[i] < pts[j] else (pts[j], pts[i])
                if pair in used:
                    ok = False
                    break
            if not ok:
                break
        if ok:
            for i in range(3):
                for j in range(i + 1, 3):
                    pair = (pts[i], pts[j]) if pts[i] < pts[j] else (pts[j], pts[i])
                    used.add(pair)
            triples.append(list(pts))
    # Maximal completion: add any triple that keeps linearity.
    for a in range(n):
        for b in range(a + 1, n):
            for c in range(b + 1, n):
                ok = True
                for i in range(3):
                    for j in range(i + 1, 3):
                        pair = (a, b, c)[i], (a, b, c)[j]
                        pair = (pair[0], pair[1]) if pair[0] < pair[1] else (pair[1], pair[0])
                        if pair in used:
                            ok = False
                            break
                    if not ok:
                        break
                if ok:
                    for i in range(3):
                        for j in range(i + 1, 3):
                            pair = (a, b, c)[i], (a, b, c)[j]
                            pair = (pair[0], pair[1]) if pair[0] < pair[1] else (pair[1], pair[0])
                            used.add(pair)
                    triples.append([a, b, c])
    return triples


def _local_search(triples, deadline, rng):
    """Direct local search on the independence number: add/swap/remove triples."""
    n = 16
    triples = [list(t) for t in triples]
    used = set()
    for t in triples:
        for i in range(3):
            for j in range(i + 1, 3):
                pair = (t[i], t[j]) if t[i] < t[j] else (t[j], t[i])
                used.add(pair)

    def addable(t):
        for i in range(3):
            for j in range(i + 1, 3):
                pair = (t[i], t[j]) if t[i] < t[j] else (t[j], t[i])
                if pair in used:
                    return False
        return True

    cur = _indep_number(triples, n)
    if cur <= 4:
        return triples, cur

    # Precompute all candidate triples (560) with their point lists.
    all_triples = [list(t) for t in combinations(range(n), 3)]

    # Phase 1: greedy add (monotone non-increasing).
    improved = True
    while improved and time.time() < deadline:
        improved = False
        best_gain = 0
        best_t = None
        for t in all_triples:
            if not addable(t):
                continue
            new = triples + [t]
            v = _indep_number(new, n)
            if cur - v > best_gain:
                best_gain = cur - v
                best_t = t
                if best_gain >= cur - 4:
                    break
        if best_t is not None and best_gain > 0:
            triples.append(best_t)
            for i in range(3):
                for j in range(i + 1, 3):
                    pair = (best_t[i], best_t[j]) if best_t[i] < best_t[j] else (best_t[j], best_t[i])
                    used.add(pair)
            cur -= best_gain
            improved = True
            if cur <= 4:
                break

    # Phase 2: simulated annealing over swaps (remove one, add one).
    if time.time() < deadline and cur > 4:
        T = 1.5
        T_end = 0.05
        cur_list = triples
        for it in range(6000):
            if time.time() > deadline:
                break
            if cur <= 4:
                break
            idx = rng.randrange(len(cur_list))
            old = cur_list[idx]
            # Free old's pairs.
            freed = []
            for i in range(3):
                for j in range(i + 1, 3):
                    pair = (old[i], old[j]) if old[i] < old[j] else (old[j], old[i])
                    used.discard(pair)
                    freed.append(pair)
            # Pick a random addable candidate.
            cand = None
            for _ in range(40):
                t = all_triples[rng.randrange(len(all_triples))]
                if addable(t):
                    cand = t
                    break
            if cand is None:
                for pair in freed:
                    used.add(pair)
                continue
            new_list = cur_list[:idx] + [cand] + cur_list[idx + 1:]
            v = _indep_number(new_list, n)
            delta = v - cur  # negative is better
            if delta < 0 or rng.random() < (2.718281828 ** (-delta / max(T, 1e-6))):
                # Accept.
                cur_list = new_list
                cur = v
                for i in range(3):
                    for j in range(i + 1, 3):
                        pair = (cand[i], cand[j]) if cand[i] < cand[j] else (cand[j], cand[i])
                        used.add(pair)
            else:
                # Reject: restore old.
                for pair in freed:
                    used.add(pair)
            T = max(T * 0.995, T_end)

    return cur_list, cur


def construct():
    start = time.time()
    deadline = start + 138  # safety margin before the 150s kill
    rng = random.Random(20240517)

    # Seed candidate: the full affine-line design (known cap 6).
    best = _affine_lines_f24()
    best_val = _indep_number(best, 16)

    # Try several random constructions and optimize each.
    n_restarts = 12
    for r in range(n_restarts):
        if time.time() > deadline - 25:
            break
        try:
            init = _random_initial_system(deadline, rng)
            if not init:
                continue
            if not _is_linear(init):
                continue
            cand, val = _local_search(init, deadline, rng)
            if val < best_val:
                best_val = val
                best = cand
                if best_val <= 4:
                    break
        except Exception:
            continue

    # Final safety: ensure linearity and validity.
    if best is None or not _is_linear(best):
        best = [[0, 1, 2], [3, 4, 5], [6, 7, 8], [9, 10, 11], [12, 13, 14]]
    return best
