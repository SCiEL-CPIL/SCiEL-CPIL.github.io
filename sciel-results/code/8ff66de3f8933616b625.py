"""Kissing number in dimension 11: construct nonzero integer vectors in R^11
whose minimum pairwise distance is at least the maximum vector norm.
Interface: construct() -> integer array of shape (k, d)."""

import itertools
import time
import numpy as np

D = 11
DEADLINE_S = 118.0  # hard wall-clock safety margin before the 120 s limit

def construct():
    t0 = time.perf_counter()
    best = None
    best_k = 0

    def now():
        return time.perf_counter()

    def elapsed():
        return now() - t0

    def is_deadline():
        return elapsed() > DEADLINE_S

    def update_best(arr):
        nonlocal best, best_k
        if arr is None:
            return
        k = arr.shape[0]
        if k > best_k:
            best = arr
            best_k = k

    # ------------------------------------------------------------------
    # 1) Fast deterministic baseline: the full "2-weight" shell of norm^2=2
    #    There are exactly C(11,2)*2^2 = 220 such vectors, and they satisfy
    #    min pairwise distance (sqrt(2) for shared coordinate) == max norm (sqrt(2)).
    # ------------------------------------------------------------------
    base = []
    for i, j in itertools.combinations(range(D), 2):
        for si in (1, -1):
            for sj in (1, -1):
                v = np.zeros(D, dtype=np.int64)
                v[i], v[j] = si, sj
                base.append(v)
    base_arr = np.array(base, dtype=np.int64)
    update_best(base_arr)

    # ------------------------------------------------------------------
    # 2) MODEL-AND-OPTIMIZE: try other fixed-norm^2 shells and a small
    #    mixed-norm extension, but only when we can finish within the budget.
    #    For a fixed-norm^2 = m shell, the condition "min distance >= max norm"
    #    is equivalent to a constant-weight code with max inner product
    #    <= floor(m/2) (since dist^2 = 2m - 2 u.v).
    #    Greedy (order by a random sign-flip heuristic) plus a couple of
    #    deterministic orderings is all we attempt; we stop well before the deadline.
    # ------------------------------------------------------------------

    def gen_weight_vectors(m, max_terms, hard_stop=30.0):
        """Yield integer vectors in R^D with L2^2 exactly = m and <= max_terms
        nonzero coordinates, values in {1,-1}, generated quickly.
        Used only for small m (3,4,5) where the set is tiny-to-moderate."""
        if m < 1:
            return []
        terms = np.zeros(D, dtype=np.int64)
        out = []
        # recursive enumeration of integer vectors with squared norm m
        # restrict support to <= max_terms for speed
        def rec(pos, sq_remaining, used_terms):
            if is_deadline() or elapsed() > hard_stop:
                return
            if sq_remaining == 0:
                out.append(terms.copy())
                return
            if pos == D:
                return
            if used_terms >= max_terms:
                # can only fill zeros here
                if sq_remaining == 0:
                    out.append(terms.copy())
                return
            for x in (-1, 0, 1):
                ns = sq_remaining - x * x
                if ns < 0:
                    continue
                # prune: we must be able to reach exactly 0 with at most (D-pos) terms of 1 or 0
                terms[pos] = x
                rec(pos + 1, ns, used_terms + (1 if x != 0 else 0))
            terms[pos] = 0

        rec(0, m, 0)
        return out

    # We'll try m = 3 and m = 4 (small shells) and a bounded greedy for each.
    def greedy_subset(vectors, m):
        """Given integer vectors all of L2^2 == m, pick a maximal (greedy)
        subset where pairwise distance^2 >= m, i.e., inner product <= floor(m/2)."""
        max_ip = m // 2  # floor
        V = np.asarray(vectors, dtype=np.int64)
        n = V.shape[0]
        if n == 0:
            return np.zeros((0, D), dtype=np.int64)
        # Compute Gram in one go
        G = V.astype(np.int64) @ V.astype(np.int64).T
        G = np.ascontiguousarray(G)
        # Greedy with vectorized max computation
        chosen = [0]
        chosen_mask = np.zeros(n, dtype=bool)
        chosen_mask[0] = True
        # Track max inner product with chosen for each candidate
        max_ip_with_chosen = G[0].copy()  # max ip with just vector 0
        max_ip_with_chosen[0] = -999999  # exclude self
        while not is_deadline():
            # Find candidates with max_ip_with_chosen <= max_ip
            valid = max_ip_with_chosen <= max_ip
            valid[chosen_mask] = False  # exclude already chosen
            if not np.any(valid):
                break
            # Among valid, pick the one with minimum max_ip_with_chosen
            # (tie-break: pick first)
            valid_indices = np.where(valid)[0]
            if len(valid_indices) == 0:
                break
            # Pick the one with smallest max_ip_with_chosen
            scores = max_ip_with_chosen[valid_indices]
            min_score = np.min(scores)
            best_c = valid_indices[np.argmin(scores)]
            chosen.append(int(best_c))
            chosen_mask[best_c] = True
            # Update max_ip_with_chosen: for each candidate, update with G[best_c]
            max_ip_with_chosen = np.maximum(max_ip_with_chosen, G[best_c])
            max_ip_with_chosen[chosen_mask] = -999999  # mark chosen
        if not chosen:
            return np.zeros((0, D), dtype=np.int64)
        return V[chosen]

    # --- m = 3 (weight-3) shell ---
    try:
        vecs3 = gen_weight_vectors(3, max_terms=3, hard_stop=10.0)
        if len(vecs3) > 0 and not is_deadline():
            sub = greedy_subset(vecs3, 3)
            update_best(sub)
    except Exception:
        pass

    # --- m = 4: weight-4 vectors (11 choose 4)*2^4 = 330*16=5280, plus single-2 vectors (11*2=22) = 5302 total
    # With the optimized greedy, we can handle this larger set.
    try:
        vecs4 = gen_weight_vectors(4, max_terms=4, hard_stop=15.0)
        if len(vecs4) > 0 and not is_deadline():
            sub4 = greedy_subset(vecs4, 4)
            update_best(sub4)
    except Exception:
        pass

    # ------------------------------------------------------------------
    # 3) Safety: if we somehow exceeded the deadline, just return the baseline.
    #    Otherwise return the best found.
    # ------------------------------------------------------------------
    if best is None or best.shape[0] == 0:
        return base_arr
    return best
