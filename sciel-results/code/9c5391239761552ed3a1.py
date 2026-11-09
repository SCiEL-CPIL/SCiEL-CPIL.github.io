"""Improved SEARCH approach for unique-triple-sum sets with systematic base-representation greedy."""
import itertools
import random
import time

def construct():
    t_start = time.time()
    DEADLINE = t_start + 85.0   # safety margin under the 90s kill

    N = 4096

    # ---------- helper: validate a set ----------
    def is_valid(vals):
        m = len(vals)
        seen = set()
        for i in range(m):
            a = vals[i]
            for j in range(i, m):
                b = vals[j]
                for k in range(j, m):
                    c = vals[k]
                    s = a + b + c
                    if s in seen:
                        return False
                    seen.add(s)
        return True

    # ---------- fast triple-sum conflict check for adding v to chosen ----------
    def check_add(v, chosen, sums):
        """Check if v can be added to chosen without creating duplicate triple sums.
        Returns (True, new_sums) if valid, (False, None) otherwise."""
        new_sums = set()
        # (v,v,v)
        new_sums.add(3 * v)
        for u in chosen:
            new_sums.add(2 * v + u)      # (v,v,u)
            new_sums.add(v + 2 * u)      # (v,u,u)
        # (v,u,w) with u,w in chosen, u<=w
        for ui in range(len(chosen)):
            u = chosen[ui]
            for wi in range(ui, len(chosen)):
                w = chosen[wi]
                new_sums.add(v + u + w)
        if new_sums & sums:
            return False, None
        return True, new_sums

    # ---------- systematic greedy with base representation ----------
    def systematic_greedy(limit, start_set=None):
        """Greedy construction using a base-representation-inspired ordering."""
        if start_set:
            chosen = list(start_set)
            sums = set()
            for x in itertools.combinations_with_replacement(chosen, 3):
                sums.add(sum(x))
        else:
            chosen = []
            sums = set()
        
        # Try candidates in ascending order, but skip those that conflict
        # Use a priority: prefer smaller numbers first for better packing
        for v in range(1, N + 1):
            if time.time() > limit:
                break
            if v in chosen:
                continue
            ok, ns = check_add(v, chosen, sums)
            if ok:
                sums |= ns
                chosen.append(v)
        
        return chosen

    # ---------- local improvement: try to add missing numbers ----------
    def local_improve(chosen, limit):
        """Try to add more numbers to an existing valid set."""
        sums = set()
        for x in itertools.combinations_with_replacement(chosen, 3):
            sums.add(sum(x))
        
        # Multiple passes to find all possible additions
        improved = True
        while improved and time.time() < limit:
            improved = False
            for v in range(1, N + 1):
                if time.time() > limit:
                    break
                if v in chosen:
                    continue
                ok, ns = check_add(v, chosen, sums)
                if ok:
                    sums |= ns
                    chosen.append(v)
                    improved = True
        return chosen

    best = []

    # ---------- 1) Start with power-of-2 core ----------
    pow2 = [2 ** i for i in range(0, 13) if 2 ** i <= 4096]
    if is_valid(pow2):
        best = pow2
    else:
        # Find the largest valid prefix
        for i in range(len(pow2), 0, -1):
            if is_valid(pow2[:i]):
                best = pow2[:i]
                break

    # ---------- 2) Systematic greedy from power-2 core ----------
    if time.time() < DEADLINE - 10:
        ext = systematic_greedy(DEADLINE - 5, best)
        if is_valid(ext) and len(ext) > len(best):
            best = ext

    # ---------- 3) Pure systematic greedy from scratch ----------
    if time.time() < DEADLINE - 10:
        pure = systematic_greedy(DEADLINE - 5, None)
        if is_valid(pure) and len(pure) > len(best):
            best = pure

    # ---------- 4) Randomized greedy with varied step sizes ----------
    random.seed(12345)
    for trial in range(15):
        if time.time() > DEADLINE - 5:
            break
        # Vary the density of the candidate pool
        density = random.uniform(0.3, 0.8)
        pool = [v for v in range(1, N + 1) if random.random() < density]
        random.shuffle(pool)
        
        chosen = []
        sums = set()
        for v in pool:
            if time.time() > DEADLINE - 5:
                break
            if v in chosen:
                continue
            ok, ns = check_add(v, chosen, sums)
            if ok:
                sums |= ns
                chosen.append(v)
        
        if is_valid(chosen) and len(chosen) > len(best):
            best = chosen

    # ---------- 5) Local improvement on best ----------
    if time.time() < DEADLINE - 3:
        improved = local_improve(list(best), DEADLINE - 1)
        if is_valid(improved) and len(improved) > len(best):
            best = improved

    # ---------- 6) Final polish: one more local improvement pass ----------
    if time.time() < DEADLINE - 1:
        final = local_improve(list(best), DEADLINE - 0.5)
        if is_valid(final) and len(final) > len(best):
            best = final

    # ---------- 7) Sanity: ensure within range, distinct, positive ----------
    best = [v for v in best if 1 <= v <= N]
    best = sorted(set(best))

    return best
