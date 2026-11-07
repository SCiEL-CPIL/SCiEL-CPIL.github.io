import time
import random


def construct():
    deadline = time.time() + 85  # safety margin before the 90s kill

    # DP representation:
    #   dp[s] = bitmask over cardinalities: bit k set  <=>  some subset of
    #   exactly k chosen elements sums to s.
    #   Empty subset: dp[0] = 1 (bit 0).
    # A candidate x is INVALID iff, after tentatively folding it in, some sum
    # s is reachable by two different cardinalities, i.e. popcount(dp[s]) > 1.
    # Folding in x:  dp[s] |= (dp[s - x] << 1)   for s = max_sum+x down to x.
    #   (the shift moves the "subset of size k summing to s-x" to "size k+1
    #    summing to s"; iterating descending keeps x usable at most once.)

    def _popcount(v):
        return v.bit_count()

    def _try_add(dp, max_sum, x):
        # returns (new_dp, new_max_sum) if valid, else None
        new_dp = dict(dp)
        new_max = max_sum + x
        ok = True
        for s in range(new_max, x - 1, -1):
            prev = dp.get(s - x)
            if prev:
                cur = new_dp.get(s)
                nv = cur | (prev << 1)
                if _popcount(nv) > 1:
                    ok = False
                    break
                new_dp[s] = nv
        if not ok:
            return None
        return new_dp, new_max

    def _greedy(available, order):
        dp = {0: 1}
        max_sum = 0
        chosen = []
        for x in order:
            if x not in available:
                continue
            r = _try_add(dp, max_sum, x)
            if r is not None:
                dp, max_sum = r
                chosen.append(x)
                available.discard(x)
        return chosen

    def _multi_pass(available, base_order, passes):
        # Repeatedly sweep the order, adding anything that now fits.
        dp = {0: 1}
        max_sum = 0
        chosen = []
        for _ in range(passes):
            progress = False
            for x in base_order:
                if x not in available:
                    continue
                r = _try_add(dp, max_sum, x)
                if r is not None:
                    dp, max_sum = r
                    chosen.append(x)
                    available.discard(x)
                    progress = True
            if not progress:
                break
        return chosen

    def _improve(chosen, available):
        # Local search: try swaps and insertions that grow the set.
        if not chosen:
            return chosen
        chosen = sorted(chosen)
        avail = set(available)
        improved = True
        while improved:
            improved = False
            # insertion: add a single available element
            for x in sorted(avail, reverse=True):
                base = _greedy(set(chosen) | {x}, sorted(set(chosen) | {x}, reverse=True))
                if len(base) > len(chosen):
                    chosen = base
                    avail = set(range(1, 257)) - set(chosen)
                    improved = True
                    break
            if improved:
                continue
            # swap: remove one chosen, add two available
            for i in range(len(chosen)):
                removed = chosen[i]
                rest = [c for j, c in enumerate(chosen) if j != i]
                cand_pool = sorted(avail | {removed}, reverse=True)
                for a in cand_pool:
                    for b in cand_pool:
                        if a == b:
                            continue
                        base = _greedy(set(rest) | {a, b},
                                       sorted(set(rest) | {a, b}, reverse=True))
                        if len(base) > len(chosen):
                            chosen = base
                            avail = set(range(1, 257)) - set(chosen)
                            improved = True
                            break
                    if improved:
                        break
                if improved:
                    break
        return chosen

    best = []

    # 1) deterministic descending (the strong baseline)
    t0 = time.time()
    cand = _greedy(set(range(1, 257)), list(range(256, 0, -1)))
    if len(cand) > len(best):
        best = cand
    # deterministic multi-pass
    cand = _multi_pass(set(range(1, 257)), list(range(256, 0, -1)), 6)
    if len(cand) > len(best):
        best = cand

    # 2) randomized restarts: perturb the order to escape local optima
    rng = random.Random(12345)
    for _ in range(2000):
        if time.time() > deadline - 8:
            break
        # biased random order: mostly descending with random shuffles of blocks
        pool = list(range(256, 0, -1))
        for _ in range(3):
            i = rng.randrange(len(pool))
            j = rng.randrange(len(pool))
            pool[i], pool[j] = pool[j], pool[i]
        cand = _greedy(set(range(1, 257)), pool)
        if len(cand) > len(best):
            best = cand

    # 3) local improvement on the current best
    if time.time() <= deadline - 10:
        improved = _improve(list(best), set(range(1, 257)) - set(best))
        if len(improved) > len(best):
            best = improved

    return [int(x) for x in best]
