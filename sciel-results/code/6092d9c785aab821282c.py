"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import math


def construct():
    start_time = time.time()
    DEADLINE = 135.0  # safety margin before the 150s kill

    def remaining():
        return DEADLINE - (time.time() - start_time)

    def instance(size, seed, limit=10000):
        rng = random.Random(seed)
        values = set()
        while len(values) < size:
            x = rng.randint(-limit, limit)
            if x != 0:
                values.add(x)
        return sorted(values)

    def is_sum_free(S):
        Ss = set(S)
        for a in S:
            for b in S:
                if a + b in Ss:
                    return False
        return True

    def make_greedy(arr, key):
        """Greedy sum-free subset. Returns list of chosen elements."""
        n = len(arr)
        chosen = []
        chosen_set = set()
        for x in sorted(arr, key=key):
            ok = True
            # Forward: x + y in chosen for some y
            for y in chosen:
                if x + y in chosen_set:
                    ok = False
                    break
            if not ok:
                continue
            # Backward: x = a + b for some a, b in chosen (a==b allowed)
            if len(chosen) >= 2:
                ch = sorted(chosen)
                m = len(ch)
                found = False
                for i in range(m):
                    a = ch[i]
                    # need b = x - a, with b >= a (to cover a==b and avoid double)
                    b = x - a
                    if b < a:
                        break
                    # binary search b in ch[i:]
                    lo, hi = i, m
                    while lo < hi:
                        mid = (lo + hi) // 2
                        if ch[mid] < b:
                            lo = mid + 1
                        else:
                            hi = mid
                    if lo < m and ch[lo] == b:
                        found = True
                        break
                if found:
                    continue
            chosen.append(x)
            chosen_set.add(x)
        return chosen

    def local_improve(subset, arr, max_iter=50):
        """Try to add back excluded elements one at a time."""
        S = list(subset)
        S_set = set(S)
        excluded = [x for x in arr if x not in S_set]
        improved = False
        for _ in range(max_iter):
            added = False
            for x in excluded:
                # Check if x can be added
                ok = True
                for y in S:
                    if x + y in S_set:
                        ok = False
                        break
                if ok:
                    # Check backward
                    if len(S) >= 2:
                        ch = sorted(S)
                        m = len(ch)
                        found = False
                        for i in range(m):
                            a = ch[i]
                            b = x - a
                            if b < a:
                                break
                            lo, hi = i, m
                            while lo < hi:
                                mid = (lo + hi) // 2
                                if ch[mid] < b:
                                    lo = mid + 1
                                else:
                                    hi = mid
                            if lo < m and ch[lo] == b:
                                found = True
                                break
                        if found:
                            ok = False
                if ok:
                    S.append(x)
                    S_set.add(x)
                    excluded.remove(x)
                    added = True
                    improved = True
                    break
            if not added:
                break
        return S

    def solve_instance(arr):
        """Try multiple orderings, return the largest valid sum-free subset."""
        n = len(arr)
        best = []
        # Order 1: by absolute value ascending (small magnitudes first)
        cand = make_greedy(arr, lambda x: (abs(x), x))
        if len(cand) > len(best):
            best = cand
        # Order 2: by value ascending
        cand = make_greedy(arr, lambda x: x)
        if len(cand) > len(best):
            best = cand
        # Order 3: by value descending
        cand = make_greedy(arr, lambda x: -x)
        if len(cand) > len(best):
            best = cand
        # Order 4: by absolute value descending (large magnitudes first)
        cand = make_greedy(arr, lambda x: (-abs(x), x))
        if len(cand) > len(best):
            best = cand
        # Order 5: by absolute value descending, tie-break by value
        cand = make_greedy(arr, lambda x: (-abs(x), -x))
        if len(cand) > len(best):
            best = cand
        # Random restarts while time remains
        rng = random.Random(12345)
        guard = 0
        while remaining() > 2.0 and guard < 200:
            guard += 1
            order = list(arr)
            rng.shuffle(order)
            cand = make_greedy(order, lambda x: x)  # identity on pre-shuffled list
            if len(cand) > len(best):
                best = cand
        # Try local improvement on best
        improved = local_improve(best, arr, max_iter=30)
        if len(improved) > len(best):
            best = improved
        # Final safety: ensure validity
        if not is_sum_free(best):
            # rebuild deterministically
            best = make_greedy(arr, lambda x: (abs(x), x))
        return best

    results = []
    for size, seed in ((60, 801), (120, 802), (240, 803)):
        arr = instance(size, seed)
        results.append(solve_instance(arr))
    return results
# EVOLVE-BLOCK-END
