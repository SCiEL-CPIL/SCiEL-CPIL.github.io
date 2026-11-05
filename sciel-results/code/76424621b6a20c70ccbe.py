"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import math


def construct():
    start_time = time.time()
    DEADLINE = 145.0  # safety margin before the 150s kill

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

    def local_improve(subset, arr, max_iter=100):
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
                # Forward conflicts: x + y in S
                for y in S:
                    if x + y in S_set:
                        ok = False
                        break
                if ok:
                    # Backward conflicts: x = a + b for a,b in S
                    # This is equivalent to checking if any pair in S sums to x
                    # We can iterate through S and check if x-y is in S
                    # But we must be careful with a==b
                    for a in S:
                        b = x - a
                        if b in S_set:
                            # If a == b, we need to ensure there are at least two of them? 
                            # No, the set contains distinct integers. 
                            # If a == b, then x = 2a. 
                            # The condition is a+b=c. If a=b, then 2a=c.
                            # Since elements are distinct, a and b are the same value.
                            # So if x = 2a and a is in S, is that a violation?
                            # Yes, if a is in S, then a+a=2a. If 2a is in S, violation.
                            # Here we are adding x. If x=2a, and a is in S, then a+a=x.
                            # So x cannot be added if x=2a for any a in S.
                            # My check `if b in S_set` covers this. 
                            # If a=b, then b=a. So if a in S_set, it returns True.
                            # This is correct.
                            ok = False
                            break
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
        """Try multiple orderings and structural candidates, return the largest valid sum-free subset."""
        n = len(arr)
        best = []
        
        # 1. Structural Candidates
        # Positive numbers: keep the largest half to avoid sums
        pos = [x for x in arr if x > 0]
        if pos:
            # Sort descending, take first half
            pos_sorted = sorted(pos, reverse=True)
            cand = pos_sorted[:len(pos_sorted)//2 + 1]
            if is_sum_free(cand):
                if len(cand) > len(best):
                    best = cand
            else:
                # Fallback to greedy on positives
                cand = make_greedy(pos, lambda x: -x)
                if len(cand) > len(best):
                    best = cand
        
        # Negative numbers: keep the smallest half
        neg = [x for x in arr if x < 0]
        if neg:
            neg_sorted = sorted(neg)
            cand = neg_sorted[:len(neg_sorted)//2 + 1]
            if is_sum_free(cand):
                if len(cand) > len(best):
                    best = cand
            else:
                cand = make_greedy(neg, lambda x: x)
                if len(cand) > len(best):
                    best = cand
        
        # Odd numbers
        odds = [x for x in arr if x % 2 != 0]
        if odds:
            # Odds are always sum-free internally? 
            # odd + odd = even. So if the set contains only odds, no sum of two odds is odd.
            # Thus, any subset of odds is sum-free.
            cand = odds
            if len(cand) > len(best):
                best = cand

        # 2. Greedy Candidates
        # Order 1: by absolute value ascending
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
        # Order 4: by absolute value descending
        cand = make_greedy(arr, lambda x: (-abs(x), x))
        if len(cand) > len(best):
            best = cand

        # 3. Local Improvement
        # Apply local improvement to the current best
        improved = local_improve(best, arr, max_iter=50)
        if len(improved) > len(best):
            best = improved

        # 4. Random Restarts
        rng = random.Random(12345)
        guard = 0
        while remaining() > 2.0 and guard < 100:
            guard += 1
            order = list(arr)
            rng.shuffle(order)
            cand = make_greedy(order, lambda x: x)
            # Quick local improve on random cand
            cand = local_improve(cand, arr, max_iter=10)
            if len(cand) > len(best):
                best = cand
                # If we found a better one, maybe refine it more
                improved = local_improve(best, arr, max_iter=20)
                if len(improved) > len(best):
                    best = improved

        # Final safety
        if not is_sum_free(best):
            best = make_greedy(arr, lambda x: (abs(x), x))
        return best

    results = []
    for size, seed in ((60, 801), (120, 802), (240, 803)):
        arr = instance(size, seed)
        results.append(solve_instance(arr))
    return results
# EVOLVE-BLOCK-END
