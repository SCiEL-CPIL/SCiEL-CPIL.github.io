"""Rewritten program: stronger algorithm for large sets without subset averages."""
# EVOLVE-BLOCK-START
import time
import random
import math

def construct():
    start_time = time.time()
    deadline = 85.0  # Safety margin from 90s limit

    def elapsed():
        return time.time() - start_time

    # Bitset DP helper: builds dp[k] = bitmask of achievable sums using exactly k elements from s
    def build_dp(s):
        n = len(s)
        max_sum = sum(s)
        # Use a list of ints as bitsets
        dp = [1]  # dp[0] = 1 (bit 0 set)
        for x in s:
            new_dp = [0] * (len(dp) + 1)
            new_dp[0] = 1
            for k in range(1, len(dp) + 1):
                val = dp[k-1]  # don't use x for position k
                if k - 1 < len(dp):
                    val |= (dp[k-1] << x)  # use x for position k
                new_dp[k] = val
            dp = new_dp
        return dp

    def check_valid(x, s, dp_s):
        """Check if x can be added to set s (with its dp_s) without violating constraints.
        Constraint: no element a in s ∪ {x} is the mean of a subset of ≥2 OTHER elements in s ∪ {x} \\ {a}.
        We need to check:
        1. x is not the mean of any subset of ≥2 elements from s.
        2. For each a in s, a is not the mean of any subset of ≥2 elements from (s \\ {a}) ∪ {x}.
        """
        n = len(s)
        # Check 1: x not mean of subset of s of size >= 2
        for k in range(2, n + 1):
            target = k * x
            if target > 24576:
                break
            if (dp_s[k] >> target) & 1:
                return False

        # Check 2: for each a in s, a not mean of subset of (s\{a}) ∪ {x} of size >= 2
        # This means: does there exist a subset T ⊆ (s\{a}) ∪ {x} with |T| >= 2 and sum(T) = |T| * a?
        # T can include x or not.
        # Case A: T ⊆ s\{a}, |T| >= 2. This was already checked when a was added to s, so it's satisfied.
        # Case B: T = U ∪ {x} where U ⊆ s\{a}, |U| >= 1, and sum(U) + x = (|U|+1)*a.
        #   i.e., sum(U) = (|U|+1)*a - x.
        # So for each a in s and each j = |U| >= 1:
        #   Check if there's a subset of s\{a} of size j with sum = (j+1)*a - x.

        for i, a in enumerate(s):
            # Build dp for s \ {a}
            # To avoid rebuilding every time, we can precompute dp_without for each element
            # But for simplicity and since n is small (~48), we rebuild
            others = s[:i] + s[i+1:]
            m = len(others)
            if m < 1:
                continue
            dp_others = build_dp(others)
            for j in range(1, m + 1):
                target = (j + 1) * a - x
                if target < 0 or target > 24576:
                    continue
                if (dp_others[j] >> target) & 1:
                    return False

        return True

    def greedy_build(seed_elements, candidate_pool, max_size=48):
        """Build a valid set starting from seed_elements, greedily adding from candidate_pool."""
        s = list(seed_elements)
        s.sort()
        dp_s = build_dp(s)
        n = len(s)

        # Shuffle candidates
        random.shuffle(candidate_pool)

        for x in candidate_pool:
            if n >= max_size:
                break
            if x in s:
                continue
            if elapsed() > deadline - 1.0:
                break

            if check_valid(x, s, dp_s):
                s.append(x)
                s.sort()
                dp_s = build_dp(s)
                n += 1

        return s

    def local_search(s, max_iters=200):
        """Try to improve set s by removing and re-adding elements."""
        current = list(s)
        current.sort()
        all_nums = set(range(1, 513))

        for it in range(max_iters):
            if elapsed() > deadline - 1.0:
                break

            # Try to add a random element not in current
            candidates = list(all_nums - set(current))
            if not candidates:
                break
            random.shuffle(candidates)

            added = False
            for x in candidates[:20]:  # Limit checks per iteration
                if len(current) >= 48:
                    break
                dp_cur = build_dp(current)
                if check_valid(x, current, dp_cur):
                    current.append(x)
                    current.sort()
                    added = True
                    break

            if added:
                continue

            # Try a swap: remove one element, try to add another
            if len(current) > 3:
                idx = random.randint(0, len(current) - 1)
                removed = current.pop(idx)
                current.sort()

                # Try to add some candidates
                candidates2 = list(all_nums - set(current))
                random.shuffle(candidates2)
                swap_added = False
                for x in candidates2[:20]:
                    dp_cur = build_dp(current)
                    if check_valid(x, current, dp_cur):
                        current.append(x)
                        current.sort()
                        swap_added = True
                        break

                if not swap_added:
                    # Put it back
                    current.append(removed)
                    current.sort()

        return current

    # Phase 1: Initial greedy builds with different orderings
    best_set = []
    best_len = 0

    # Spread ordering
    spread = []
    lo, hi = 1, 512
    while lo <= hi:
        spread.append(lo)
        if lo != hi:
            spread.append(hi)
        lo += 1
        hi -= 1

    initial_orders = [
        spread,
        list(range(1, 513)),
        list(range(512, 0, -1)),
    ]

    for order in initial_orders:
        if elapsed() > deadline - 10.0:
            break
        pool = [x for x in order if x not in best_set]
        res = greedy_build([], pool, 48)
        if len(res) > best_len:
            best_set = res
            best_len = len(res)

    # Phase 2: Random restarts with warm starts
    restart_count = 0
    while elapsed() < deadline - 5.0:
        restart_count += 1

        # Warm start: take a random subset of best_set
        if best_len > 3 and random.random() < 0.7:
            keep_frac = random.uniform(0.6, 0.85)
            keep_n = max(1, int(best_len * keep_frac))
            seed_elems = random.sample(best_set, min(keep_n, len(best_set)))
        else:
            seed_elems = []

        pool = list(set(range(1, 513)) - set(seed_elems))
        random.shuffle(pool)

        res = greedy_build(seed_elems, pool, 48)

        # Local search on the result
        if len(res) > 0:
            res = local_search(res, max_iters=50)

        if len(res) > best_len:
            best_set = res
            best_len = len(res)
            if best_len >= 48:
                break

    # Final local search on best set
    if best_len > 0 and elapsed() < deadline - 3.0:
        best_set = local_search(best_set, max_iters=100)

    # Sort final result
    best_set.sort()
    return best_set

# EVOLVE-BLOCK-END
