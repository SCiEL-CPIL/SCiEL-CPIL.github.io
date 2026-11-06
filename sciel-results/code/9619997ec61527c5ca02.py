import time
import sys

def construct():
    start = time.time()
    DEADLINE = start + 75.0  # safety margin under 90s
    sys.setrecursionlimit(100000)

    def time_up():
        return time.time() > DEADLINE

    MAX_VAL = 256
    # We need: strictly increasing a_1 < a_2 < ... < a_n, each in [1,256]
    # All nonempty consecutive block sums distinct.
    # Consecutive block sum a_i..a_j = prefix[j] - prefix[i-1]
    # So we need all prefix[j]-prefix[i] for 0<=i<j<=n to be distinct (Sidon-like)
    # AND prefix[n] <= sum of n values each <=256... actually prefix values can be large.
    # Wait: a_i in [1,256], strictly increasing. prefix[n] = sum of a_i <= n*256.
    # Number of block sums = n(n+1)/2. All must be distinct positive integers.
    # Max possible block sum = prefix[n] <= n*256.
    # So need n(n+1)/2 <= n*256 => (n+1)/2 <= 256 => n <= 511. But also a_i strictly increasing in [1,256]
    # so n <= 256. And prefix[n] = sum a_i. With a_i strictly increasing in [1,256],
    # min prefix[n] = 1+2+...+n = n(n+1)/2. Max = (256-n+1)*n/2... 
    # Actually the constraint is a_i in [1,256] strictly increasing, so n <= 256.
    # Block sums range from min a_i=1 to max prefix[n].
    # Number of distinct block sums needed: n(n+1)/2.
    # Max block sum = prefix[n] = sum a_i.
    # So need n(n+1)/2 <= sum a_i.
    # sum a_i >= n(n+1)/2 (min when a_i = 1,2,...,n). So this is always satisfiable in that direction.
    # The real constraint is that all n(n+1)/2 differences prefix[j]-prefix[i] are distinct.
    # This is exactly a Sidon set condition on the prefix sums (all pairwise differences distinct).
    # Prefix sums P_0=0 < P_1 < ... < P_n, all pairwise differences P_j-P_i (i<j) distinct.
    # That's n(n+1)/2 differences, all in [1, P_n]. Need P_n >= n(n+1)/2.
    # P_n = sum a_i, a_i strictly increasing in [1,256].
    # To maximize n, we want P_n as large as possible relative to n, i.e., large a_i.
    # Max P_n = 256+255+...+(256-n+1) = n(512-n+1)/2 = n(513-n)/2.
    # Need n(513-n)/2 >= n(n+1)/2 => 513-n >= n+1 => 512 >= 2n => n <= 256.
    # So n=256 is theoretically possible IF a Sidon set exists with the right structure.
    # But a_i must be strictly increasing AND the prefix sums must form a Sidon set.

    # Bitset for used differences. Max difference <= P_n <= 256*256 = 65536.
    # Use a Python int as bitset: bit d set means difference d is used.
    # Collision check: for new P, check if any (P - P_i) is already used.

    best_seq = [1]
    best_len = 1

    def update_best(a_list):
        nonlocal best_seq, best_len
        if len(a_list) > best_len:
            best_len = len(a_list)
            best_seq = a_list[:]

    # Verify a candidate sequence
    def is_valid(a):
        n = len(a)
        if n == 0:
            return True
        # strictly increasing
        for i in range(n-1):
            if a[i+1] <= a[i]:
                return False
        for v in a:
            if v < 1 or v > 256:
                return False
        # all consecutive block sums distinct
        prefix = [0]
        for v in a:
            prefix.append(prefix[-1] + v)
        sums = set()
        for i in range(n+1):
            for j in range(i+1, n+1):
                s = prefix[j] - prefix[i]
                if s in sums:
                    return False
                sums.add(s)
        return True

    # Backtracking solver with bitset
    # State: prefix list, bitset of used differences, last_gap (last a_i)
    # We build a_1, a_2, ..., a_n. a_i > a_{i-1}, a_i <= 256.
    # When we add a_i, new prefix P_i = P_{i-1} + a_i.
    # New differences: P_i - P_j for j=0..i-1.
    # Check none of these are in bitset, and they're distinct among themselves (they are, since P_j distinct).
    # Actually P_i - P_j for different j are all different (since P_j different). So just check against bitset.

    def solve(max_n, start_a_min=1):
        """Try to find a valid sequence of length max_n. Returns list or None."""
        prefix = [0]
        used_bits = 0  # bit d set if difference d is used
        last_a = 0
        a_list = []

        # Precompute: max possible prefix sum = sum of largest max_n values in [1,256]
        # = 256 + 255 + ... + (256-max_n+1)
        max_prefix = max_n * (513 - max_n) // 2

        def backtrack(idx, prefix_sum, last_a_val, used):
            nonlocal time_up
            if time_up():
                return None
            if idx == max_n:
                return a_list[:]
            # Determine range for a_idx
            # a_idx > last_a_val, a_idx <= 256
            # Also prefix_sum + a_idx must allow remaining (max_n - idx - 1) elements
            # each > a_idx, <= 256.
            # Min remaining sum: (a_idx+1) + (a_idx+2) + ... (a_idx + max_n-idx-1)
            rem = max_n - idx - 1
            # For feasibility: need enough room. a_idx + rem <= 256 (since a_{idx+1} >= a_idx+1, ..., a_n >= a_idx+rem)
            # So a_idx <= 256 - rem
            a_max = min(256, 256 - rem)
            if a_max <= last_a_val:
                return None
            # Try values. Strategy: try larger values first to maximize prefix sum (helps Sidon)
            # but also try smaller to keep flexibility. Let's try from a_max down to last_a_val+1.
            for a_val in range(a_max, last_a_val, -1):
                if time_up():
                    return None
                new_prefix = prefix_sum + a_val
                # Check new differences: new_prefix - prefix[j] for j in 0..idx
                ok = True
                for j in range(idx + 1):
                    d = new_prefix - prefix[j]
                    if d <= 0:
                        ok = False
                        break
                    if (used >> d) & 1:
                        ok = False
                        break
                if not ok:
                    continue
                # Also need: the new differences don't collide with each other
                # They are new_prefix - prefix[0], ..., new_prefix - prefix[idx]
                # Since prefix[j] are all different, these are all different. Good.
                # Update
                prefix.append(new_prefix)
                a_list.append(a_val)
                new_used = used
                for j in range(idx + 1):
                    d = new_prefix - prefix[j]
                    new_used |= (1 << d)
                result = backtrack(idx + 1, new_prefix, a_val, new_used)
                if result is not None:
                    return result
                prefix.pop()
                a_list.pop()
            return None

        return backtrack(0, 0, 0, 0)

    # Binary search on length, with multiple starting strategies
    # First, try greedy to get a baseline
    def greedy_build():
        prefix = [0]
        used = 0
        last_a = 0
        a_list = []
        while True:
            found = False
            for a_val in range(1, 257):
                if a_val <= last_a:
                    continue
                new_prefix = prefix[-1] + a_val
                ok = True
                for j in range(len(prefix)):
                    d = new_prefix - prefix[j]
                    if (used >> d) & 1:
                        ok = False
                        break
                if ok:
                    prefix.append(new_prefix)
                    a_list.append(a_val)
                    for j in range(len(prefix) - 1):
                        d = new_prefix - prefix[j]
                        used |= (1 << d)
                    last_a = a_val
                    found = True
                    break
            if not found:
                break
        return a_list

    g_seq = greedy_build()
    if is_valid(g_seq):
        update_best(g_seq)

    # Now try to find longer sequences using backtracking
    # Try lengths from best_len+1 upward, with time budget
    # Use multiple strategies: different orderings of a_val

    def solve_with_strategy(max_n, strategy="desc"):
        """strategy: 'desc' = try large a first, 'asc' = try small a first, 'mid' = middle out"""
        prefix = [0]
        a_list = []

        def backtrack(idx, prefix_sum, last_a_val, used):
            if time_up():
                return None
            if idx == max_n:
                return a_list[:]
            rem = max_n - idx - 1
            a_max = min(256, 256 - rem)
            if a_max <= last_a_val:
                return None
            a_min = last_a_val + 1
            if strategy == "desc":
                candidates = range(a_max, a_min - 1, -1)
            elif strategy == "asc":
                candidates = range(a_min, a_max + 1)
            else:  # mid
                mid = (a_min + a_max) // 2
                candidates = sorted(range(a_min, a_max + 1), key=lambda x: abs(x - mid))
            for a_val in candidates:
                if time_up():
                    return None
                new_prefix = prefix_sum + a_val
                ok = True
                for j in range(idx + 1):
                    d = new_prefix - prefix[j]
                    if (used >> d) & 1:
                        ok = False
                        break
                if not ok:
                    continue
                prefix.append(new_prefix)
                a_list.append(a_val)
                new_used = used
                for j in range(idx + 1):
                    d = new_prefix - prefix[j]
                    new_used |= (1 << d)
                result = backtrack(idx + 1, new_prefix, a_val, new_used)
                if result is not None:
                    return result
                prefix.pop()
                a_list.pop()
            return None

        return backtrack(0, 0, 0, 0)

    # Try increasing lengths
    target_len = best_len + 1
    while target_len <= 256 and not time_up():
        found = False
        for strat in ["desc", "asc", "mid"]:
            if time_up():
                break
            result = solve_with_strategy(target_len, strat)
            if result is not None and is_valid(result):
                update_best(result)
                found = True
                break
        if found:
            target_len += 1
        else:
            # Try a few more strategies or break
            break

    if best_seq is None or len(best_seq) == 0:
        best_seq = [1]

    return best_seq
