# EVOLVE-BLOCK-START
import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # 5s safety margin before 90s kill
    N = 512
    MAX_SIZE = 48

    # Precompute: for each possible target value t (1..512),
    # and for each possible subset size k, store the set of pairs (i,j)
    # such that a_i + a_j = t (for k=2). For larger k, we build up.
    #
    # Actually, the constraint is: no element a in the set is the mean of
    # any subset of >=2 OTHER elements.
    # i.e., for no a in S does there exist a subset T of S\{a} with |T|>=2
    # such that sum(T) = |T| * a.
    #
    # Key reformulation: for each candidate element a and each possible
    # subset size k >= 2, the "forbidden sum" is k*a. We need to ensure
    # that no subset of the other elements sums to exactly k*a.
    #
    # Efficient check strategy:
    # For a given set S, to check if it's valid:
    #   For each a in S:
    #     For each k from 2 to len(S)-1:
    #       Check if any subset of S\{a} of size k sums to k*a
    #
    # This is still expensive. Let's use a different approach:
    # Precompute for the FULL range [1,512] which "forbidden configurations" exist.
    #
    # Alternative: Use the fact that the constraint is local in a sense.
    # If we build the set incrementally and maintain for each element a
    # the set of achievable (size, sum) pairs from the other elements,
    # we can check quickly.
    #
    # Let me try a different structural approach:
    # Use simulated annealing with a fast penalty-based objective.

    def is_valid_set(S):
        """Check if set S satisfies the constraint."""
        n = len(S)
        if n < 3:
            return True
        S_set = set(S)
        for a in S:
            others = [x for x in S if x != a]
            m = len(others)
            if m < 2:
                continue
            # For each k from 2 to m, check if any subset of others of size k
            # sums to k*a
            # Use DP: dp[k] = set of achievable sums with k elements
            # But this is expensive. Use bitset approach.
            max_sum = k_max = m
            # dp[k] is a bitset where bit s is set if sum s is achievable with k elements
            # We only care about sums up to m*a (max possible target)
            max_target = m * a
            if max_target > 512 * m:  # can't happen since elements <= 512
                max_target = 512 * m

            # Use Python integers as bitsets
            dp = [0] * (m + 1)
            dp[0] = 1  # bit 0 set: empty subset has sum 0
            for x in others:
                for k in range(m, 0, -1):
                    dp[k] |= (dp[k-1] << x)
                # Mask to keep only relevant bits
                mask = (1 << (max_target + 1)) - 1
                for k in range(m + 1):
                    dp[k] &= mask

            for k in range(2, m + 1):
                target = k * a
                if target <= max_target:
                    if (dp[k] >> target) & 1:
                        return False
        return True

    def penalty_score(S):
        """Return number of violations (0 = valid). Lower is better."""
        n = len(S)
        if n < 3:
            return 0
        violations = 0
        S_set = set(S)
        for a in S:
            others = [x for x in S if x != a]
            m = len(others)
            if m < 2:
                continue
            max_target = m * a
            if max_target > 512 * m:
                max_target = 512 * m

            dp = [0] * (m + 1)
            dp[0] = 1
            for x in others:
                for k in range(m, 0, -1):
                    dp[k] |= (dp[k-1] << x)
                mask = (1 << (max_target + 1)) - 1
                for k in range(m + 1):
                    dp[k] &= mask

            for k in range(2, m + 1):
                target = k * a
                if target <= max_target:
                    if (dp[k] >> target) & 1:
                        violations += 1
        return violations

    # Strategy: Greedy construction with multiple orderings, then local search
    # to maximize size.

    best_set = []

    # Fast greedy: try to add elements in various orders
    def greedy_construct(order):
        selected = []
        selected_set = set()
        for x in order:
            if len(selected) >= MAX_SIZE:
                break
            if x in selected_set:
                continue
            # Quick check: would adding x create a 3-AP?
            # x = (a+b)/2 for some a,b in selected => 2x = a+b
            # a = (x+b)/2 for some b in selected, a!=x => 2a = x+b
            # b = (a+x)/2 for some a in selected, b!=x => 2b = a+x
            conflict = False
            for a in selected:
                # x is mean of a and b: b = 2x - a
                b = 2 * x - a
                if b in selected_set and b != a:
                    conflict = True
                    break
                # a is mean of x and b: b = 2a - x
                b = 2 * a - x
                if b in selected_set and b != a and b != x:
                    conflict = True
                    break
            if conflict:
                continue
            # Full check
            candidate = selected + [x]
            if is_valid_set(candidate):
                selected.append(x)
                selected_set.add(x)
        return selected

    # Try multiple greedy orderings
    orders = [
        list(range(1, N + 1)),
        list(range(N, 0, -1)),
        list(range(1, N + 1, 2)),  # odds only
        list(range(2, N + 1, 2)),  # evens only
        list(range(1, N + 1, 3)),
        list(range(2, N + 1, 3)),
    ]

    for order in orders:
        if time.time() > deadline:
            break
        s = greedy_construct(order)
        if len(s) > len(best_set):
            best_set = s[:]

    # Randomized greedy with many seeds
    for seed in range(200):
        if time.time() > deadline:
            break
        rng = random.Random(seed)
        order = list(range(1, N + 1))
        rng.shuffle(order)
        s = greedy_construct(order)
        if len(s) > len(best_set):
            best_set = s[:]

    # Local search: try to add elements to best_set
    if time.time() < deadline - 10:
        current = best_set[:]
        current_set = set(current)
        improved = True
        while improved and time.time() < deadline - 10:
            improved = False
            for x in range(1, N + 1):
                if x in current_set:
                    continue
                if len(current) >= MAX_SIZE:
                    break
                candidate = current + [x]
                if is_valid_set(candidate):
                    current.append(x)
                    current_set.add(x)
                    improved = True
                    if len(current) > len(best_set):
                        best_set = current[:]
                    break
            if not improved:
                # Try swapping
                for i in range(len(current)):
                    if time.time() > deadline - 10:
                        break
                    for x in range(1, N + 1):
                        if x in current_set:
                            continue
                        new_current = current[:i] + [x] + current[i+1:]
                        if is_valid_set(new_current):
                            # Try to add more after swap
                            new_set = set(new_current)
                            for y in range(1, N + 1):
                                if y in new_set:
                                    continue
                                if len(new_current) >= MAX_SIZE:
                                    break
                                cand2 = new_current + [y]
                                if is_valid_set(cand2):
                                    new_current.append(y)
                                    new_set.add(y)
                                    improved = True
                                    break
                            current = new_current
                            current_set = set(current)
                            if len(current) > len(best_set):
                                best_set = current[:]
                            break
                    if improved:
                        break

    # Simulated annealing to try to find larger valid sets
    if time.time() < deadline - 5:
        rng = random.Random(42)
        current = best_set[:]
        current_pen = penalty_score(current)
        best_sa = current[:]
        best_sa_pen = current_pen
        best_sa_size = len(current)

        T = 10.0
        cooling = 0.995

        for iteration in range(50000):
            if time.time() > deadline - 5:
                break
            if T < 0.01:
                break

            move_type = rng.random()
            if move_type < 0.3:
                # Remove a random element
                if len(current) > 1:
                    idx = rng.randrange(len(current))
                    new_current = current[:idx] + current[idx+1:]
                    new_pen = penalty_score(new_current)
                    delta = new_pen - current_pen
                    if delta <= 0 or rng.random() < np.exp(-delta / T):
                        current = new_current
                        current_pen = new_pen
                        if len(current) > best_sa_size or (len(current) == best_sa_size and current_pen < best_sa_pen):
                            best_sa = current[:]
                            best_sa_pen = current_pen
                            best_sa_size = len(current)
            elif move_type < 0.6:
                # Add a random element
                if len(current) < MAX_SIZE:
                    x = rng.randint(1, N)
                    if x not in current:
                        new_current = current + [x]
                        new_pen = penalty_score(new_current)
                        delta = new_pen - current_pen
                        if delta <= 0 or rng.random() < np.exp(-delta / T):
                            current = new_current
                            current_pen = new_pen
                            if len(current) > best_sa_size or (len(current) == best_sa_size and current_pen < best_sa_pen):
                                best_sa = current[:]
                                best_sa_pen = current_pen
                                best_sa_size = len(current)
            else:
                # Swap a random element
                if len(current) > 1:
                    idx = rng.randrange(len(current))
                    x = rng.randint(1, N)
                    if x not in current:
                        new_current = current[:idx] + [x] + current[idx+1:]
                        new_pen = penalty_score(new_current)
                        delta = new_pen - current_pen
                        if delta <= 0 or rng.random() < np.exp(-delta / T):
                            current = new_current
                            current_pen = new_pen
                            if len(current) > best_sa_size or (len(current) == best_sa_size and current_pen < best_sa_pen):
                                best_sa = current[:]
                                best_sa_pen = current_pen
                                best_sa_size = len(current)

            T *= cooling

        if len(best_sa) > len(best_set):
            best_set = best_sa[:]

    # Final: try to extend best_set with all possible additions
    if time.time() < deadline - 2:
        current = best_set[:]
        current_set = set(current)
        for x in range(1, N + 1):
            if time.time() > deadline - 2:
                break
            if x in current_set:
                continue
            if len(current) >= MAX_SIZE:
                break
            candidate = current + [x]
            if is_valid_set(candidate):
                current.append(x)
                current_set.add(x)
        if len(current) > len(best_set):
            best_set = current[:]

    return best_set[:MAX_SIZE]
# EVOLVE-BLOCK-END
