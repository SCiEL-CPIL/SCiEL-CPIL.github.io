"""Combined program: fast DP-based validity + geometric core + diverse greedy search."""
# EVOLVE-BLOCK-START
import time
import random
import math

def construct():
    deadline = time.time() + 80.0  # safety margin before 90s limit
    n = 512
    target_size = 48

    def _build_dp(arr):
        """dp[k] = bitset of all subset sums of size k from arr."""
        m = len(arr)
        if m == 0:
            return [1]
        dp = [0] * (m + 1)
        dp[0] = 1
        for val in arr:
            for k in range(m, 0, -1):
                dp[k] |= (dp[k - 1] << val)
        return dp

    def _is_valid_incremental(existing, x):
        """Check if adding x to existing keeps it valid. Returns True if valid."""
        if not existing:
            return True
        m = len(existing)
        total_sum = sum(existing)
        dp = _build_dp(existing)

        # Check 1: x is not the mean of any subset of existing of size >= 2
        for k in range(2, m + 1):
            target = k * x
            if target > total_sum:
                break
            if (dp[k] >> target) & 1:
                return False

        # Check 2: no element a in existing is the mean of a subset of existing∪{x}
        # of size >= 2 that includes x. Such subset = x + t elements from existing,
        # total size t+1 >= 2 so t >= 1. Mean = a => sum(t elements) = (t+1)*a - x.
        for a in existing:
            for t in range(1, m + 1):
                target = (t + 1) * a - x
                if target < 0:
                    continue
                if target > total_sum:
                    break
                if (dp[t] >> target) & 1:
                    return False
        return True

    def _greedy_from_order(order, selected_start=None):
        """Build a valid set by greedily adding elements in the given order."""
        if selected_start is None:
            selected = []
            selected_set = set()
        else:
            selected = list(selected_start)
            selected_set = set(selected)
        for x in order:
            if x in selected_set:
                continue
            if _is_valid_incremental(selected, x):
                selected.append(x)
                selected_set.add(x)
                if len(selected) >= target_size:
                    break
        return selected

    def _local_search(selected, max_iterations=30):
        """Try to improve the set by swapping elements."""
        selected = list(selected)
        selected_set = set(selected)
        for _ in range(max_iterations):
            if time.time() > deadline - 5.0:
                break
            improved = False
            for i in range(len(selected)):
                if time.time() > deadline - 5.0:
                    break
                candidate_remove = selected[i]
                remaining = [s for j, s in enumerate(selected) if j != i]
                remaining_set = set(remaining)
                for x in range(1, n + 1):
                    if x in remaining_set or x == candidate_remove:
                        continue
                    if _is_valid_incremental(remaining, x):
                        selected = remaining + [x]
                        selected_set = remaining_set | {x}
                        improved = True
                        break
                if improved:
                    break
            if not improved:
                break
        return selected

    # --- Strategy 1: Geometric core (provably valid) + greedy expansion ---
    # Powers of 2: no element is the mean of any subset of the others,
    # since any subset sum of {1,2,4,...,2^(k-1)} is < 2^k.
    geo_core = [1, 2, 4, 8, 16, 32, 64, 128, 256]
    rest = [x for x in range(1, n + 1) if x not in geo_core]
    best = _greedy_from_order(rest, selected_start=geo_core)

    if len(best) >= target_size:
        return best

    # --- Strategy 2: Greedy from natural order ---
    order = list(range(1, n + 1))
    result = _greedy_from_order(order)
    if len(result) > len(best):
        best = result
        if len(best) >= target_size:
            return best

    # --- Strategy 3: Greedy from descending order ---
    order_desc = list(range(n, 0, -1))
    result = _greedy_from_order(order_desc)
    if len(result) > len(best):
        best = result
        if len(best) >= target_size:
            return best

    # --- Strategy 4: Random orderings ---
    for trial in range(120):
        if time.time() > deadline - 15.0:
            break
        random.seed(trial)
        order = list(range(1, n + 1))
        random.shuffle(order)
        result = _greedy_from_order(order)
        if len(result) > len(best):
            best = result
            if len(best) >= target_size:
                return best

    # --- Strategy 5: Modular class orderings ---
    for mod in [3, 4, 5, 6, 7, 8, 10, 12]:
        if time.time() > deadline - 8.0:
            break
        for rem in range(mod):
            order = [x for x in range(1, n + 1) if x % mod == rem]
            result = _greedy_from_order(order)
            if len(result) > len(best):
                best = result
                if len(best) >= target_size:
                    return best

    # --- Strategy 6: Biased random orders (fewer divisors = higher priority) ---
    for trial in range(40):
        if time.time() > deadline - 5.0:
            break
        random.seed(5000 + trial)
        weights = []
        for x in range(1, n + 1):
            div_count = 0
            for d in range(1, int(math.isqrt(x)) + 1):
                if x % d == 0:
                    div_count += 1
                    if d != x // d:
                        div_count += 1
            weights.append(1.0 / max(div_count, 1))

        # Sample without replacement using weights
        selected_order = []
        remaining = list(range(n))
        remaining_weights = weights[:]
        for _ in range(n):
            total_w = sum(remaining_weights)
            r = random.random() * total_w
            cum = 0
            for idx, w in enumerate(remaining_weights):
                cum += w
                if cum >= r:
                    selected_order.append(remaining[idx])
                    remaining.pop(idx)
                    remaining_weights.pop(idx)
                    break
        result = _greedy_from_order(selected_order)
        if len(result) > len(best):
            best = result
            if len(best) >= target_size:
                return best

    # --- Strategy 7: Local search on best found ---
    if len(best) < target_size:
        improved = _local_search(best, max_iterations=20)
        if len(improved) > len(best):
            best = improved
            if len(best) >= target_size:
                return best

    return best
# EVOLVE-BLOCK-END
