"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from math import gcd
from functools import reduce

def construct():
    start_time = time.time()
    deadline = start_time + 80.0

    denominators = list(range(2, 49))
    n = len(denominators)

    def lcm(a, b):
        return a * b // gcd(a, b)

    L = reduce(lcm, denominators)
    vals = [L // d for d in denominators]

    # Sort by value descending
    order = sorted(range(n), key=lambda i: -vals[i])
    sorted_vals = [vals[i] for i in order]

    # Suffix sums
    suffix = [0] * (n + 1)
    for i in range(n - 1, -1, -1):
        suffix[i] = suffix[i + 1] + sorted_vals[i]

    best_excess = None
    best_mask = None

    # DFS: find min subset sum > L
    # Pruning: if current_sum + suffix[idx] <= L, can't reach > L from this branch
    # Once current_sum > L, we record and stop (adding more only worsens)

    def dfs(idx, current_sum, mask):
        nonlocal best_excess, best_mask

        if time.time() > deadline:
            return

        # Prune: can't reach > L
        if current_sum + suffix[idx] <= L:
            return

        if current_sum > L:
            excess = current_sum - L
            if best_excess is None or excess < best_excess:
                best_excess = excess
                best_mask = mask
            return  # Don't add more items

        if idx == n:
            return

        # Try including item idx
        new_sum = current_sum + sorted_vals[idx]
        new_mask = mask | (1 << idx)
        dfs(idx + 1, new_sum, new_mask)

        # Try excluding item idx
        dfs(idx + 1, current_sum, mask)

    dfs(0, 0, 0)

    if best_mask is not None:
        # best_mask represents the "excluded" set S (sum > L)
        # Answer is complement: indices not in S
        result = []
        for i in range(n):
            # Check if i is in the excluded set
            # order[i] is the original index of the i-th sorted item
            # mask bit i corresponds to order[i]
            in_excluded = False
            for j in range(n):
                if best_mask & (1 << j):
                    if order[j] == i:
                        in_excluded = True
                        break
            if not in_excluded:
                result.append(i)
        return result
    else:
        selected, total = [], 0
        for i, d in enumerate(denominators):
            v = L // d
            if total + v < L:
                selected.append(i)
                total += v
        return selected

# EVOLVE-BLOCK-END
