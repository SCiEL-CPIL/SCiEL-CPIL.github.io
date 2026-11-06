import time
from fractions import Fraction

def construct():
    start_time = time.time()
    deadline = start_time + 85.0

    denominators = list(range(2, 49))
    n = len(denominators)

    # Process in descending order of denominator value (i.e., ascending fraction value)
    # Actually, descending denominator = smaller fractions first
    # We want to try larger fractions first to get good initial solutions
    # So sort by denominator ascending: 2, 3, 4, ... 48
    # Index 0 -> denominator 2 (largest fraction 1/2)
    # Index 47 -> denominator 48 (smallest fraction 1/48)

    # Precompute reciprocal sums using integer arithmetic with LCM
    from math import lcm
    L = 1
    for d in denominators:
        L = lcm(L, d)

    # weight[i] = L / denominators[i], so sum of weights = L * sum of reciprocals
    weights = [L // d for d in denominators]

    # Precompute suffix sums: suffix_sum[i] = sum of weights[i:]
    # This allows us to bound the maximum additional sum from position i onward
    suffix_sum = [0] * (n + 1)
    for i in range(n - 1, -1, -1):
        suffix_sum[i] = suffix_sum[i + 1] + weights[i]

    best_sum = -1  # We want to maximize the sum (minimize deficit), keep it < L
    best_mask = 0

    # DFS: at position pos, we've accumulated current_sum
    # We can either include or exclude denominators[pos]
    # Pruning: if current_sum + suffix_sum[pos] <= best_sum, no need to continue
    # Also: if current_sum >= L, we've exceeded 1, so this branch is invalid

    def dfs(pos, current_sum, mask):
        nonlocal best_sum, best_mask

        if time.time() > deadline:
            return

        # Pruning: even if we take all remaining, can't beat best
        if current_sum + suffix_sum[pos] <= best_sum:
            return

        # Base case: we've considered all positions
        if pos == n:
            if current_sum > best_sum and current_sum < L:
                best_sum = current_sum
                best_mask = mask
            return

        # Option 1: Include current element
        new_sum = current_sum + weights[pos]
        if new_sum < L:  # Must stay strictly below L
            dfs(pos + 1, new_sum, mask | (1 << pos))

        # Option 2: Exclude current element
        dfs(pos + 1, current_sum, mask)

    dfs(0, 0, 0)

    # Convert best_mask to indices
    if best_mask == 0:
        return [0]

    result_indices = []
    for i in range(n):
        if best_mask & (1 << i):
            result_indices.append(i)

    # Local search refinement with 1-swap moves
    recip = [Fraction(1, d) for d in denominators]
    current_subset = set(result_indices)
    current_sum_frac = sum(recip[i] for i in current_subset)
    best_deficit = Fraction(1, 1) - current_sum_frac

    if best_deficit <= 0:
        return result_indices

    improved = True
    while improved and time.time() < deadline:
        improved = False
        in_sub = list(current_subset)
        not_in_sub = [i for i in range(n) if i not in current_subset]

        best_swap = None
        best_gain = Fraction(0)

        for i in in_sub:
            for j in not_in_sub:
                new_sum = current_sum_frac - recip[i] + recip[j]
                if new_sum < 1:
                    gain = recip[j] - recip[i]
                    if gain > best_gain:
                        best_gain = gain
                        best_swap = (i, j)

        if best_swap:
            i, j = best_swap
            current_subset.remove(i)
            current_subset.add(j)
            current_sum_frac += best_gain
            improved = True
            new_deficit = Fraction(1, 1) - current_sum_frac
            if new_deficit < best_deficit:
                best_deficit = new_deficit
                result_indices = list(current_subset)

    return result_indices
