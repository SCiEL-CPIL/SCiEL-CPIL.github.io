"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def construct():
    from fractions import Fraction
    import math

    # Denominators are [2, 3, ..., 48]
    denominators = list(range(2, 49))
    n = len(denominators)
    reciprocals = [Fraction(1, d) for d in denominators]

    # Compute suffix sums: suffix_sum[i] = sum of reciprocals[i:]
    suffix_sum = [Fraction() for _ in range(n + 1)]
    for i in range(n - 1, -1, -1):
        suffix_sum[i] = suffix_sum[i + 1] + reciprocals[i]

    # Greedy initial solution: take as many as possible in order (ascending denominator)
    # This gives a valid starting upper bound on deficit
    best_indices = []
    best_sum = Fraction()
    for i in range(n):
        if best_sum + reciprocals[i] < 1:
            best_indices.append(i)
            best_sum += reciprocals[i]
    best_deficit = 1 - best_sum  # positive deficit

    # Branch and bound DFS, iterating denominators in descending order
    # We process from largest denominator (smallest reciprocal) to smallest
    # Actually, let's iterate in the order that maximizes pruning.
    # Process in descending order of denominator (ascending index from n-1 down to 0)
    # means we consider large denominators first. But for pruning, we want to know
    # the max possible additional sum from remaining choices.
    
    # Let's restructure: iterate indices from 0 to n-1, but at each step we can
    # either include or exclude. Prune if current_sum + suffix_sum[i+1] <= best_sum
    # (i.e., even taking all remaining can't beat best_sum, so deficit can't be smaller)
    # Wait: we want to MINIMIZE deficit = 1 - sum, so MAXIMIZE sum subject to sum < 1.
    # Prune if current_sum + suffix_sum[i+1] <= best_sum (can't improve)
    # Also prune if current_sum >= 1 (invalid, but since we only add if < 1, this shouldn't happen
    # if we check before adding)

    # Actually the pruning condition: if even taking ALL remaining reciprocals from i onward
    # gives sum <= best_sum, then no branch from here can improve.
    # So: if current_sum + suffix_sum[i] <= best_sum: prune (but we're at index i, so remaining is i..n-1)
    # Let me redefine: at node (i, current_sum), we decide on indices i, i+1, ..., n-1.
    # Max possible sum = current_sum + suffix_sum[i]
    # If current_sum + suffix_sum[i] <= best_sum: prune
    # Also, if current_sum >= 1: prune (invalid)

    # For better pruning, process in an order that finds good solutions early.
    # Greedy ascending gave a decent start. Let's do DFS in ascending index order
    # (small denominators first) since they have larger reciprocals and impact the sum more.

    # But the directive says "iterates denominators in descending order". Let me follow that.
    # Descending order of denominators = descending order of index (48, 47, ..., 2)
    # = indices n-1, n-2, ..., 0

    # Let me re-index: let's process in order of decreasing denominator.
    # Let order = list(range(n-1, -1, -1))  # indices n-1 down to 0
    # At position k in this order, we're considering index order[k].
    # Suffix sum for this order: suffix2[k] = sum of reciprocals[order[k]] for k..len(order)-1

    order = list(range(n - 1, -1, -1))  # indices from n-1 down to 0
    m = len(order)
    suffix2 = [Fraction() for _ in range(m + 1)]
    for k in range(m - 1, -1, -1):
        suffix2[k] = suffix2[k + 1] + reciprocals[order[k]]

    best_sum_val = best_sum
    best_indices_val = best_indices[:]

    def dfs(k, current_sum, current_indices):
        nonlocal best_sum_val, best_indices_val

        # Prune: if even taking all remaining can't beat best
        if current_sum + suffix2[k] <= best_sum_val:
            return

        # If we've considered all denominators
        if k == m:
            if current_sum > best_sum_val and current_sum < 1:
                best_sum_val = current_sum
                best_indices_val = current_indices[:]
            return

        # Option 1: skip this denominator
        dfs(k + 1, current_sum, current_indices)

        # Option 2: include this denominator
        new_sum = current_sum + reciprocals[order[k]]
        if new_sum < 1:  # must be strictly below 1
            current_indices.append(order[k])
            dfs(k + 1, new_sum, current_indices)
            current_indices.pop()

    dfs(0, Fraction(), [])

    return best_indices_val
# EVOLVE-BLOCK-END
