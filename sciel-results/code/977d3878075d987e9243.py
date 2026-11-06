"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def construct():
    from fractions import Fraction
    import math

    denominators = list(range(2, 49))
    n = len(denominators)

    # Precompute suffix sums of reciprocals for pruning
    # suffix_sum[i] = sum of 1/denominators[j] for j from i to n-1
    suffix_sum = [Fraction(0)] * (n + 1)
    for i in range(n - 1, -1, -1):
        suffix_sum[i] = suffix_sum[i + 1] + Fraction(1, denominators[i])

    # We want to find a subset with sum < 1 that minimizes 1 - sum (i.e., maximizes sum < 1)
    best_sum = Fraction(0)
    best_indices = []

    def dfs(start_idx, current_sum, current_indices):
        nonlocal best_sum, best_indices

        # Pruning: if current_sum + suffix_sum[start_idx] <= best_sum, we can't improve
        if current_sum + suffix_sum[start_idx] <= best_sum:
            return

        # If current_sum is already >= 1, we can't add more (but we need strict < 1)
        # Actually, if current_sum >= 1, this branch is invalid for final answer,
        # but we might have found a better sum in a sub-branch? No, adding more only increases sum.
        # So if current_sum >= 1, we cannot use this set or any superset.
        # But wait, we are building the set by adding elements. If current_sum >= 1,
        # any superset will also be >= 1. So we can prune here too?
        # Actually, the condition is that the FINAL sum must be < 1.
        # If current_sum >= 1, we can't add any more elements without violating < 1.
        # But could current_sum itself be a valid candidate? Only if current_sum < 1.
        # So if current_sum >= 1, this branch is dead for producing a valid final set.
        # However, we might have already updated best_sum in a previous step?
        # No, because we only update best_sum when we consider a valid set (sum < 1).
        # So if current_sum >= 1, we just return.
        if current_sum >= 1:
            return

        # Update best if current set is valid (sum < 1) and better than current best
        # The current set is represented by current_indices
        # But we only consider it as a candidate if we stop here.
        # Actually, every prefix of the search path is a valid candidate set.
        # So we should check if current_sum > best_sum.
        if current_sum > best_sum:
            best_sum = current_sum
            best_indices = list(current_indices)

        # Try adding each subsequent element
        for i in range(start_idx, n):
            new_sum = current_sum + Fraction(1, denominators[i])
            # Pruning: if new_sum >= 1, adding this element makes sum >= 1,
            # and any further additions will keep it >= 1, so skip this and all further in this level?
            # Not exactly, because we might skip this element and add a later one.
            # But if new_sum >= 1, we can't use this element in the set.
            # However, we can still try adding elements after i.
            # But wait, if we skip i, we go to i+1. The loop handles that.
            # So we just skip adding i if new_sum >= 1? No, we can't add i if it makes sum >= 1.
            # But we can still consider sets that include i? No, because sum would be >= 1.
            # So we skip i. But we still need to try i+1, i+2, etc.
            # The loop naturally does that.
            # However, if new_sum >= 1, we can prune the branch where we include i.
            # But we still explore branches that exclude i (by continuing the loop).
            # So we just don't recurse on i if new_sum >= 1.
            if new_sum >= 1:
                continue

            # Pruning: if new_sum + suffix_sum[i+1] <= best_sum, we can't improve
            if new_sum + suffix_sum[i + 1] <= best_sum:
                # Since denominators are in ascending order, suffix_sum is decreasing as i increases?
                # No, suffix_sum[i] = sum from i to n-1. As i increases, suffix_sum[i] decreases.
                # So if new_sum + suffix_sum[i+1] <= best_sum, then for any j > i,
                # new_sum + suffix_sum[j+1] <= new_sum + suffix_sum[i+1] <= best_sum.
                # So we can break out of the loop entirely.
                break

            current_indices.append(i)
            dfs(i + 1, new_sum, current_indices)
            current_indices.pop()

    dfs(0, Fraction(0), [])

    return best_indices
# EVOLVE-BLOCK-END
