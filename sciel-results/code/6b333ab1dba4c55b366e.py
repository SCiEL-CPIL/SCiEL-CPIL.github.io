"""Rewritten program using branch-and-bound DFS with descending denominators."""
# EVOLVE-BLOCK-START
def construct():
    from fractions import Fraction
    import math

    denominators = list(range(2, 49))  # [2, 3, ..., 48]
    n = len(denominators)
    
    # We want to select a subset of indices such that sum(1/d_i) < 1,
    # minimizing the deficit 1 - sum(1/d_i).
    # Equivalently, maximize the sum subject to sum < 1.
    
    # Branch and bound DFS:
    # - Iterate denominators in descending order (larger denominators first? or smaller?)
    #   The directive says "iterates denominators in descending order". Let's interpret this
    #   as processing indices in order of decreasing denominator value, i.e., from 48 down to 2.
    #   Actually, let's re-read: "iterates denominators in descending order". This likely means
    #   we consider the list of denominators sorted in descending order: [48, 47, ..., 2].
    #   But for branch and bound, the order matters for pruning efficiency.
    #   
    #   Let's think about it differently. We have a set of items with values 1/d.
    #   We want to find a subset with sum just below 1.
    #   
    #   For effective pruning, we should process items in an order that allows good bounds.
    #   A common approach is to sort by value (or density) and use the sum of remaining items
    #   as an upper bound on what can be added.
    #
    #   Let's sort denominators in ascending order: [2, 3, 4, ..., 48].
    #   Then process from smallest to largest? Or largest to smallest?
    #
    #   The directive says "descending order". Let's use denominators sorted descending: [48, 47, ..., 2].
    #   When we process a denominator d, the remaining denominators are those smaller than d (since we're going down).
    #   The sum of all remaining reciprocals gives an upper bound on additional sum.
    #
    #   Actually, let me reconsider. If we process in descending order of denominator (48, 47, ..., 2),
    #   then at any point, the remaining items have denominators smaller than the current one,
    #   so their reciprocals are larger. This might not give the tightest bound.
    #
    #   Alternatively, if we process in ascending order of denominator (2, 3, ..., 48),
    #   then remaining items have larger denominators, so smaller reciprocals.
    #   The sum of remaining reciprocals would be a smaller upper bound, which is better for pruning.
    #
    #   But the directive explicitly says "descending order". Let's follow it.
    #   Maybe "descending order" refers to the order in which we make decisions, 
    #   i.e., we decide on whether to include each denominator starting from the largest.
    #
    #   Let's just implement it as stated: process denominators in descending order [48, 47, ..., 2].
    #   At each step, we have a current sum S and a current index i in the sorted list.
    #   The remaining denominators are those at indices i+1, i+2, ..., n-1 (which are smaller denominators).
    #   The maximum additional sum we can get is the sum of all remaining reciprocals.
    #   If S + sum_remaining < 1, we could potentially add all of them, but we need S + added < 1.
    #   The best we can do from this branch is at most S + sum_remaining (if it's < 1).
    #   If S + sum_remaining < best_sum_found, we can prune.
    #
    #   Wait, we want to maximize the sum subject to sum < 1.
    #   best_sum is the best sum < 1 found so far.
    #   If current_sum + sum_of_all_remaining <= best_sum, we can prune.
    #   Also, if current_sum >= 1, we prune (invalid).
    
    # Sort denominators in descending order
    sorted_denoms = sorted(denominators, reverse=True)  # [48, 47, ..., 2]
    n = len(sorted_denoms)
    
    # Precompute suffix sums: suffix_sum[i] = sum of 1/d for d in sorted_denoms[i:]
    suffix_sum = [Fraction(0)] * (n + 1)
    for i in range(n - 1, -1, -1):
        suffix_sum[i] = suffix_sum[i + 1] + Fraction(1, sorted_denoms[i])
    
    best_sum = Fraction(0)
    best_indices = []
    
    def dfs(idx, current_sum, current_indices):
        nonlocal best_sum, best_indices
        
        # Pruning: if current sum is already >= 1, invalid
        if current_sum >= 1:
            return
        
        # Pruning: even if we add all remaining, can't beat best_sum
        if current_sum + suffix_sum[idx] <= best_sum:
            return
        
        # Base case: no more items to consider
        if idx == n:
            if current_sum > best_sum:
                best_sum = current_sum
                best_indices = list(current_indices)
            return
        
        # Option 1: Skip current denominator
        dfs(idx + 1, current_sum, current_indices)
        
        # Option 2: Take current denominator
        new_sum = current_sum + Fraction(1, sorted_denoms[idx])
        if new_sum < 1:  # Only valid if still strictly below 1
            current_indices.append(idx)
            dfs(idx + 1, new_sum, current_indices)
            current_indices.pop()
    
    # Map from position in sorted_denoms to original index in denominators [2,3,...,48]
    # Original denominators: index i corresponds to denominator i+2
    # sorted_denoms[j] is some denominator d, its original index is d-2
    
    dfs(0, Fraction(0), [])
    
    # Convert positions in sorted_denoms to original zero-based indices
    # sorted_denoms[j] = d, original index = d - 2
    result = [sorted_denoms[i] - 2 for i in best_indices]
    
    return result
# EVOLVE-BLOCK-END
