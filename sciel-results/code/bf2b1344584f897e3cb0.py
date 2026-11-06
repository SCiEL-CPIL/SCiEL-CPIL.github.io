"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def construct():
    from fractions import Fraction
    from functools import lru_cache

    # Denominators are 2..48, zero-based indices 0..46
    n = 47
    denoms = list(range(2, 49))
    recips = [Fraction(1, d) for d in denoms]

    # Precompute suffix sums: suffix[i] = sum of recips[i:]
    suffix = [Fraction(0)] * (n + 1)
    for i in range(n - 1, -1, -1):
        suffix[i] = suffix[i + 1] + recips[i]

    best_deficit = None  # We want to minimize 1 - sum, i.e., maximize sum < 1
    best_indices = []

    # Branch and bound DFS
    # We iterate in descending order of denominator (ascending index) but the directive says
    # "iterates denominators in descending order". Let's re-read: "iterates denominators in descending order".
    # Denominators are 2..48. Descending means 48, 47, ..., 2.
    # So we should consider indices in reverse: 46, 45, ..., 0.
    # However, branch and bound works better if we process larger fractions first (smaller denominators)
    # to prune more effectively? Or does the directive specifically require descending denominators?
    # The directive says: "iterates denominators in descending order".
    # Let's follow the directive strictly: order = [46, 45, ..., 0] corresponding to denoms [48, 47, ..., 2].
    
    # Wait, if we iterate in descending order of denominators (48 down to 2), 
    # the "remaining" set at any step k (where we've considered the first k largest denominators)
    # would be the smaller denominators (2..something).
    
    # Let's define the order of consideration:
    # order[i] is the index into denoms for the i-th decision.
    # We want descending denominators: 48, 47, ..., 2.
    # denoms[46]=48, denoms[45]=47, ..., denoms[0]=2.
    # So the sequence of indices to consider is [46, 45, ..., 0].
    
    order = list(range(n - 1, -1, -1))
    
    # Precompute suffix sums based on this order?
    # Let's create an array of reciprocals in the order we will visit them.
    recips_in_order = [recips[order[i]] for i in range(n)]
    
    # suffix_order[i] = sum of recips_in_order[i:]
    suffix_order = [Fraction(0)] * (n + 1)
    for i in range(n - 1, -1, -1):
        suffix_order[i] = suffix_order[i + 1] + recips_in_order[i]

    best_sum = Fraction(0)  # We want to maximize sum < 1
    best_indices_found = []

    def dfs(idx, current_sum, current_indices):
        nonlocal best_sum, best_indices_found, best_deficit

        # Pruning:
        # If current_sum + suffix_order[idx] <= best_sum, we cannot improve (since we need strictly better sum, i.e., lower deficit).
        # Actually, if current_sum + suffix_order[idx] <= best_sum, then even taking all remaining, we can't beat best_sum.
        # So we can prune.
        if current_sum + suffix_order[idx] <= best_sum:
            return

        # Base case: no more items to consider
        if idx == n:
            if current_sum < 1 and current_sum > best_sum:
                best_sum = current_sum
                best_indices_found = list(current_indices)
            return

        # Option 1: Exclude item at idx
        dfs(idx + 1, current_sum, current_indices)

        # Option 2: Include item at idx
        new_sum = current_sum + recips_in_order[idx]
        if new_sum < 1:
            current_indices.append(order[idx])
            dfs(idx + 1, new_sum, current_indices)
            current_indices.pop()

    dfs(0, Fraction(0), [])

    # best_indices_found contains indices into the original denoms list (0..46)
    # The problem asks for zero-based indices into denominators [2,3,...,48].
    # So index 0 corresponds to denom 2, index 46 corresponds to denom 48.
    # Our 'order' list maps position in DFS to original index.
    # 'current_indices' stores the original indices.
    
    return best_indices_found
# EVOLVE-BLOCK-END
