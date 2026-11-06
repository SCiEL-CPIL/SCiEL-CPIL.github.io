"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def construct():
    from fractions import Fraction
    import time

    deadline = time.time() + 80.0  # safety margin before 90s kill

    denominators = list(range(2, 49))
    n = len(denominators)
    # Reciprocals in descending order of value (ascending denominator)
    # We want to explore large fractions first to get good solutions quickly
    # but for branch and bound with suffix sums, we need a fixed order.
    # Let's use ascending index (denominator 2, 3, ..., 48) for the DFS path.
    # Actually, to prune effectively, we want to know the max possible sum
    # from the remaining items. If we process in ascending denominator order,
    # the suffix sum (sum of 1/d for d > current) is the max additional sum.
    
    # Precompute suffix sums: suffix[i] = sum(1/denominators[j] for j in range(i, n))
    # If we are at index i, and we have current sum S, the max we can add is suffix[i].
    # If S + suffix[i] < best_sum (or deficit > best_deficit), prune.
    # Wait, we want to MINIMIZE deficit (1 - sum). So we want to MAXIMIZE sum < 1.
    # Pruning condition: if current_sum + suffix[i] <= best_sum, we can't improve best_sum.
    # Note: strict inequality sum < 1. So best_sum is always < 1.
    
    suffix_sums = [Fraction(0)] * (n + 1)
    for i in range(n - 1, -1, -1):
        suffix_sums[i] = suffix_sums[i + 1] + Fraction(1, denominators[i])
    
    best_sum = Fraction(0)
    best_indices = []
    
    # DFS state:
    # idx: current index to consider
    # current_sum: sum of selected so far
    # selected: list of selected indices
    
    # Optimization: Use a list for selected, and update in place.
    # Also, we can skip items if adding them would make sum >= 1.
    
    def dfs(idx, current_sum, selected):
        nonlocal best_sum, best_indices
        
        if time.time() > deadline:
            return
            
        # If we've reached the end
        if idx == n:
            if current_sum > best_sum:
                best_sum = current_sum
                best_indices = list(selected)
            return
            
        # Pruning: even if we take all remaining, can we beat best_sum?
        if current_sum + suffix_sums[idx] <= best_sum:
            return
            
        # Option 1: Skip current item
        dfs(idx + 1, current_sum, selected)
        
        # Option 2: Take current item
        new_sum = current_sum + Fraction(1, denominators[idx])
        if new_sum < 1:
            selected.append(idx)
            dfs(idx + 1, new_sum, selected)
            selected.pop()
            
    dfs(0, Fraction(0), [])
    
    if not best_indices:
        # Fallback: pick the single largest fraction < 1
        # 1/2 is the largest
        return [0]
        
    return best_indices
# EVOLVE-BLOCK-END
