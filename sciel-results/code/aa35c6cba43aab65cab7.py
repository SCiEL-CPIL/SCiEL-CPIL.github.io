"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from fractions import Fraction

DENOMS = list(range(2, 49))
N = len(DENOMS)
RECS = [Fraction(1, d) for d in DENOMS]
ONE = Fraction(1, 1)

def construct():
    start_time = time.time()
    deadline = start_time + 85  # 85s margin before 90s kill

    best_indices = []
    best_deficit = None  # 1 - sum, positive

    def consider(indices):
        nonlocal best_indices, best_deficit
        if not indices:
            return
        s = sum(RECS[i] for i in indices)
        if s < ONE:
            deficit = ONE - s
            if best_deficit is None or deficit < best_deficit:
                best_deficit = deficit
                best_indices = list(indices)

    # Branch and bound: explore subsets in a DFS with pruning
    # We process denominators in descending order of reciprocal (i.e., ascending denominator)
    # This way, we add large reciprocals first, and the "remaining possible" bound is tighter.
    
    # Actually, for branch-and-bound to work well, we want to prune branches where
    # even adding all remaining items can't beat the current best deficit.
    # Deficit = 1 - sum. We want to minimize deficit, i.e., maximize sum subject to sum < 1.
    # So we want sum as close to 1 as possible from below.
    
    # Pruning: if current_sum + sum_of_all_remaining < best_sum, we can prune.
    # But sum_of_all_remaining depends on which items are left.
    
    # Better approach: sort by reciprocal descending (smallest denominators first).
    # At each node, we decide to include or exclude the next item.
    # Include: new_sum = cur_sum + rec[i]. If new_sum >= 1, we can still record this as a candidate
    #   only if it's < 1... wait, it must be strictly below 1. So if new_sum >= 1, we CANNOT include it.
    #   Actually, if new_sum >= 1, including it violates the constraint, so we skip the include branch.
    # Exclude: continue without this item.
    
    # Pruning bound: if cur_sum + (sum of all remaining reciprocals) <= best_sum, prune.
    # Because even adding everything remaining can't beat best_sum.
    
    # Precompute suffix sums of reciprocals for the sorted order.
    # Order: indices sorted by denominator ascending (reciprocal descending)
    order = list(range(N))  # index i -> denominator i+2, so order is already ascending denom
    
    # suffix_sum[k] = sum of RECS[order[k]], RECS[order[k+1]], ..., RECS[order[N-1]]
    suffix_sum = [Fraction(0, 1)] * (N + 1)
    for k in range(N - 1, -1, -1):
        suffix_sum[k] = suffix_sum[k + 1] + RECS[order[k]]

    best_sum = Fraction(0, 1)  # We'll track best sum (max sum < 1)
    
    def dfs(pos, cur_sum, selected):
        nonlocal best_sum, best_indices, best_deficit
        
        if time.time() > deadline - 1:
            return
        
        # Pruning: if cur_sum + suffix_sum[pos] <= best_sum, no improvement possible
        if cur_sum + suffix_sum[pos] <= best_sum:
            return
        
        if pos == N:
            # Leaf: check if cur_sum < 1 and > best_sum
            if cur_sum < ONE and cur_sum > best_sum:
                best_sum = cur_sum
                best_indices = list(selected)
                best_deficit = ONE - cur_sum
            return
        
        # Option 1: Include order[pos]
        new_sum = cur_sum + RECS[order[pos]]
        if new_sum < ONE:
            selected.append(order[pos])
            dfs(pos + 1, new_sum, selected)
            selected.pop()
        
        # Option 2: Exclude order[pos]
        # Pruning for exclude: if cur_sum + suffix_sum[pos+1] <= best_sum, skip
        if cur_sum + suffix_sum[pos + 1] > best_sum:
            dfs(pos + 1, cur_sum, selected)
    
    # Run DFS
    dfs(0, Fraction(0, 1), [])
    
    # If DFS didn't find anything (shouldn't happen), fallback to greedy
    if not best_indices:
        greedy_indices = []
        total = Fraction(0, 1)
        for i in range(N):
            if total + RECS[i] < ONE:
                greedy_indices.append(i)
                total += RECS[i]
        return greedy_indices
    
    return best_indices
# EVOLVE-BLOCK-END
