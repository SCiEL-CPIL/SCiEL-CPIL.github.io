import numpy as np
import time

def heuristics(demands, capacity):
    n = len(demands)
    demands = np.array(demands)
    
    # 1. Run FFD to get bin assignment
    # Sort indices by demand descending
    sorted_indices = np.argsort(demands)[::-1]
    
    # Bins: list of lists of item indices
    bins = []
    # remaining capacity for each bin
    bin_remaining = []
    
    for idx in sorted_indices:
        item_size = demands[idx]
        placed = False
        for k in range(len(bins)):
            if bin_remaining[k] >= item_size:
                bins[k].append(idx)
                bin_remaining[k] -= item_size
                placed = True
                break
        if not placed:
            bins.append([idx])
            bin_remaining.append(capacity - item_size)
            
    # 2. Build heuristic matrix
    # Initialize with base value
    H = np.ones((n, n))
    
    # Map item to bin index
    item_bin = {}
    for b_idx, b in enumerate(bins):
        for item in b:
            item_bin[item] = b_idx
            
    # Boost pairs in the same bin
    # First, general same-bin boost
    # To be efficient, iterate through bins
    for b in bins:
        # All pairs in this bin get a moderate boost
        # We want to avoid O(n^2) per bin if bins are large, but n=120 is small.
        # For each pair (u, v) in b, H[u, v] = max(H[u, v], 10.0)
        for i in range(len(b)):
            for j in range(len(b)):
                if i != j:
                    u = b[i]
                    v = b[j]
                    H[u, v] = max(H[u, v], 10.0)
                    
    # Boost consecutive pairs in FFD order (higher value)
    for b in bins:
        for k in range(len(b) - 1):
            u = b[k]
            v = b[k+1]
            # The item v was placed right after u in FFD
            # This specific transition is "good"
            H[u, v] = max(H[u, v], 100.0)
            # Also consider the reverse? In FFD, the larger is usually first.
            # If d[u] > d[v], u is placed before v.
            # If d[v] > d[u], v is placed before u.
            # The list `b` is in order of placement.
            # So b[k] is placed before b[k+1].
            # So H[b[k], b[k+1]] is the "forward" FFD step.
            # Should we also boost H[b[k+1], b[k]]? 
            # If the ACO tries to place b[k+1] before b[k], it's deviating from FFD order.
            # But since FFD puts larger first, and b[k] is placed first, d[b[k]] >= d[b[k+1]] is likely (not strictly if equal sizes, but generally).
            # Let's just boost the forward direction strongly.
            
    # Ensure all values are positive.
    # The prompt says "Heuristic entries that are not positive become 1e-9".
    # Our values are 1.0, 10.0, 100.0. All positive.
    # What about infeasible pairs?
    # The prompt implies the ACO only picks *fitting* items.
    # If we assume the evaluator filters by fitness, the value for infeasible pairs doesn't affect probability.
    # However, if we want to be robust against an evaluator that calculates probability over all items and then fails if infeasible is picked (unlikely, "picks a fitting item" is explicit), 
    # or if we want to discourage infeasible pairs just in case of a bug in the evaluator's filtering logic?
    # Let's check the constraint: "demands are integers from 20 to 99", "capacity is 150".
    # Max sum is 99+99=198 > 150. So infeasible pairs exist.
    # If the evaluator does NOT filter, it must handle the probability. 
    # "picks a fitting item with probability proportional to..."
    # This phrasing is standard for "restrict candidate set to fitting items, then select".
    # I will assume filtering happens.
    # But just in case, should I set infeasible pairs to 1e-9?
    # If I set them to 1e-9, and the evaluator filters, it's ignored.
    # If I set them to 1e-9, and the evaluator DOESN'T filter, the probability of picking an infeasible item is tiny (if pheromone is similar magnitude). 
    # This is safer.
    
    pair_sums = demands[:, None] + demands[None, :]
    infeasible = pair_sums > capacity
    H[infeasible] = 1e-9
    
    return H
