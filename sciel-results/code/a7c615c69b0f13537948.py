import time
import sys
from fractions import Fraction

def construct():
    start = time.time()
    deadline = start + 80  # safety margin within 90s
    
    denominators = list(range(2, 49))
    n = len(denominators)
    
    # Calculate LCM of 2..48
    def gcd(a, b):
        while b:
            a, b = b, a % b
        return a
            
    def lcm(a, b):
        return a // gcd(a, b) * b
        
    L = 1
    for d in denominators:
        L = lcm(L, d)
        
    # Weights are integers: weight[i] = L / denominators[i]
    weights = [L // d for d in denominators]
    
    # Sort indices by weight descending (larger reciprocals first)
    # This helps branch and bound prune earlier
    sorted_indices = sorted(range(n), key=lambda i: weights[i], reverse=True)
    
    # Precompute suffix sums for pruning
    # suffix_sum[i] = sum of weights[sorted_indices[i:]]
    suffix_sum = [0] * (n + 1)
    for i in range(n - 1, -1, -1):
        suffix_sum[i] = suffix_sum[i + 1] + weights[sorted_indices[i]]
    
    # Branch and bound to find subset with sum closest to L but < L
    best_sum = 0
    best_mask = 0  # bitmask over sorted_indices
    
    # Initial seed: greedy approach
    # Try to build a good initial solution
    current_sum = 0
    current_mask = 0
    for i in range(n):
        idx = sorted_indices[i]
        if current_sum + weights[idx] < L:
            current_sum += weights[idx]
            current_mask |= (1 << i)
    
    if current_sum > best_sum:
        best_sum = current_sum
        best_mask = current_mask
    
    # Also try a few heuristic starting points
    # Start with just the largest few
    for k in range(1, min(10, n)):
        s = 0
        m = 0
        for i in range(k):
            s += weights[sorted_indices[i]]
            m |= (1 << i)
        if s < L and s > best_sum:
            best_sum = s
            best_mask = m
    
    # Branch and bound DFS
    # We process indices in order of sorted_indices
    # State: (pos, current_sum, current_mask)
    # Prune if current_sum + suffix_sum[pos] <= best_sum (can't improve)
    # Prune if current_sum >= L (invalid)
    
    def dfs(pos, current_sum, current_mask):
        nonlocal best_sum, best_mask
        
        if time.time() > deadline - 5:
            return
        
        if pos == n:
            if current_sum < L and current_sum > best_sum:
                best_sum = current_sum
                best_mask = current_mask
            return
        
        # Upper bound: current_sum + suffix_sum[pos]
        # If even taking all remaining can't beat best, prune
        if current_sum + suffix_sum[pos] <= best_sum:
            return
        
        # Option 1: skip sorted_indices[pos]
        dfs(pos + 1, current_sum, current_mask)
        
        # Option 2: take sorted_indices[pos]
        new_sum = current_sum + weights[sorted_indices[pos]]
        if new_sum < L:
            dfs(pos + 1, new_sum, current_mask | (1 << pos))
    
    # Run DFS
    dfs(0, 0, 0)
    
    # Reconstruct result from best_mask
    selected = set()
    for i in range(n):
        if best_mask & (1 << i):
            selected.add(sorted_indices[i])
    
    # Local search polish: try to improve by single swaps or additions
    current_sum = sum(weights[i] for i in selected)
    current_deficit = L - current_sum
    
    improved = True
    while improved and time.time() < deadline - 10:
        improved = False
        best_improve = None
        best_improve_deficit = current_deficit
        
        # Try adding one element
        for add_idx in range(n):
            if add_idx in selected:
                continue
            new_sum = current_sum + weights[add_idx]
            if new_sum < L:
                new_deficit = L - new_sum
                if new_deficit < best_improve_deficit:
                    best_improve_deficit = new_deficit
                    best_improve = (add_idx, None)
        
        # Try swapping: add one, remove one
        for add_idx in range(n):
            if add_idx in selected:
                continue
            for rem_idx in list(selected):
                new_sum = current_sum + weights[add_idx] - weights[rem_idx]
                if 0 < new_sum < L:
                    new_deficit = L - new_sum
                    if new_deficit < best_improve_deficit:
                        best_improve_deficit = new_deficit
                        best_improve = (add_idx, rem_idx)
        
        if best_improve is not None:
            add_idx, rem_idx = best_improve
            selected.add(add_idx)
            if rem_idx is not None:
                selected.discard(rem_idx)
            current_sum = sum(weights[i] for i in selected)
            current_deficit = L - current_sum
            improved = True
    
    # Convert to sorted list of original indices
    result = sorted(selected)
    
    # Verify
    if result:
        total_frac = sum(Fraction(1, denominators[k]) for k in result)
        if total_frac < 1:
            return result
    
    # Fallback
    return [0, 1, 5]
