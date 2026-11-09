import time
from math import lcm
from functools import reduce
from fractions import Fraction
import heapq

def construct():
    start_time = time.time()
    deadline = 50.0  # 50 seconds safety margin
    
    denominators = list(range(2, 49))
    n = len(denominators)
    
    # Calculate LCM of denominators
    L = reduce(lcm, denominators)
    
    # Convert to integer values
    vals = [L // d for d in denominators]
    
    # Choose a prime P > max possible deficit
    # The deficit is at most L (if we pick nothing, deficit = L)
    # But realistically, deficit will be small. Let's use a prime around 10^6
    # Actually, we want P such that the modular arithmetic is efficient
    # and P is large enough to distinguish deficits
    # Let's use a prime P where P > L is too large. Instead, we'll use
    # a different approach: direct Dijkstra on the sum space
    
    # Better approach: Use Dijkstra on states (current_sum, mask)
    # where current_sum is the sum of selected reciprocals as integer (scaled by L)
    # We want to find the subset with sum closest to L but < L
    
    # Since L is huge (~10^18), we can't do full Dijkstra on sum values
    # Instead, let's use a smarter approach
    
    # Approach: Greedy + local search with integer arithmetic
    # Start with a good initial solution, then do local search
    
    # Initial: greedy from largest denominators (smallest fractions)
    selected = set()
    current_sum = 0  # integer sum (scaled by L)
    
    # Sort by value (largest values first = smallest denominators)
    order = sorted(range(n), key=lambda i: -vals[i])
    
    for i in order:
        if current_sum + vals[i] < L:
            selected.add(i)
            current_sum += vals[i]
    
    best_sum = current_sum
    best_selected = selected.copy()
    
    # Local search: 1-swap, 1-add, 1-remove, 1-for-2-swap
    # Use integer arithmetic for speed
    
    # Precompute for speed
    in_set = [False] * n
    for i in selected:
        in_set[i] = True
    
    def try_add(i):
        nonlocal current_sum, in_set
        if not in_set[i] and current_sum + vals[i] < L:
            new_sum = current_sum + vals[i]
            if L - new_sum < L - best_sum:
                return True, new_sum
        return False, current_sum
    
    def try_remove(i):
        nonlocal current_sum, in_set
        if in_set[i]:
            new_sum = current_sum - vals[i]
            if new_sum > 0 and L - new_sum < L - best_sum:
                return True, new_sum
        return False, current_sum
    
    def try_swap(i, j):
        nonlocal current_sum, in_set
        if in_set[i] and not in_set[j]:
            new_sum = current_sum - vals[i] + vals[j]
            if 0 < new_sum < L and L - new_sum < L - best_sum:
                return True, new_sum
        return False, current_sum
    
    # Local search iterations
    improved = True
    while improved and time.time() < deadline:
        improved = False
        
        # Try 1-add
        for i in range(n):
            if not in_set[i]:
                success, new_sum = try_add(i)
                if success:
                    in_set[i] = True
                    current_sum = new_sum
                    if L - current_sum < L - best_sum:
                        best_sum = current_sum
                        best_selected = set(i for i in range(n) if in_set[i])
                    improved = True
                    break
        
        if improved:
            continue
        
        # Try 1-remove
        for i in range(n):
            if in_set[i]:
                success, new_sum = try_remove(i)
                if success:
                    in_set[i] = False
                    current_sum = new_sum
                    if L - current_sum < L - best_sum:
                        best_sum = current_sum
                        best_selected = set(i for i in range(n) if in_set[i])
                    improved = True
                    break
        
        if improved:
            continue
        
        # Try 1-swap
        for i in range(n):
            if in_set[i]:
                for j in range(n):
                    if not in_set[j]:
                        success, new_sum = try_swap(i, j)
                        if success:
                            in_set[i] = False
                            in_set[j] = True
                            current_sum = new_sum
                            if L - current_sum < L - best_sum:
                                best_sum = current_sum
                                best_selected = set(i for i in range(n) if in_set[i])
                            improved = True
                            break
                if improved:
                    break
        
        if improved:
            continue
        
        # Try 1-for-2-swap: remove one, add two
        for i in range(n):
            if in_set[i]:
                for j in range(n):
                    if not in_set[j]:
                        for k in range(n):
                            if not in_set[k] and k != j:
                                new_sum = current_sum - vals[i] + vals[j] + vals[k]
                                if 0 < new_sum < L and L - new_sum < L - best_sum:
                                    in_set[i] = False
                                    in_set[j] = True
                                    in_set[k] = True
                                    current_sum = new_sum
                                    if L - current_sum < L - best_sum:
                                        best_sum = current_sum
                                        best_selected = set(i for i in range(n) if in_set[i])
                                    improved = True
                                    break
                            if improved:
                                break
                        if improved:
                            break
                    if improved:
                        break
                if improved:
                    break
        
        if improved:
            continue
        
        # Try 2-for-1-swap: remove two, add one
        for i in range(n):
            if in_set[i]:
                for j in range(n):
                    if in_set[j] and j != i:
                        for k in range(n):
                            if not in_set[k]:
                                new_sum = current_sum - vals[i] - vals[j] + vals[k]
                                if 0 < new_sum < L and L - new_sum < L - best_sum:
                                    in_set[i] = False
                                    in_set[j] = False
                                    in_set[k] = True
                                    current_sum = new_sum
                                    if L - current_sum < L - best_sum:
                                        best_sum = current_sum
                                        best_selected = set(i for i in range(n) if in_set[i])
                                    improved = True
                                    break
                            if improved:
                                break
                        if improved:
                            break
                    if improved:
                        break
                if improved:
                    break
    
    # If no improvement found, use the best we have
    if not best_selected:
        # Fallback: greedy
        best_selected = set()
        current_sum = 0
        for i in order:
            if current_sum + vals[i] < L:
                best_selected.add(i)
                current_sum += vals[i]
    
    return sorted(best_selected)
