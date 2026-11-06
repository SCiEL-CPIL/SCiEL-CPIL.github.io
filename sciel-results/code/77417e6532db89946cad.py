"""Improved program for Large sets with Property P."""
import time
import random
import sys

# EVOLVE-BLOCK-START
def _is_valid(S):
    """Check if set S satisfies: for all a<b<=c in S, a does NOT divide b+c."""
    S_sorted = sorted(S)
    n = len(S_sorted)
    for i in range(n):
        a = S_sorted[i]
        for j in range(i + 1, n):
            b = S_sorted[j]
            for k in range(j, n):
                c = S_sorted[k]
                if (b + c) % a == 0:
                    return False
    return True

def _quick_validity_check(S, sample_pairs=5000):
    """Fast probabilistic validity check for large sets."""
    S_sorted = sorted(S)
    n = len(S_sorted)
    if n < 2:
        return True
    # Check all pairs for small sets
    if n <= 30:
        return _is_valid(S)
    # For larger sets, check systematically but with early exit
    # The worst case is when a is small
    for i in range(n):
        a = S_sorted[i]
        # For small a, check all b,c
        for j in range(i + 1, n):
            b = S_sorted[j]
            for k in range(j, n):
                c = S_sorted[k]
                if (b + c) % a == 0:
                    return False
    return True

def _greedy_construct():
    """Greedy construction: start from top, try to add elements from bottom."""
    # Start with the known valid upper tail
    # Condition: 3*min(S) > 2*max(S) guarantees validity
    # For max=256, min > 256*2/3 = 170.67, so min >= 171
    # This gives [171, 256], size 86
    
    # Try to extend downward
    S = set(range(171, 257))
    
    # Try adding elements from 1 to 170
    for x in range(1, 171):
        candidate = S | {x}
        if _quick_validity_check(candidate):
            S = candidate
    
    return S

def _local_search(S, max_iterations=50000, timeout=55.0):
    """Local search to improve the set."""
    start_time = time.time()
    best_S = set(S)
    best_size = len(best_S)
    
    # Precompute for efficiency
    all_elements = list(range(1, 257))
    
    for iteration in range(max_iterations):
        if time.time() - start_time > timeout:
            break
        
        # Try adding a random element not in S
        if len(best_S) < 256:
            not_in_S = [x for x in all_elements if x not in best_S]
            if not_in_S:
                x = random.choice(not_in_S)
                candidate = best_S | {x}
                if _quick_validity_check(candidate):
                    best_S = candidate
                    best_size = len(best_S)
        
        # Try swapping: remove one, add another
        if len(best_S) > 1:
            elements_in_S = list(best_S)
            if len(elements_in_S) >= 2:
                x_to_remove = random.choice(elements_in_S)
                not_in_S = [x for x in all_elements if x not in best_S]
                if not_in_S:
                    x_to_add = random.choice(not_in_S)
                    candidate = (best_S - {x_to_remove}) | {x_to_add}
                    if _quick_validity_check(candidate):
                        best_S = candidate
                        best_size = len(best_S)
        
        # Try removing one element and adding two
        if len(best_S) > 1:
            elements_in_S = list(best_S)
            x_to_remove = random.choice(elements_in_S)
            not_in_S = [x for x in all_elements if x not in best_S]
            if len(not_in_S) >= 2:
                x1, x2 = random.sample(not_in_S, 2)
                candidate = (best_S - {x_to_remove}) | {x1, x2}
                if _quick_validity_check(candidate):
                    best_S = candidate
                    best_size = len(best_S)
    
    return best_S

def construct():
    start_time = time.time()
    
    # Start with greedy construction
    S = _greedy_construct()
    size_greedy = len(S)
    
    # Local search with multiple restarts
    best_S = S
    best_size = size_greedy
    
    # Use a fixed seed for reproducibility but allow some randomness
    random.seed(42)
    
    remaining_time = 55.0 - (time.time() - start_time)
    if remaining_time > 5:
        best_S = _local_search(S, max_iterations=100000, timeout=remaining_time - 3)
        best_size = len(best_S)
    
    # Additional restarts if time permits
    remaining_time = 55.0 - (time.time() - start_time)
    restart_count = 0
    while remaining_time > 10 and restart_count < 5:
        # Perturb the best solution
        perturbed = set(best_S)
        for _ in range(3):
            if perturbed:
                x_to_remove = random.choice(list(perturbed))
                perturbed.remove(x_to_remove)
            not_in_S = [x for x in range(1, 257) if x not in perturbed]
            if not_in_S:
                x_to_add = random.choice(not_in_S)
                perturbed.add(x_to_add)
        
        result = _local_search(perturbed, max_iterations=50000, timeout=remaining_time - 5)
        if len(result) > best_size:
            best_S = result
            best_size = len(result)
        restart_count += 1
        remaining_time = 55.0 - (time.time() - start_time)
    
    # Final safety check
    if not _is_valid(best_S) and len(best_S) > 0:
        # Fall back to the guaranteed valid set
        return list(range(171, 257))
    
    return list(best_S)
# EVOLVE-BLOCK-END
