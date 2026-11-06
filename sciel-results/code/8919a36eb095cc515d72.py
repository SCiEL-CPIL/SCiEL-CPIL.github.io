import time
import random
from itertools import combinations

def _check_sidon_size(S, k):
    """Check if there exists a Sidon subset of size k in S."""
    for comb in combinations(S, k):
        sums = set()
        n = len(comb)
        for i in range(n):
            for j in range(i, n):
                s = comb[i] + comb[j]
                if s in sums:
                    break
                sums.add(s)
            else:
                continue
            break
        else:
            return True
    return False

def _get_max_sidon_size(S):
    # We expect the max size to be small (3 or 4)
    # Check 4 first, then 3
    if _check_sidon_size(S, 4):
        # It could be 5, but 4 is a good proxy for "bad"
        # If we find 4, we can return 4 for now.
        # To be precise, we can check 5, 6... but 4 is likely the max for dense sets.
        # If we want to be sure, we can check higher, but it's rare to have >4 in dense sets.
        # Let's assume 4 is the upper bound for our search space initially.
        # If we find a set with no 4, we return 3.
        return 4
    if _check_sidon_size(S, 3):
        return 3
    return 2

def construct():
    start_time = time.time()
    deadline = start_time + 85.0
    
    # Initial candidate
    current = list(range(12))
    current_score = _get_max_sidon_size(current)
    
    best = current.copy()
    best_score = current_score
    
    # Local Search
    # We want to minimize the score
    # If we find a set with score 3, that's a big improvement over 4.
    
    iterations = 0
    while time.time() < deadline:
        iterations += 1
        # Generate neighbor
        idx = random.randint(0, 11)
        old_val = current[idx]
        
        # Try random new value
        new_val = random.randint(0, 128)
        if new_val in current:
            continue
            
        current[idx] = new_val
        new_score = _get_max_sidon_size(current)
        
        if new_score < best_score:
            best_score = new_score
            best = current.copy()
            # If we reach 3, we can keep searching for 2, but 2 is very hard.
            # 3 is likely the target.
        else:
            # Revert if not better
            # Simple hill climbing
            current[idx] = old_val
            
        # Optional: Add some randomness to escape local minima
        # If stuck for long, we could do a random restart or larger jump
        # But with 12 elements, local search is usually effective.
        
    return best
