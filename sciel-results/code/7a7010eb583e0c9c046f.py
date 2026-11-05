import time
import random
from itertools import combinations

def is_sidon(subset):
    """Check if a subset of integers has all unique pair sums (including diagonals)."""
    n = len(subset)
    sums = set()
    for i in range(n):
        for j in range(i, n):
            s = subset[i] + subset[j]
            if s in sums:
                return False
            sums.add(s)
    return True

def max_sidon_size(A):
    """
    Compute the maximum size of a Sidon subset in A.
    A is a list of distinct integers.
    """
    n = len(A)
    # Check from largest possible size down to 1
    for k in range(n, 0, -1):
        for combo in combinations(A, k):
            if is_sidon(combo):
                return k
    return 0

def construct():
    start_time = time.time()
    deadline = start_time + 55  # Safety margin for 60s limit
    
    # Initial candidate: Contiguous block is a strong starting point
    best_subset = list(range(12))
    best_score = max_sidon_size(best_subset)
    
    random.seed(42)
    
    # Search Strategy:
    # 1. Local search around the current best.
    # 2. Random restarts with "dense" initialization.
    
    current_subset = list(best_subset)
    
    # We want to minimize best_score.
    # If best_score is 4, can we get 3?
    # Theoretical min for 12 elements:
    # If max Sidon is 3, then any 4 elements have a collision.
    
    for restart in range(200):
        if time.time() > deadline:
            break
            
        # Generate candidate
        # Bias towards dense sets (small range)
        if restart == 0:
            candidate = list(current_subset)
        else:
            # Pick a random window size, usually small
            window_size = random.randint(12, 30)
            start = random.randint(0, 128 - window_size + 1)
            # Sample 12 distinct values from [start, start+window_size-1]
            available = range(start, start + window_size)
            candidate = random.sample(list(available), 12)
        
        current_score = max_sidon_size(candidate)
        
        if current_score < best_score:
            best_score = current_score
            best_subset = candidate
            current_subset = candidate
            if best_score <= 3:
                break # 3 is likely the minimum, try to confirm or just stop
        
        # Local Search: Hill Climbing
        improved = True
        while improved and time.time() < deadline:
            improved = False
            # Try to improve by swapping one element
            for i in range(12):
                if time.time() > deadline:
                    break
                old_val = candidate[i]
                
                # Try replacing old_val with a new value
                # To keep it dense, try values near the existing ones
                # Or just try all values in [0, 128]? That's 129 * 12 * cost_eval.
                # cost_eval is ~1000 ops. 129*12*1000 = 1.5M ops per pass.
                # 1 pass ~ 0.1s? Maybe.
                
                # Let's try a subset of candidates for new_val
                # Neighbors of existing values + random
                candidates_new_val = set()
                for v in candidate:
                    for d in range(-5, 6):
                        nv = v + d
                        if 0 <= nv <= 128:
                            candidates_new_val.add(nv)
                
                # Also add some randoms
                for _ in range(10):
                    candidates_new_val.add(random.randint(0, 128))
                
                for new_val in candidates_new_val:
                    if new_val == old_val:
                        continue
                    if new_val in candidate and new_val != old_val:
                        continue
                    
                    candidate[i] = new_val
                    new_score = max_sidon_size(candidate)
                    
                    if new_score < current_score:
                        current_score = new_score
                        improved = True
                        if new_score < best_score:
                            best_score = new_score
                            best_subset = candidate
                            current_subset = candidate
                            if best_score <= 3:
                                break
                        # Restart inner loop? Or continue?
                        # Continuing might find further improvements
                        break # Found improvement, restart outer for loop logic?
                    else:
                        candidate[i] = old_val
                
                if best_score <= 3:
                    break
            
            if best_score <= 3:
                break
                
        if best_score <= 3:
            break

    return sorted(best_subset)
