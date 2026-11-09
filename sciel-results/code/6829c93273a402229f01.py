import random
import time
from itertools import combinations

def is_sidon(subset):
    """Check if a tuple/list of integers is a Sidon set (unique pair sums)."""
    sums = set()
    n = len(subset)
    for i in range(n):
        for j in range(i, n):
            s = subset[i] + subset[j]
            if s in sums:
                return False
            sums.add(s)
    return True

def has_sidon_subset(S, size):
    """Check if any subset of S of given size is Sidon."""
    for comb in combinations(S, size):
        if is_sidon(comb):
            return True
    return False

def get_max_sidon_size(S):
    """Find the size of the largest Sidon subset of S."""
    # Check down from 12
    for k in range(12, 2, -1):
        if has_sidon_subset(S, k):
            return k
    if has_sidon_subset(S, 3):
        return 3
    return 2

def construct():
    start_time = time.time()
    time_limit = 55.0
    
    best_set = None
    best_score = float('inf')
    
    # Run many trials to find a set with minimal max Sidon size
    # Theoretical min is 3. We aim for 3.
    
    num_trials = 500
    
    for trial in range(num_trials):
        if time.time() - start_time > time_limit:
            break
            
        current_set = []
        candidates = list(range(0, 129))
        sums = set()
        
        # Optional: Randomize the order of candidates to break symmetry
        random.shuffle(candidates)
        
        for i in range(12):
            best_x = None
            max_collisions = -1
            best_candidates = []
            
            for x in candidates:
                if x in current_set:
                    continue
                
                new_sums = [2 * x]
                for c in current_set:
                    new_sums.append(x + c)
                
                collisions = 0
                for s in new_sums:
                    if s in sums:
                        collisions += 1
                
                if collisions > max_collisions:
                    max_collisions = collisions
                    best_candidates = [x]
                elif collisions == max_collisions:
                    best_candidates.append(x)
            
            if not best_candidates:
                best_x = random.choice(candidates)
            else:
                best_x = random.choice(best_candidates)
            
            current_set.append(best_x)
            candidates.remove(best_x)
            
            sums.add(2 * best_x)
            for c in current_set:
                if c != best_x:
                    sums.add(best_x + c)
        
        score = get_max_sidon_size(current_set)
        
        if score < best_score:
            best_score = score
            best_set = current_set.copy()
            
            if best_score == 3:
                # Found the theoretical minimum (likely). 
                # We could break, but maybe 2 is possible? No.
                # We break to save time, as 3 is the best we can do.
                break

    if best_set is None:
        # Fallback: Try to construct an AP-like set or random
        # An AP of 12 elements has max Sidon size ~4.
        # A random set has max Sidon size ~5-6.
        # We want to minimize.
        # If we failed to find 3, return the best we found or a default.
        best_set = random.sample(range(0, 129), 12)
        # Evaluate fallback just in case
        fallback_score = get_max_sidon_size(best_set)
        if fallback_score < best_score:
            best_score = fallback_score
            best_set = best_set

    return best_set
