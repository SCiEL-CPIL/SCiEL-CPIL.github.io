import itertools
import random

def is_sidon(subset):
    """
    Check if a subset is a Sidon set.
    A Sidon set requires all unordered pair sums (including diagonals a+a) to be unique.
    """
    sums = set()
    for i in range(len(subset)):
        for j in range(i, len(subset)):
            s = subset[i] + subset[j]
            if s in sums:
                return False
            sums.add(s)
    return True

def count_sidon_4sets(S):
    """
    Count the number of 4-element subsets of S that are Sidon.
    """
    count = 0
    for subset in itertools.combinations(S, 4):
        if is_sidon(subset):
            count += 1
    return count

def construct():
    # We want to minimize the largest Sidon subset size.
    # This is equivalent to minimizing the number of Sidon 4-subsets.
    # If the count is 0, the max Sidon size is 3.
    
    best_set = None
    best_count = float('inf')
    
    rng = random.Random(42)
    
    # Initial candidates
    initial_sets = [
        list(range(12)),
        list(range(0, 24, 2)),
        list(range(0, 36, 3)),
    ]
    
    # Add random restarts
    for _ in range(50):
        start_range = rng.randint(20, 35)
        candidates = rng.sample(range(0, start_range), 12)
        initial_sets.append(sorted(candidates))
    
    for S_init in initial_sets:
        S = S_init.copy()
        
        current_count = count_sidon_4sets(S)
        
        if current_count < best_count:
            best_count = current_count
            best_set = S.copy()
            if best_count == 0:
                break
        
        # Local search
        for _ in range(2000):
            if best_count == 0:
                break
                
            idx = rng.randint(0, 11)
            old_val = S[idx]
            
            # Try to replace with a value from a small range
            # Sample from [0, 35]
            new_val = rng.randint(0, 35)
            
            # Ensure distinctness
            if new_val in S and new_val != old_val:
                continue
            
            S[idx] = new_val
            S.sort()
            
            new_count = count_sidon_4sets(S)
            
            if new_count < best_count:
                best_count = new_count
                best_set = S.copy()
                if best_count == 0:
                    break
            else:
                # Rollback
                S[idx] = old_val
                S.sort()

    if best_set is None:
        best_set = list(range(12))
        
    return best_set
