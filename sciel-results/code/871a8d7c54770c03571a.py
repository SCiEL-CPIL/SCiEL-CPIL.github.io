import random
import time
import itertools

def is_sidon(subset):
    sums = set()
    n = len(subset)
    for i in range(n):
        for j in range(i, n):
            s = subset[i] + subset[j]
            if s in sums:
                return False
            sums.add(s)
    return True

def count_sidon_4_subsets(s):
    count = 0
    for combo in itertools.combinations(s, 4):
        if is_sidon(combo):
            count += 1
    return count

def largest_sidon_subset_size(s):
    # Check sizes 6, 5, 4, 3
    # For n=12, we check if any Sidon subset of size k exists
    for k in range(6, 2, -1):
        for combo in itertools.combinations(s, k):
            if is_sidon(combo):
                return k
    return 3

def construct():
    start_time = time.time()
    deadline = start_time + 55.0
    
    # Initial candidates
    # 1. AP - known to have large Sidon subsets
    cand_ap = list(range(12))
    
    # 2. Start with a random set
    random.seed(42)
    
    best_set = list(cand_ap)
    best_true_size = largest_sidon_subset_size(best_set)
    best_surrogate = count_sidon_4_subsets(best_set)
    
    # Local search with simulated annealing
    current_set = sorted(random.sample(range(129), 12))
    current_surrogate = count_sidon_4_subsets(current_set)
    
    if current_surrogate < best_surrogate:
        best_set = list(current_set)
        best_surrogate = current_surrogate
        best_true_size = largest_sidon_subset_size(best_set)
    
    # Simulated annealing parameters
    T = 10.0
    T_min = 0.1
    cooling = 0.999
    
    while time.time() < deadline:
        if best_surrogate == 0 and best_true_size == 3:
            break
            
        # Pick a position to modify
        idx = random.randint(0, 11)
        old_val = current_set[idx]
        
        # Try a random new value
        new_val = random.randint(0, 128)
        if new_val in current_set and new_val != old_val:
            continue
            
        # Compute new surrogate incrementally
        # Only 4-subsets containing the changed element are affected
        # 4-subsets containing idx: C(11,3) = 165
        # 4-subsets not containing idx: C(11,4) = 330
        # Total: 495
        
        # Quick check: count Sidon 4-subsets involving the new element
        new_score = 0
        others = [current_set[j] for j in range(12) if j != idx]
        
        # 4-subsets containing new_val: choose 3 from others
        for triple in itertools.combinations(others, 3):
            subset = tuple(sorted([new_val] + list(triple)))
            if is_sidon(subset):
                new_score += 1
        
        # 4-subsets not containing new_val: choose 4 from others
        # These are unchanged, so we can reuse part of the old count
        # But for simplicity, let's just compute all if it's fast enough
        # C(11,4) = 330, plus the 165 above = 495 total. Fast enough.
        
        # Actually, let's just recount all for correctness
        new_set = current_set[:]
        new_set[idx] = new_val
        new_score = count_sidon_4_subsets(new_set)
        
        delta = new_score - current_surrogate
        
        # Accept if better, or with probability exp(-delta/T) if worse
        if delta < 0 or random.random() < pow(2.718281828, -delta / T):
            current_set = new_set
            current_surrogate = new_score
            
            if current_surrogate < best_surrogate:
                best_set = list(current_set)
                best_surrogate = current_surrogate
                # Update true size occasionally (expensive)
                if random.random() < 0.05:
                    true_size = largest_sidon_subset_size(best_set)
                    if true_size < best_true_size:
                        best_true_size = true_size
        
        # Cool down
        T = max(T * cooling, T_min)
        
        # Occasionally restart
        if random.random() < 0.005:
            current_set = sorted(random.sample(range(129), 12))
            current_surrogate = count_sidon_4_subsets(current_set)
            if current_surrogate < best_surrogate:
                best_set = list(current_set)
                best_surrogate = current_surrogate
                best_true_size = largest_sidon_subset_size(best_set)
            T = 5.0
    
    # Final: verify the best set
    final_true_size = largest_sidon_subset_size(best_set)
    
    return best_set
