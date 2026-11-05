"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from math import gcd

def construct():
    start_time = time.time()
    deadline = start_time + 85.0
    
    N = 22
    universe = list(range(1, N + 1))
    
    # Compute LCM of 1..22
    def lcm(a, b):
        return a * b // gcd(a, b)
    
    L = 1
    for i in range(1, N + 1):
        L = lcm(L, i)
    
    # weights[i-1] = L // i
    weights = [L // i for i in range(1, N + 1)]
    
    # Greedy: try to build largest set with distinct subset sums
    # Sort by weight (which is L/i, so larger i = smaller weight)
    # Try different orderings and greedily add if it preserves distinctness
    
    def has_distinct_subset_sums(w):
        """Check if all subset sums of w are distinct."""
        k = len(w)
        if k > 20:
            return True  # optimistic
        sums = {0}
        for weight in w:
            new_sums = []
            for s in sums:
                ns = s + weight
                if ns in sums:
                    return False
                new_sums.append(ns)
            # Check for duplicates within new_sums and against existing
            for ns in new_sums:
                if ns in sums:
                    return False
            sums.update(new_sums)
        return True
    
    # Try multiple greedy orderings
    best_set = []
    
    # Ordering 1: by index (1, 2, 3, ..., 22)
    # Ordering 2: by weight ascending (larger indices first)
    # Ordering 3: by weight descending (smaller indices first)
    # Ordering 4: random-ish
    
    orderings = [
        list(range(N)),                    # by index
        sorted(range(N), key=lambda i: weights[i]),  # by weight asc
        sorted(range(N), key=lambda i: -weights[i]), # by weight desc
        sorted(range(N), key=lambda i: (weights[i] % 7, i)),  # some heuristic
        [i for i in range(N) if i % 2 == 0] + [i for i in range(N) if i % 2 == 1],
        [i for i in range(N) if i % 2 == 1] + [i for i in range(N) if i % 2 == 0],
    ]
    
    for order in orderings:
        if time.time() > deadline:
            break
        
        current_set = []
        current_weights = []
        
        for idx in order:
            if time.time() > deadline:
                break
            
            test_weights = current_weights + [weights[idx]]
            if has_distinct_subset_sums(test_weights):
                current_set.append(idx)
                current_weights.append(weights[idx])
        
        if len(current_set) > len(best_set):
            best_set = list(current_set)
    
    # Also try a local search / hill climbing from the best found
    if best_set and time.time() < deadline - 5:
        # Try to improve by swapping or adding
        best_set_set = set(best_set)
        improved = True
        while improved and time.time() < deadline - 5:
            improved = False
            for add_idx in range(N):
                if add_idx in best_set_set:
                    continue
                test_set = list(best_set) + [add_idx]
                test_weights = [weights[i] for i in test_set]
                if has_distinct_subset_sums(test_weights):
                    best_set = test_set
                    best_set_set.add(add_idx)
                    improved = True
                    break
    
    # Verify final result
    if best_set:
        w = [weights[i] for i in best_set]
        sums = {0}
        valid = True
        for weight in w:
            new_sums = set()
            for s in sums:
                ns = s + weight
                if ns in sums or ns in new_sums:
                    valid = False
                    break
                new_sums.add(ns)
            if not valid:
                break
            sums.update(new_sums)
        if valid:
            return [i + 1 for i in best_set]
    
    return [1, 2, 3, 5, 7, 11, 13, 17, 19]
# EVOLVE-BLOCK-END
