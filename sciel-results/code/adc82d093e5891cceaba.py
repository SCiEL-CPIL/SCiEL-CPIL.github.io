def largest_sidon_subset(S):
    """Find largest subset of S with all pair sums (i<=j) unique."""
    S = sorted(S)
    n = len(S)
    from itertools import combinations
    
    for k in range(n, 1, -1):
        for subset in combinations(S, k):
            sums = set()
            valid = True
            for i in range(len(subset)):
                for j in range(i, len(subset)):
                    s = subset[i] + subset[j]
                    if s in sums:
                        valid = False
                        break
                    sums.add(s)
                if not valid:
                    break
            if valid:
                return k
    return 2  # any two distinct elements form a Sidon set

def construct():
    import random
    
    # Start with the densest set (AP) which minimizes Sidon subset size
    best_set = list(range(12))
    best_size = largest_sidon_subset(best_set)
    
    # Local search: try to swap elements to reduce max Sidon subset size
    random.seed(42)
    current_set = best_set[:]
    current_size = best_size
    
    # Perform hill climbing
    for _ in range(200):
        # Pick a random index to change
        idx = random.randint(0, 11)
        old_val = current_set[idx]
        
        # Pick a new random value
        new_val = random.randint(0, 128)
        
        # Ensure distinctness
        if new_val in current_set and new_val != old_val:
            continue
            
        # Apply swap
        current_set[idx] = new_val
        
        # Evaluate
        new_size = largest_sidon_subset(current_set)
        
        if new_size < current_size:
            current_size = new_size
            best_set = current_set[:]
            best_size = new_size
            if best_size == 3: # Can't get better than 3 (2 is impossible for n=12)
                break
        else:
            # Revert if not improved (strict improvement)
            current_set[idx] = old_val

    return best_set
