import time
import random
import itertools

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
        for combo in itertools.combinations(A, k):
            if is_sidon(combo):
                return k
    return 0

def has_sidon_subset(A, k):
    """
    Check if A has ANY Sidon subset of size k.
    Returns True if found, False otherwise.
    This is faster than max_sidon_size when we only care about a threshold.
    """
    n = len(A)
    for combo in itertools.combinations(A, k):
        if is_sidon(combo):
            return True
    return False

def count_sidon_subsets(A, k):
    """
    Count the number of Sidon subsets of size k in A.
    Lower count means "harder" to find a Sidon subset, which is good for us.
    """
    n = len(A)
    count = 0
    for combo in itertools.combinations(A, k):
        if is_sidon(combo):
            count += 1
    return count

def construct():
    start_time = time.time()
    # Safety margin: stop 5 seconds before 60s limit
    deadline = start_time + 55.0
    
    # Initial candidate: Contiguous block is a strong starting point
    # [0, 11] has max Sidon size 4.
    current_A = list(range(12))
    best_A = list(current_A)
    best_val = max_sidon_size(best_A)
    
    # If we start at 4, we are looking for 3.
    # If we start higher, we look for lower.
    
    # Track "soft" metric: number of Sidon subsets of size best_val (or best_val+1 if best_val is high)
    # To make the landscape smoother, we track the count of Sidon subsets of size (best_val)
    # If best_val is 4, we want to minimize count_sidon_subsets(A, 4).
    # If count becomes 0, then max_sidon_size < 4, i.e., 3.
    
    target_k = best_val
    current_count = count_sidon_subsets(current_A, target_k)
    best_count = current_count
    
    # Search parameters
    iterations = 0
    no_improve_counter = 0
    
    while time.time() < deadline:
        iterations += 1
        
        # Create a perturbed copy
        A_new = list(current_A)
        idx = random.randint(0, 11)
        
        # Biased random value
        # Strategy: Try to create collisions.
        # Picking values close to existing ones increases the chance of sum collisions.
        
        if random.random() < 0.6:
            # Small step
            step = random.randint(-2, 2)
            new_val = A_new[idx] + step
        else:
            # Random value in a small window
            min_val = min(current_A)
            max_val = max(current_A)
            # Expand window slightly
            window = max(5, (max_val - min_val) // 2)
            new_val = random.randint(max(0, min_val - window), min(128, max_val + window))
        
        # Ensure validity: [0, 128] and distinct
        if new_val < 0 or new_val > 128:
            new_val = random.randint(0, 128)
            
        if new_val in A_new:
            # Try to resolve conflict
            # Pick a random value that is not in A_new
            attempts = 0
            while new_val in A_new and attempts < 10:
                new_val = random.randint(0, 128)
                attempts += 1
            if new_val in A_new:
                continue
        
        A_new[idx] = new_val
        
        # Evaluate A_new
        # First, check if we improved the max Sidon size
        # We only need to check if there is a Sidon subset of size best_val.
        # If yes, max_val >= best_val.
        # If no, max_val < best_val.
        
        if has_sidon_subset(A_new, best_val):
            # Max size is still at least best_val
            new_val_score = best_val
            # We might be able to find a larger one? 
            # Since we are minimizing, and we started from a good state, 
            # it's unlikely to jump up, but possible if we broke structure.
            # However, checking for larger sizes is expensive.
            # Let's just assume it's best_val for now, and check exact size if we think it might be better.
            # Actually, if it has a Sidon subset of size best_val, its max size is >= best_val.
            # It could be larger. But we want to minimize.
            # So if it's >= best_val, it's not an improvement in max size.
            # We should compare "soft" metrics.
            
            # Soft metric: Count of Sidon subsets of size best_val
            new_count = count_sidon_subsets(A_new, best_val)
            
            # Accept if soft metric is better (fewer Sidon subsets)
            # Or if it's equal, accept with some probability (simulated annealing style)
            if new_count < best_count:
                # Improvement in soft metric
                current_A = A_new
                current_count = new_count
                best_count = new_count
                no_improve_counter = 0
            else:
                # No improvement in soft metric
                # Accept with small probability to escape local minima
                if random.random() < 0.1:
                    current_A = A_new
                    current_count = new_count
                else:
                    # Revert
                    pass
                no_improve_counter += 1
                
        else:
            # No Sidon subset of size best_val
            # This means max_sidon_size(A_new) < best_val
            # This is an improvement!
            
            # Calculate exact max size
            new_val_score = max_sidon_size(A_new)
            
            if new_val_score < best_val:
                best_val = new_val_score
                best_A = A_new
                current_A = A_new
                no_improve_counter = 0
                
                # Update target_k and counts for the new best_val
                target_k = best_val
                current_count = count_sidon_subsets(current_A, target_k)
                best_count = current_count
                
                # If we reached 3, we can stop or try for 2 (unlikely)
                if best_val <= 3:
                    # Check if we can get 2? 
                    # A Sidon set of size 2 is always Sidon (2 elements: a, b -> sums a+a, a+b, b+b. 
                    # a+a=2a, a+b, b+b=2b. 2a != a+b != 2b for distinct a,b? 
                    # 2a = a+b => a=b (no). a+b = 2b => a=b (no). 2a=2b => a=b (no).
                    # So any 2 distinct integers form a Sidon set.
                    # So max_sidon_size is always at least 2 for any set of distinct integers.
                    # Can we get max size 2? That means no 3-element Sidon subset.
                    # A 3-element set {a,b,c} is Sidon if:
                    # 2a, a+b, a+c, 2b, b+c, 2c are all distinct.
                    # 2a = a+b => a=b (no)
                    # 2a = a+c => a=c (no)
                    # 2a = 2b => a=b (no)
                    # a+b = a+c => b=c (no)
                    # a+b = 2b => a=b (no)
                    # a+b = 2c => a+b=2c
                    # ...
                    # It is possible to have a set of 12 integers with no 3-element Sidon subset.
                    # Example: {0, 1, 2, 4, 8, ...} powers of 2? 
                    # {0,1,2}: 0+0=0, 0+1=1, 0+2=2, 1+1=2. Collision 0+2=1+1. Not Sidon.
                    # So {0,1,2} is NOT Sidon.
                    # Can we construct a set of 12 with max Sidon size 2?
                    # This is a very strong constraint.
                    # Let's just return if we get 3, as 2 is extremely hard.
                    return best_A
            else:
                # This shouldn't happen if has_sidon_subset returned False
                # But just in case
                pass
        
        # Periodic reset if stuck
        if no_improve_counter > 500:
            # Reset to best_A and perturb
            current_A = list(best_A)
            current_count = best_count
            no_improve_counter = 0
            
            # Apply a few random perturbations
            for _ in range(3):
                idx = random.randint(0, 11)
                A_new = list(current_A)
                A_new[idx] = random.randint(0, 128)
                while A_new.count(A_new[idx]) > 1:
                    A_new[idx] = random.randint(0, 128)
                current_A = A_new
                current_count = count_sidon_subsets(current_A, target_k)

    return best_A
