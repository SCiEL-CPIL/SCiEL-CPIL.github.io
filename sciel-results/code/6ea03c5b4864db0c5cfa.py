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
    for k in range(n, 0, -1):
        for combo in itertools.combinations(A, k):
            if is_sidon(combo):
                return k
    return 0

def count_sidon_subsets(A, k):
    """
    Count the number of Sidon subsets of size k in A.
    Lower count means 'harder' to find a Sidon subset, which is good for us.
    Returns 0 if no Sidon subset of size k exists.
    """
    n = len(A)
    count = 0
    for combo in itertools.combinations(A, k):
        if is_sidon(combo):
            count += 1
    return count

def construct():
    start_time = time.time()
    # Safety margin: stop 5 seconds before 90s limit
    deadline = start_time + 85.0
    
    # Initial candidate: Contiguous block is a strong starting point
    # [0, 11] has max Sidon size 4.
    current_A = list(range(12))
    best_A = list(current_A)
    best_val = max_sidon_size(best_A)  # Expected 4
    
    # Soft metric: count of Sidon subsets of size best_val
    # If best_val is 4, we want to minimize count_sidon_subsets(A, 4)
    # If count becomes 0, then max_sidon_size < 4, i.e., 3 or less
    target_k = best_val
    current_count = count_sidon_subsets(current_A, target_k)
    best_count = current_count
    
    # Search parameters
    iterations = 0
    no_improve_counter = 0
    best_so_far_val = best_val
    
    while time.time() < deadline:
        iterations += 1
        
        # Create a perturbed copy
        A_new = list(current_A)
        idx = random.randint(0, 11)
        
        # Biased random value: try to create collisions
        # Picking values close to existing ones increases sum collision probability
        if random.random() < 0.6:
            # Small step
            step = random.randint(-2, 2)
            new_val = A_new[idx] + step
        else:
            # Random value in a small window around current set
            min_val = min(current_A)
            max_val = max(current_A)
            window = max(5, (max_val - min_val) // 2)
            new_val = random.randint(max(0, min_val - window), min(128, max_val + window))
        
        # Ensure validity: [0, 128] and distinct
        if new_val < 0 or new_val > 128:
            new_val = random.randint(0, 128)
        
        if new_val in A_new:
            # Try to resolve conflict by picking another random value
            for _ in range(10):
                new_val = random.randint(0, 128)
                if new_val not in A_new:
                    break
            else:
                continue
        
        A_new[idx] = new_val
        
        # Calculate new max Sidon size
        new_val_score = max_sidon_size(A_new)
        
        if new_val_score < best_val:
            # Found a better (smaller) max Sidon size
            best_val = new_val_score
            best_A = list(A_new)
            current_A = A_new
            best_count = count_sidon_subsets(current_A, best_val)
            current_count = best_count
            no_improve_counter = 0
            # If we find 3 or less, it's excellent
            if best_val <= 3:
                break
        elif new_val_score == best_val:
            # Same max Sidon size; use soft metric to guide
            new_count = count_sidon_subsets(A_new, best_val)
            if new_count < best_count:
                # Fewer Sidon subsets of size best_val - softer landscape improvement
                best_count = new_count
                best_A = list(A_new)
                current_A = A_new
                current_count = new_count
                no_improve_counter = 0
            elif new_count == best_count:
                # Accept with some probability to allow movement in plateau
                if random.random() < 0.3:
                    current_A = A_new
                    current_count = new_count
                no_improve_counter += 1
            else:
                # More Sidon subsets - worse soft metric
                no_improve_counter += 1
        else:
            # Worse max Sidon size
            no_improve_counter += 1
        
        # Periodic reset if stuck
        if no_improve_counter > 2000:
            # Reset current_A to best_A with aggressive perturbation
            current_A = list(best_A)
            for _ in range(3):
                idx = random.randint(0, 11)
                A_temp = list(current_A)
                # Aggressive random jump
                new_val = random.randint(0, 128)
                while new_val in A_temp:
                    new_val = random.randint(0, 128)
                A_temp[idx] = new_val
                current_A = A_temp
            current_count = count_sidon_subsets(current_A, best_val)
            no_improve_counter = 0
        
        # Check deadline
        if time.time() > deadline:
            break
    
    # Final verification: ensure best_A is valid
    # Verify it has exactly 12 distinct integers in [0, 128]
    assert len(best_A) == 12
    assert len(set(best_A)) == 12
    assert all(0 <= x <= 128 for x in best_A)
    
    return best_A
