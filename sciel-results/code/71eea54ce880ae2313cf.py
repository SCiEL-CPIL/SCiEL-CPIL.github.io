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

def has_sidon_subset(A, k):
    """Check if A has ANY Sidon subset of size k."""
    n = len(A)
    for combo in itertools.combinations(A, k):
        if is_sidon(combo):
            return True
    return False

def count_sidon_subsets(A, k):
    """Count the number of Sidon subsets of size k in A."""
    n = len(A)
    count = 0
    for combo in itertools.combinations(A, k):
        if is_sidon(combo):
            count += 1
    return count

def max_sidon_size(A):
    """Compute the maximum size of a Sidon subset in A."""
    n = len(A)
    for k in range(n, 0, -1):
        for combo in itertools.combinations(A, k):
            if is_sidon(combo):
                return k
    return 0

def collision_count(A):
    """Count the number of sum collisions in A (pairs of pairs with same sum).
    Higher collision count means more additive structure, which is good for us."""
    sums = {}
    n = len(A)
    for i in range(n):
        for j in range(i, n):
            s = A[i] + A[j]
            if s not in sums:
                sums[s] = 0
            sums[s] += 1
    # Count collisions: for each sum with m pairs, there are C(m,2) collisions
    total = 0
    for m in sums.values():
        if m >= 2:
            total += m * (m - 1) // 2
    return total

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin before 60s limit
    
    # Strategy: Build from overlapping APs to maximize local additive dependencies
    # An AP {a, a+d, a+2d} gives collision: a+(a+2d) = (a+d)+(a+d)
    # Overlapping APs create a web of collisions
    
    # Try multiple structural templates based on APs
    templates = []
    
    # Template 1: Two overlapping APs of length 4
    # {0,1,2,3} and {2,3,4,5} overlap at {2,3}
    # Combined: {0,1,2,3,4,5} - 6 elements, extend with more APs
    # Full set: {0,1,2,3,4,5,6,7,8,9,10,11} - just contiguous
    
    # Template 2: Interleaved APs with different differences
    # AP1: {0, 2, 4, 6} (d=2)
    # AP2: {1, 3, 5, 7} (d=2)  
    # AP3: {8, 10, 12} (d=2)
    # AP4: {9, 11, 13} (d=2)
    # Combined: {0,1,2,3,4,5,6,7,8,9,10,11,12,13} - too many, pick 12
    
    # Template 3: Dense cluster with internal AP structure
    # {0,1,2,3,4,5,6,7,8,9,10,11}
    
    # Template 4: Two separate dense clusters
    # {0,1,2,3,4,5} and {6,7,8,9,10,11}
    
    # Template 5: AP-based with varied differences
    # {0, 1, 2, 3, 5, 6, 7, 8, 10, 11, 12, 13}
    # Contains APs: {0,1,2}, {1,2,3}, {5,6,7}, {6,7,8}, {10,11,12}, {11,12,13}
    
    # Template 6: Maximum density - contiguous
    # {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11}
    
    # Let's try several candidates and pick the best
    candidates = [
        list(range(12)),  # Contiguous [0,11]
        list(range(0, 12)),  # Same
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
        # AP-based: {0,2,4,6,8,10} ∪ {1,3,5,7,9,11} = [0,11]
        # Try: {0,1,2,3,4,5,6,7,8,9,10,13} - gap at 11,12
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 13],
        # Try: {0,1,2,3,4,5,6,7,8,9,11,12} - gap at 10
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 11, 12],
        # Try: {0,1,2,3,4,5,6,7,8,10,11,12} - gap at 9
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 10, 11, 12],
        # Try: {0,1,2,3,4,5,6,7,9,10,11,12} - gap at 8
        [0, 1, 2, 3, 4, 5, 6, 7, 9, 10, 11, 12],
        # Try: {0,1,2,3,4,5,6,8,9,10,11,12} - gap at 7
        [0, 1, 2, 3, 4, 5, 6, 8, 9, 10, 11, 12],
        # Try: {0,1,2,3,4,5,7,8,9,10,11,12} - gap at 6
        [0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12],
        # Try: {0,1,2,3,4,6,7,8,9,10,11,12} - gap at 5
        [0, 1, 2, 3, 4, 6, 7, 8, 9, 10, 11, 12],
        # Try: {0,1,2,3,5,6,7,8,9,10,11,12} - gap at 4
        [0, 1, 2, 3, 5, 6, 7, 8, 9, 10, 11, 12],
        # Try: {0,1,2,4,5,6,7,8,9,10,11,12} - gap at 3
        [0, 1, 2, 4, 5, 6, 7, 8, 9, 10, 11, 12],
        # Try: {0,1,3,4,5,6,7,8,9,10,11,12} - gap at 2
        [0, 1, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
        # Try: {0,2,3,4,5,6,7,8,9,10,11,12} - gap at 1
        [0, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
        # Try: {1,2,3,4,5,6,7,8,9,10,11,12} - shifted
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
    ]
    
    # Evaluate all candidates and pick the best
    best_A = None
    best_val = float('inf')
    best_count = float('inf')
    
    for cand in candidates:
        if len(cand) != 12 or len(set(cand)) != 12:
            continue
        if any(x < 0 or x > 128 for x in cand):
            continue
        
        # Check max Sidon size
        val = max_sidon_size(cand)
        
        if val < best_val or (val == best_val and collision_count(cand) > collision_count(best_A) if best_A else False):
            best_val = val
            best_A = list(cand)
            best_count = count_sidon_subsets(cand, val) if val <= 5 else 0
    
    if best_A is None:
        best_A = list(range(12))
        best_val = max_sidon_size(best_A)
    
    current_A = list(best_A)
    current_count = count_sidon_subsets(current_A, best_val) if best_val <= 5 else 0
    
    # If we already have val=3, we're done
    if best_val <= 3:
        return best_A
    
    # Local search to try to improve
    no_improve = 0
    max_no_improve = 5000
    
    while time.time() < deadline:
        no_improve += 1
        if no_improve > max_no_improve:
            # Reset to best and try different perturbation
            current_A = list(best_A)
            current_count = count_sidon_subsets(current_A, best_val) if best_val <= 5 else 0
            no_improve = 0
        
        # Create perturbed copy
        A_new = list(current_A)
        idx = random.randint(0, 11)
        
        # Biased perturbation: try to create new sum collisions
        r = random.random()
        if r < 0.4:
            # Small step: move by -2 to 2
            step = random.randint(-2, 2)
            new_val = A_new[idx] + step
        elif r < 0.7:
            # Try a value that creates a collision with existing elements
            # Pick two existing elements a, b and try a+b-a = b (trivial) or 
            # try values that make a + new = b + c for some a,b,c in A
            # Simpler: pick a random existing element and offset it
            other_idx = random.randint(0, 11)
            if other_idx != idx:
                offset = random.choice([-2, -1, 1, 2])
                new_val = A_new[other_idx] + offset
            else:
                new_val = A_new[idx] + random.randint(-2, 2)
        else:
            # Random value in a small window around the set
            min_val = min(current_A)
            max_val = max(current_A)
            window = max(3, (max_val - min_val) // 3)
            new_val = random.randint(max(0, min_val - window), min(128, max_val + window))
        
        # Ensure validity
        if new_val < 0 or new_val > 128:
            new_val = random.randint(0, min(128, max(0, min(current_A) - 5), max(current_A) + 5))
            if new_val < 0 or new_val > 128:
                new_val = random.randint(0, 128)
        
        if new_val in A_new:
            # Try to resolve
            attempts = 0
            while new_val in A_new and attempts < 5:
                new_val = random.randint(max(0, min(current_A) - 3), min(128, max(current_A) + 3))
                attempts += 1
            if new_val in A_new:
                continue
        
        A_new[idx] = new_val
        
        # Evaluate: first check if we can beat current best_val
        if has_sidon_subset(A_new, best_val):
            # Still has Sidon subset of size best_val, so max >= best_val
            # Use soft metric: count of Sidon subsets of size best_val
            new_count = count_sidon_subsets(A_new, best_val)
            
            if new_count < current_count:
                current_A = A_new
                current_count = new_count
                no_improve = 0
            elif new_count == current_count and random.random() < 0.2:
                current_A = A_new
            # else reject
        else:
            # No Sidon subset of size best_val!
            # This means max_sidon_size(A_new) < best_val
            # Calculate exact size
            new_val_score = max_sidon_size(A_new)
            
            if new_val_score < best_val:
                best_val = new_val_score
                best_A = list(A_new)
                current_A = A_new
                current_count = count_sidon_subsets(current_A, best_val) if best_val <= 5 else 0
                no_improve = 0
                
                if best_val <= 3:
                    return best_A
            else:
                # Shouldn't happen, but just in case
                current_A = A_new
    
    return best_A
