import random
import itertools
import time

def is_sidon(S):
    """Check if set S is a Sidon set (all unordered pair sums unique including diagonals)."""
    sums = set()
    for i, x in enumerate(S):
        for j, y in enumerate(S):
            if i <= j:
                s = x + y
                if s in sums:
                    return False
                sums.add(s)
    return True

def has_sidon_subset_of_size_k(A, k):
    """Check if A contains a Sidon subset of size k."""
    for combo in itertools.combinations(A, k):
        if is_sidon(combo):
            return True
    return False

def count_unique_sums(A):
    """Count the number of unique pair sums in A."""
    sums = set()
    for i, x in enumerate(A):
        for j, y in enumerate(A):
            if i <= j:
                sums.add(x + y)
    return len(sums)

def count_sum_collisions(A):
    """Count total number of sum collisions (excess over unique sums).
    Total pairs = n*(n+1)/2. Collisions = total_pairs - unique_sums."""
    n = len(A)
    total_pairs = n * (n + 1) // 2
    unique = count_unique_sums(A)
    return total_pairs - unique

def max_sidon_subset_size(A):
    """Find the maximum size of a Sidon subset in A by trying all sizes from large to small."""
    n = len(A)
    for k in range(n, 0, -1):
        if has_sidon_subset_of_size_k(A, k):
            return k
    return 0

def construct():
    start_time = time.time()
    deadline = start_time + 50  # Safety margin before 90s limit
    
    best_set = None
    best_sidon_size = float('inf')
    
    # Strategy: Use conflict-graph-based greedy + local search
    # Goal: maximize sum collisions to minimize max Sidon subset size
    
    # Initial candidates: various structured sets
    initial_sets = [
        list(range(12)),
        random.sample(range(129), 12),
        random.sample(range(129), 12),
        random.sample(range(129), 12),
        [0, 1, 2, 4, 8, 16, 32, 64, 96, 112, 120, 128],
        [0, 3, 6, 9, 12, 15, 18, 21, 24, 27, 30, 33],  # AP with step 3
        [0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55],  # AP with step 5
    ]
    
    for init_set in initial_sets:
        if time.time() > deadline - 15:
            break
            
        current_set = sorted(init_set)
        
        # Local search: try to maximize sum collisions (minimize unique sums)
        improved = True
        while improved and time.time() < deadline - 15:
            improved = False
            current_collisions = count_sum_collisions(current_set)
            
            # Try all possible single-element replacements
            for i in range(12):
                if time.time() > deadline - 15:
                    break
                old_val = current_set[i]
                
                # Try replacing with values that create more collisions
                # Focus on values that would create sum collisions with existing pairs
                candidates = []
                for new_val in range(129):
                    if new_val in current_set:
                        continue
                    # Quick check: does this value create collisions?
                    # Check if new_val + any existing element creates a sum that already exists
                    new_set = current_set[:]
                    new_set[i] = new_val
                    new_set = sorted(new_set)
                    new_collisions = count_sum_collisions(new_set)
                    if new_collisions > current_collisions:
                        candidates.append((new_collisions, new_val, new_set))
                
                if candidates:
                    # Pick the best candidate
                    candidates.sort(reverse=True)
                    best_collisions, best_val, best_new_set = candidates[0]
                    current_set = best_new_set
                    improved = True
                    break
        
        # Evaluate the actual Sidon size
        if time.time() > deadline - 10:
            break
            
        sidon_size = max_sidon_subset_size(current_set)
        
        if sidon_size < best_sidon_size:
            best_sidon_size = sidon_size
            best_set = current_set[:]
            
            if best_sidon_size <= 4:
                # Great result!
                return best_set
    
    # If we found a good set, return it
    if best_set is not None:
        return best_set
    
    # Fallback: return a set with many collisions
    # Try a few more random searches
    for _ in range(10):
        if time.time() > deadline - 5:
            break
        candidate = sorted(random.sample(range(129), 12))
        sidon_size = max_sidon_subset_size(candidate)
        if sidon_size < best_sidon_size:
            best_sidon_size = sidon_size
            best_set = candidate
            if best_sidon_size <= 4:
                return best_set
    
    if best_set is None:
        return list(range(12))
    
    return best_set
