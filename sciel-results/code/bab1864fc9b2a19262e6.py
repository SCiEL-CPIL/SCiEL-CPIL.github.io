import random
import itertools
import time

def _is_sidon(subset):
    """Check if a list of distinct ints is a Sidon set."""
    sums = set()
    for i in range(len(subset)):
        for j in range(i, len(subset)):
            s = subset[i] + subset[j]
            if s in sums:
                return False
            sums.add(s)
    return True

def _max_sidon_size(A):
    """Find the maximum size of a Sidon subset of A."""
    # Since we suspect max is 3 or 4, we can check for 4 first.
    # If no 4, check for 3.
    # If no 3, check for 2.
    # For efficiency in the search, we just need to know if it's <= 3.
    pass

def _has_sidon_subset_of_size(A, k):
    """Check if A contains a Sidon subset of size k."""
    for subset in itertools.combinations(A, k):
        if _is_sidon(list(subset)):
            return True
    return False

def construct():
    start_time = time.time()
    deadline = start_time + 50  # 50s safety margin
    
    best_set = None
    best_score = float('inf') # We want to minimize max sidon size
    
    # Seed set
    seed_set = list(range(12))
    
    # Try to find a set with no Sidon subset of size 4
    # Search in [0, 40]
    while time.time() < deadline:
        # Random set in [0, 40]
        A = random.sample(range(41), 12)
        
        # Check if it has a Sidon subset of size 4
        has_4 = _has_sidon_subset_of_size(A, 4)
        
        if not has_4:
            # Check if it has a Sidon subset of size 3
            # It almost certainly does.
            # If it does, max size is 3.
            # If not, max size is 2 (unlikely/impossible).
            if _has_sidon_subset_of_size(A, 3):
                return A
            else:
                # Max size 2? Return this too, it's even better.
                return A
                
    # If not found, return seed or best effort
    # We can try to minimize the number of Sidon 4-subsets
    # But for now, just return seed if nothing better
    return seed_set
