"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import itertools
import time
import random

# EVOLVE-BLOCK-START

def _is_sidon_with_diagonals(subset):
    """Check if subset has all unordered pair sums unique including diagonals.
    For a set S, the pair sums are {a+b : a,b in S, a<=b}.
    All these sums must be distinct.
    """
    sums = set()
    n = len(subset)
    for i in range(n):
        for j in range(i, n):
            s = subset[i] + subset[j]
            if s in sums:
                return False
            sums.add(s)
    return True

def _max_sidon_size(candidate_set):
    """Find the maximum size of a Sidon subset (with diagonals) within candidate_set.
    Returns the maximum k such that there exists a k-element subset with all pair sums distinct.
    """
    arr = sorted(candidate_set)
    n = len(arr)
    
    # Try from largest possible size down
    for k in range(n, 0, -1):
        # Check if any k-subset is Sidon
        for combo in itertools.combinations(arr, k):
            if _is_sidon_with_diagonals(list(combo)):
                return k
    return 0

def construct():
    """Return exactly 12 distinct integers in [0,128] that minimize the largest Sidon subset size."""
    start_time = time.time()
    deadline = start_time + 45  # 45 second deadline, 15s safety margin
    
    # Fast Sidon check
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

    def max_sidon_size(arr):
        n = len(arr)
        # Check from largest to smallest
        # Optimization: If we find one of size k, return k.
        # We can prune if we know max possible is small.
        for k in range(n, 0, -1):
            for combo in itertools.combinations(arr, k):
                if is_sidon(combo):
                    return k
        return 0

    # Start with an arithmetic progression which is known to have small Sidon subsets
    # {0, 1, ..., 11}
    best_set = list(range(12))
    best_score = max_sidon_size(best_set)
    
    # Local search to try to reduce the max Sidon size
    random.seed(42)
    current = best_set[:]
    current_score = best_score
    
    # Pre-compute some random swaps to speed up
    # We will try to swap elements to create more collisions
    
    while time.time() < deadline:
        # Try a batch of random perturbations
        for _ in range(50):
            if time.time() > deadline:
                break
            
            # Pick a random element to change
            i = random.randint(0, 11)
            old_val = current[i]
            
            # Try to replace with a value that creates a collision with existing pairs
            # Or just random
            new_val = random.randint(0, 128)
            if new_val in current:
                continue
            
            candidate = current[:]
            candidate[i] = new_val
            candidate.sort()
            
            score = max_sidon_size(candidate)
            
            if score < current_score:
                current = candidate
                current_score = score
                best_set = candidate
                best_score = score
                if best_score <= 3:
                    return best_set
            elif score == current_score:
                # Accept sideways moves to explore
                if random.random() < 0.5:
                    current = candidate
        # Reset current to best if we haven't improved in a while? 
        # No, simple hill climbing with sideways moves is fine.
        
    return best_set

# EVOLVE-BLOCK-END
