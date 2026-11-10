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
    deadline = start_time + 55  # 55 second deadline, 5s safety margin
    
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
        for k in range(n, 0, -1):
            for combo in itertools.combinations(arr, k):
                if is_sidon(combo):
                    return k
        return 0

    def count_sidon_subsets(arr, k):
        """Count how many k-element subsets are Sidon."""
        arr = sorted(arr)
        count = 0
        for combo in itertools.combinations(arr, k):
            if is_sidon(combo):
                count += 1
        return count

    # Start with an arithmetic progression
    best_set = list(range(12))
    best_score = max_sidon_size(best_set)
    
    # Local search to try to reduce the max Sidon size
    random.seed(42)
    current = best_set[:]
    current_score = best_score
    
    # Track the number of Sidon subsets at current_score for tie-breaking
    if current_score > 0:
        current_sidon_count = count_sidon_subsets(current, current_score)
    else:
        current_sidon_count = 0
    
    while time.time() < deadline:
        # Try a batch of random perturbations
        for _ in range(100):
            if time.time() > deadline:
                break
            
            # Pick a random element to change
            i = random.randint(0, 11)
            old_val = current[i]
            
            # Try to replace with a value that creates a collision with existing pairs
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
                if best_score > 0:
                    current_sidon_count = count_sidon_subsets(current, current_score)
                else:
                    current_sidon_count = 0
                if best_score <= 3:
                    return best_set
            elif score == current_score:
                # For ties, prefer fewer Sidon subsets (harder to find)
                if score > 0:
                    cand_sidon_count = count_sidon_subsets(candidate, score)
                else:
                    cand_sidon_count = 0
                
                if cand_sidon_count < current_sidon_count:
                    current = candidate
                    current_sidon_count = cand_sidon_count
                elif random.random() < 0.3:
                    # Occasionally accept sideways moves for exploration
                    current = candidate
                    current_sidon_count = cand_sidon_count
    
    return best_set

# EVOLVE-BLOCK-END
