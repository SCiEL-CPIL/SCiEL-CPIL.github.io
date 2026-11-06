import itertools
import random
import sys

def is_sidon(subset):
    """
    Check if a subset is a Sidon set.
    A Sidon set requires all unordered pair sums (including diagonals a+a) to be unique.
    """
    sums = set()
    for i in range(len(subset)):
        for j in range(i, len(subset)):
            s = subset[i] + subset[j]
            if s in sums:
                return False
            sums.add(s)
    return True

def has_sidon_subset_of_size_k(S, k):
    """
    Check if S contains a Sidon subset of size k using backtracking.
    S should be sorted.
    """
    n = len(S)
    if k == 0:
        return True
    if k > n:
        return False
    
    # Optimization: if k is large, checking all combinations is bad.
    # Backtracking with pruning is better.
    
    # We will build the subset one element at a time.
    # We maintain the set of sums generated so far.
    # We iterate through S, choosing elements to add.
    
    def backtrack(start_idx, current_subset, current_sums):
        if len(current_subset) == k:
            return True
        
        # Pruning: if not enough elements left to reach k
        if n - start_idx < k - len(current_subset):
            return False
            
        for i in range(start_idx, n):
            val = S[i]
            # Check if adding val to current_subset creates any sum collision
            valid = True
            new_sums = list(current_sums) # We need to add sums of val with all in current_subset and val+val
            for existing_val in current_subset:
                s = val + existing_val
                if s in current_sums:
                    valid = False
                    break
            if valid:
                s = val + val
                if s in current_sums:
                    valid = False
            
            if valid:
                # Recurse
                # We need to pass the updated sums. 
                # To avoid copying the set every time, we can use a set and add/remove, 
                # but for small k (<=12) and small S, copying is fine or we can use a list and check.
                # Let's use a set for O(1) lookup.
                new_sums_set = set(current_sums)
                for existing_val in current_subset:
                    new_sums_set.add(val + existing_val)
                new_sums_set.add(val + val)
                
                if backtrack(i + 1, current_subset + [val], new_sums_set):
                    return True
        return False

    return backtrack(0, [], set())

def max_sidon_size(S):
    """
    Find the size of the largest Sidon subset of S.
    S is a list of distinct integers.
    """
    n = len(S)
    S_sorted = sorted(S)
    # Check from largest possible size down to 1
    for k in range(n, 0, -1):
        if has_sidon_subset_of_size_k(S_sorted, k):
            return k
    return 0

def construct():
    # We want to minimize the largest Sidon subset size.
    # The minimum possible is likely 3 or 4.
    # We try to find a set of 12 integers where the largest Sidon subset has size 3.
    
    best_set = None
    best_size = 12
    
    # Strategy: Randomized Local Search / Simulated Annealing
    # Start with a dense set (like AP) and try to break Sidon structures by creating more sum collisions.
    
    rng = random.Random(42)
    
    # Initial candidate: Arithmetic Progression is very "bad" for Sidon (lots of collisions)
    # But we verified range(12) has a Sidon subset of size 4.
    # Let's try to mutate it to destroy all size-4 Sidon subsets.
    
    S = list(range(12))
    current_size = max_sidon_size(S)
    best_size = current_size
    best_set = S.copy()
    
    # If we already found 3, we are done.
    if best_size == 3:
        return best_set

    # Local Search
    # We want to minimize max_sidon_size.
    # Move: Replace one element with another random value in [0, 128].
    # Or swap two elements.
    
    max_iterations = 5000
    for _ in range(max_iterations):
        if best_size == 3:
            break
            
        # Pick a random index to change
        idx = rng.randint(0, 11)
        old_val = S[idx]
        
        # Pick a new random value
        # Bias towards values that might create collisions with existing elements?
        # Just random for now, maybe range [0, 20] is dense enough.
        new_val = rng.randint(0, 20)
        
        if new_val == old_val:
            continue
            
        # Check distinctness
        if new_val in S:
            continue
            
        S[idx] = new_val
        new_size = max_sidon_size(S)
        
        if new_size < best_size:
            best_size = new_size
            best_set = S.copy()
            if best_size == 3:
                break
        else:
            # Rollback if not improved (or accept with small prob for SA, but let's stick to hill climbing first)
            S[idx] = old_val

    # If we didn't find 3, try a few random restarts with different initial structures
    for restart in range(10):
        # Random dense set
        S = sorted(rng.sample(range(0, 30), 12))
        current_size = max_sidon_size(S)
        if current_size < best_size:
            best_size = current_size
            best_set = S.copy()
            if best_size == 3:
                break
                
        # Local search from this start
        for _ in range(1000):
            if best_size == 3:
                break
            idx = rng.randint(0, 11)
            old_val = S[idx]
            new_val = rng.randint(0, 30)
            if new_val == old_val or new_val in S:
                continue
            S[idx] = new_val
            new_size = max_sidon_size(S)
            if new_size < best_size:
                best_size = new_size
                best_set = S.copy()
                if best_size == 3:
                    break
            else:
                S[idx] = old_val

    return best_set
