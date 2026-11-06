"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random
import numpy as np

def _check_triple_sums_unique(arr):
    """Check if all unordered triples (with replacement) have unique sums."""
    n = len(arr)
    if n < 3:
        return True
    # For efficiency, we can use a set of sums
    # But for large n, this is O(n^3) which is too slow.
    # We need a smarter check or just rely on the construction.
    # Actually, for the greedy approach, we check incrementally.
    # This function is only used for final verification or small sets.
    sums = set()
    for i in range(n):
        for j in range(i, n):
            for k in range(j, n):
                s = arr[i] + arr[j] + arr[k]
                if s in sums:
                    return False
                sums.add(s)
    return True

def construct():
    n_max = 4096
    target_size = 80
    
    # We will use a randomized greedy approach.
    # The key insight is that if we pick numbers that are "spread out" enough,
    # we avoid collisions. 
    # A simple strategy: pick random numbers, check if they cause a collision
    # with existing picks. To make it fast, we maintain a set of all possible
    # triple sums involving at least one new element? No, that's complex.
    
    # Let's try a different approach: 
    # 1. Randomly shuffle candidates.
    # 2. Iterate through candidates.
    # 3. For each candidate, check if adding it to the current set creates a duplicate sum.
    #    To check efficiently, we pre-compute the sums of all triples in the current set?
    #    That's O(k^3) for each addition, where k is current size. 
    #    Total O(N * K^3) which might be too slow if N is large and K is 80.
    #    4096 * 80^3 = 4096 * 512000 = 2 billion. Too slow.
    
    # We need a faster collision check.
    # When adding a new number x to a set S, the new sums are:
    # x + x + x
    # x + x + s for s in S
    # x + s1 + s2 for s1, s2 in S, s1 <= s2
    # We need to check if any of these are already in the set of existing sums.
    # We maintain a set `existing_sums` of all triple sums from S.
    # For new x, we generate new_sums and check intersection with existing_sums.
    # If no intersection, add x to S, and update existing_sums with new_sums.
    
    # Complexity per addition: O(|S|^2) to generate new sums and check against existing_sums (which is a set, so O(1) lookup).
    # Total complexity: O(K^2 * N) where K is final size (80) and N is number of candidates tried.
    # If we try all 4096 candidates, 80^2 * 4096 = 6400 * 4096 = 26 million operations. This is fast.
    
    # Strategy:
    # 1. Initialize S = [], existing_sums = set()
    # 2. Shuffle range(1, 4097)
    # 3. For each x in shuffled list:
    #    a. Generate all new triple sums involving x and elements of S.
    #    b. If any new sum is in existing_sums, skip x.
    #    c. Else, add x to S, add new sums to existing_sums.
    #    d. If len(S) == target_size, break.
    
    # To improve results, we can run this multiple times with different random seeds/orderings
    # and keep the best (largest) set found.
    
    best_set = []
    
    # Run multiple trials with different strategies
    num_trials = 50
    
    for trial in range(num_trials):
        candidates = list(range(1, n_max + 1))
        
        # Vary the ordering strategy based on trial index
        if trial % 5 == 0:
            random.seed(trial)
            random.shuffle(candidates)
        elif trial % 5 == 1:
            # Descending order
            candidates.sort(reverse=True)
        elif trial % 5 == 2:
            # Ascending order
            candidates.sort()
        elif trial % 5 == 3:
            # Random with bias towards larger numbers
            random.seed(trial + 100)
            # Sort by random key that favors larger numbers
            candidates.sort(key=lambda x: random.random() * x)
        else:
            # Pure random with different seed
            random.seed(trial + 200)
            random.shuffle(candidates)
        
        S = []
        existing_sums = set()
        
        # Optimization: local variable references
        S_local = S
        sums_local = existing_sums
        
        for x in candidates:
            if len(S_local) >= target_size:
                break
            
            # Generate new sums involving x
            # Using list comprehension or loop
            new_sums = set()
            new_sums.add(3 * x)
            
            # x, x, s
            for s in S_local:
                new_sums.add(2 * x + s)
            
            # x, s1, s2
            # S_local is small, so nested loop is fine
            for i in range(len(S_local)):
                si = S_local[i]
                for j in range(i, len(S_local)):
                    new_sums.add(x + si + S_local[j])
            
            # Check for collision
            if new_sums & sums_local:
                continue
            
            # No collision, add x
            S_local.append(x)
            sums_local.update(new_sums)
        
        if len(S_local) > len(best_set):
            best_set = S_local[:]
        
        # Early stopping if we found 80
        if len(best_set) == target_size:
            break
    
    return best_set
# EVOLVE-BLOCK-END
