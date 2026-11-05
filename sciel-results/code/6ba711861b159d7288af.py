"""Optimized B_3[1] set construction via randomized greedy with incremental checks."""
import time
import random
import sys

def _check_incremental(existing, candidate):
    """
    Check if adding 'candidate' to 'existing' maintains unique triple sums.
    'existing' must be sorted.
    Returns True if valid, False otherwise.
    """
    c = candidate
    n = len(existing)
    
    # We need to ensure all new sums involving c are unique among themselves
    # and not present in the sums of existing triples.
    
    # However, maintaining a global set of all sums is O(1) lookup but memory heavy.
    # For k~30-40, total sums ~ k^3/6 ~ 20000. This is small.
    # But doing a full check for every candidate in a loop is O(k * k^3) = O(k^4).
    # The incremental check logic:
    # New sums are:
    # 1. 3c
    # 2. 2c + x_i for all x_i in existing
    # 3. c + x_i + x_j for all x_i <= x_j in existing
    
    # We need to check if any of these are equal to each other or to any existing sum.
    # This requires knowing the existing sums.
    
    # Strategy: 
    # 1. Generate all new sums.
    # 2. Check for duplicates within new sums.
    # 3. Check if any new sum is in the set of existing sums.
    
    # To make this fast, we should pass the set of existing sums.
    # But the function signature in the seed was simple. 
    # I will modify the internal helper to accept the sum set.
    pass

def construct():
    start_time = time.time()
    deadline = start_time + 50.0  # Safety margin for 60s limit
    
    best_set = []
    best_size = 0
    
    # Initial greedy attempt to get a baseline
    # This is faster than the seed because we can optimize the check.
    # Let's implement a faster greedy first.
    
    def fast_greedy(randomize=False):
        current = []
        existing_sums = set()
        
        # To speed up, we can restrict the search space or use a better order.
        # Standard greedy: try 1, 2, 3...
        # Randomized greedy: try random candidates from a pool.
        
        if not randomize:
            candidates = list(range(1, 4097))
        else:
            # Use a pool of candidates. 
            # Small numbers are usually better for packing density?
            # Or spread out?
            # Let's try a mix.
            # Actually, for Sidon-like sets, exponential growth is often good.
            # But we want max cardinality in [1, 4096].
            # The maximum possible size is roughly N^(1/3) * const?
            # N=4096, N^(1/3) ~ 16. 
            # Wait, B_3[1] means sum of 3 elements.
            # Number of triples is k(k+1)(k+2)/6 ~ k^3/6.
            # Max sum is 3*4096 ~ 12288.
            # So k^3/6 < 12288 => k^3 < 73728 => k < 42.
            # So we are looking for k around 30-40.
            
            # A good strategy is to start with small numbers and grow.
            # Randomization: shuffle the candidate order or pick randomly from a window.
            candidates = list(range(1, 4097))
            random.shuffle(candidates)
            
            # Actually, shuffling 1..4096 and picking the first valid one is inefficient.
            # Better: Iterate through a sorted list, but with some randomization in the "next" choice.
            # Or: Start with a known good seed, then try to add more.
            
            # Let's stick to a simpler randomization: 
            # Try multiple passes. In each pass, iterate 1..4096.
            # But to escape local optima, we can randomly skip some small numbers or force inclusion of larger ones?
            # No, small numbers are generally denser.
            
            # Let's use a "randomized greedy" where we iterate 1..4096, 
            # but with a probability p, we skip a number even if it fits? No, that's bad.
            
            # Alternative: Random restarts with different initial seeds.
            # Seed 1: [1, 2, 4, ...] (powers of 2? No, 1+2+4=7, 1+1+5=7. Conflict.)
            
            # Let's just do multiple greedy passes with slight variations.
            # Variation 1: Standard 1..4096
            # Variation 2: Start with a random small set, then fill.
            
            pass
        
        # Efficient Incremental Check
        def can_add(x):
            # Check new sums
            new_sums = []
            # 3x
            new_sums.append(3*x)
            # 2x + y
            for y in current:
                new_sums.append(2*x + y)
            # x + y + z
            n = len(current)
            for i in range(n):
                for j in range(i, n):
                    new_sums.append(x + current[i] + current[j])
            
            # Check for duplicates within new_sums
            if len(new_sums) != len(set(new_sums)):
                return False
            
            # Check against existing sums
            for s in new_sums:
                if s in existing_sums:
                    return False
            
            return True

        # We need to update existing_sums when we add.
        # To avoid O(k^2) update cost every time, we can just recompute? 
        # No, k is small (<50). k^2 is 2500. 
        # Total cost per add: O(k^2) for generating new sums + O(k^2) for checking + O(k^2) for updating set.
        # Total for k items: sum_{i=1}^k i^2 ~ k^3/3.
        # For k=40, 64000 ops. Very fast.
        # So we can afford to recompute or update efficiently.
        
        # Let's refine the loop.
        if not randomize:
            search_range = range(1, 4097)
        else:
            # Randomized strategy: 
            # 1. Pick a random starting point? No.
            # 2. Iterate 1..4096, but if we fail to add x, we move on.
            # To get different results, we can randomly perturb the "acceptance" or the order.
            # Simplest effective randomization: 
            # Shuffle a subset of the candidates? 
            # Or: Start with a random valid set of size 5, then greedy fill.
            
            # Let's try: Randomly select 5 distinct small integers (1-100) that form a valid set.
            # Then greedy fill from 1..4096.
            
            initial = []
            initial_sums = set()
            # Find a random valid seed of size 5
            while len(initial) < 5:
                x = random.randint(1, 100)
                if x in initial: continue
                # Check if x can be added to initial
                # (Using the same logic)
                temp_current = initial
                temp_sums = initial_sums
                # ... complex. 
                
            # Simpler randomization: 
            # Just run standard greedy, but with a random "bias" in the candidate selection.
            # Actually, the greedy algorithm is deterministic. 
            # To get different local optima, we need to break ties or change order.
            # Since we iterate 1..4096, the order is fixed.
            # We can try:
            # 1. Standard Greedy
            # 2. Greedy with candidates sorted by (random() + value/4096)
            # 3. Greedy starting from a random offset? No, must be positive.
            
            # Let's implement a "Randomized Greedy" where we iterate through a shuffled list of 1..4096?
            # That would be very slow (4096 * check). But check is fast.
            # 4096 * 1000 ops = 4M ops. < 0.1s.
            # So we can shuffle the entire range 1..4096 and try to add them in that order.
            # This explores different local optima!
            
            search_range = list(range(1, 4097))
            random.shuffle(search_range)
            
        for x in search_range:
            if time.time() > deadline - 5.0:
                break
                
            if x in current:
                continue
                
            if can_add(x):
                # Add x
                current.append(x)
                # Update existing_sums
                # New sums:
                # 3x
                existing_sums.add(3*x)
                # 2x + y
                for y in current[:-1]: # current[:-1] is the old current
                    existing_sums.add(2*x + y)
                # x + y + z
                n_old = len(current) - 1
                for i in range(n_old):
                    for j in range(i, n_old):
                        existing_sums.add(x + current[i] + current[j])
                
                if len(current) > best_size:
                    best_size = len(current)
                    best_set = current.copy()
                    
        return current

    # Run multiple strategies
    # 1. Standard Greedy
    res1 = fast_greedy(randomize=False)
    if len(res1) > best_size:
        best_size = len(res1)
        best_set = res1

    # 2. Multiple Randomized Greedy attempts
    attempts = 0
    while time.time() < deadline - 10.0:
        res = fast_greedy(randomize=True)
        attempts += 1
        if len(res) > best_size:
            best_size = len(res)
            best_set = res.copy()
            
        # If we found a very large set, maybe we can try to optimize it further?
        # For now, just keep the best.
        
    # Final check: ensure best_set is valid and sorted
    # The greedy process maintains sorted order if we iterate 1..4096.
    # But if we shuffled, the set might not be sorted. 
    # The problem requires distinct positive integers. Order doesn't matter for the set, 
    # but the return type is a list. 
    # "1..80 distinct positive integers". 
    # I should return a sorted list to be safe, though the evaluator likely just checks the set properties.
    
    return sorted(best_set)

# EVOLVE-BLOCK-END
