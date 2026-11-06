import random
import time
import sys

def construct():
    N = 32
    MAX_VAL = 512
    start_time = time.time()
    DEADLINE = 140.0  # Leave 10s margin for safety and cleanup
    
    # Helper to calculate score
    def score_set(S):
        S_set = set(S)
        diffs = set()
        # Iterate over pairs to find 3-term progressions
        # a, a+d, a+2d
        # Equivalent to: for x, y in S, if y-x = d and y+d in S
        # To avoid duplicates and ensure d > 0:
        # Let x < y. d = y - x. Check if y + d is in S.
        # Note: x, y, z form a progression if z - y == y - x => z = 2y - x.
        # So for each pair (x, y) with x < y, let z = 2y - x.
        # If z in S, then d = y - x is a valid difference.
        for x in S_set:
            for y in S_set:
                if x < y:
                    z = 2 * y - x
                    if z in S_set:
                        d = y - x
                        if d > 0:
                            diffs.add(d)
        return len(diffs)

    # Initial solution: Random
    current_S = set(random.sample(range(1, MAX_VAL + 1), N))
    current_score = score_set(current_S)
    
    best_S = set(current_S)
    best_score = current_score
    
    # Simulated Annealing parameters
    T = 10.0
    cooling_rate = 0.9995
    min_T = 0.01
    
    # Track consecutive non-improvements to trigger restarts or kicks if needed
    no_improve_count = 0
    max_no_improve = 5000
    
    while time.time() < DEADLINE:
        # Perform a move
        # Remove a random element
        elem_to_remove = random.choice(list(current_S))
        current_S.remove(elem_to_remove)
        
        # Add a random element not in S
        # To ensure we don't pick an element already in S, we can sample and retry
        # Or sample from the complement.
        # Since N=32 and MAX_VAL=512, complement is large.
        # Simple approach:
        while True:
            candidate = random.randint(1, MAX_VAL)
            if candidate not in current_S:
                break
        current_S.add(candidate)
        
        new_score = score_set(current_S)
        
        if new_score >= current_score:
            # Accept if better or equal
            current_S = set(current_S) # Ensure it's a new set object? No, we mutated in place.
            # Actually, we mutated current_S. 
            # We need to keep track of the set correctly.
            # Let's re-structure to keep current_S as a set.
            # The mutation above modified current_S.
            # If we reject, we need to revert.
            # It's safer to create a new set for the candidate.
            
            # Let's restart the loop logic properly with candidate sets to avoid state corruption on rejection.
            pass # This block is just thinking, the actual code below is cleaner.

    # Re-implementing the loop properly below in the final code block.
    # The logic above had a flaw in state management.
    
    # Corrected Simulated Annealing Loop
    current_S = set(random.sample(range(1, MAX_VAL + 1), N))
    current_score = score_set(current_S)
    best_S = set(current_S)
    best_score = current_score
    
    T = 10.0
    cooling_rate = 0.999
    min_T = 0.001
    
    while time.time() < DEADLINE:
        # Generate candidate
        candidate_S = set(current_S)
        
        # Remove one random element
        elem_to_remove = random.choice(list(candidate_S))
        candidate_S.remove(elem_to_remove)
        
        # Add one random element
        while True:
            candidate_elem = random.randint(1, MAX_VAL)
            if candidate_elem not in candidate_S:
                break
        candidate_S.add(candidate_elem)
        
        new_score = score_set(candidate_S)
        
        if new_score > current_score:
            current_S = candidate_S
            current_score = new_score
            if new_score > best_score:
                best_S = set(candidate_S)
                best_score = new_score
        else:
            # Accept with probability exp((new_score - current_score) / T)
            delta = new_score - current_score
            if delta < 0:
                prob = 2.718281828 ** (delta / T)
                if random.random() < prob:
                    current_S = candidate_S
                    current_score = new_score
        
        T *= cooling_rate
        if T < min_T:
            T = min_T
            
    return sorted(best_S)

# The above code has a slight inefficiency: creating a new set for every candidate.
# Given N=32, this is cheap.
# However, we can optimize score_set.
# Let's refine the score function and the loop for maximum speed.

def construct():
    N = 32
    MAX_VAL = 512
    start_time = time.time()
    DEADLINE = 145.0  # 150s limit, 5s margin
    
    def score_set(S):
        # S is a set of ints
        diffs = set()
        # Using list for iteration might be faster than set in some cases? 
        # No, set is O(1) lookup.
        # To avoid double counting pairs (x,y) and (y,x), we just iterate all pairs.
        # If x < y, z = 2y - x. If z in S, d = y - x is valid.
        # We don't need to check x > y because z = 2y - x would be < y, 
        # and we would have found it when iterating (z, y) if z < y.
        # Actually, if we iterate all pairs (x, y), we might count d multiple times?
        # No, we add to a set `diffs`. So duplicates are handled.
        # But we only need to find if *any* triplet exists for d.
        # So checking all pairs is correct.
        
        # Optimization: 
        # Iterate x in S
        # Iterate y in S
        # if x == y: continue
        # z = 2*y - x
        # if z in S:
        #    d = abs(y - x)
        #    diffs.add(d)
        
        # This is O(N^2).
        # 32^2 = 1024 operations.
        # Very fast.
        
        diffs = set()
        for x in S:
            for y in S:
                if x != y:
                    z = 2 * y - x
                    if z in S:
                        d = y - x
                        if d > 0:
                            diffs.add(d)
        return len(diffs)

    # Initial solution
    current_S = set(random.sample(range(1, MAX_VAL + 1), N))
    current_score = score_set(current_S)
    best_S = set(current_S)
    best_score = current_score
    
    T = 5.0
    cooling_rate = 0.999
    min_T = 0.01
    
    # To speed up, we can pre-allocate or use local variables
    while time.time() < DEADLINE:
        # Candidate generation
        # Copy current_S
        candidate_S = set(current_S)
        
        # Remove random
        idx = random.randint(0, N - 1)
        # Converting set to list every time is O(N). 
        # We can maintain a list representation?
        # But set is needed for O(1) lookup in score.
        # Let's stick to set for simplicity, N is small.
        
        # Efficient removal:
        # random.choice(list(candidate_S)) is O(N)
        # Since N=32, it's fine.
        
        elem_to_remove = random.choice(list(candidate_S))
        candidate_S.remove(elem_to_remove)
        
        # Add random
        while True:
            new_elem = random.randint(1, MAX_VAL)
            if new_elem not in candidate_S:
                break
        candidate_S.add(new_elem)
        
        new_score = score_set(candidate_S)
        
        if new_score > current_score:
            current_S = candidate_S
            current_score = new_score
            if new_score > best_score:
                best_S = set(candidate_S)
                best_score = new_score
        else:
            delta = new_score - current_score
            if delta < 0:
                # Accept with probability exp(delta/T)
                # To avoid overflow/underflow issues with very negative delta/T,
                # though T is small, delta is integer.
                # If delta is very negative, prob is ~0.
                prob = 2.718281828 ** (delta / T)
                if random.random() < prob:
                    current_S = candidate_S
                    current_score = new_score
        
        T *= cooling_rate
        if T < min_T:
            T = min_T
            
    return sorted(best_S)

# Final check:
# The score is the number of distinct d.
# The return type is a list of 32 distinct integers.
# The constraints are satisfied.

# One potential improvement: 
# Instead of pure random addition, we could bias towards values that complete progressions.
# But SA with random moves explores the space well.
# Given the time budget (150s), we can run many iterations.
# Let's estimate iterations:
# One iteration: 
# - set copy: ~us
# - list conversion: ~us
# - random choice: ~us
# - remove/add: ~us
# - score: ~500 hash lookups ~ us
# - total: ~10-20 us
# 150s / 20us = 7.5 million iterations.
# This is a very large search space exploration.
# It should find a good solution.

# To be safe against the "150s wall-clock" kill, I set DEADLINE to 145s.
# The function returns immediately after the loop.

# Let's refine the score function slightly for speed.
# We can break early if we find all possible d? 
# Max d is 255 (512/2 approx). 
# Actually max d is 255 if 1, 256, 511. 
# So max possible distinct d is 255.
# If len(diffs) == 255, we can stop. But unlikely.

# I will use the standard SA approach.

# One detail: `random.sample` for initialization.
# I will add a small "kick" mechanism if stuck for too long?
# No, cooling handles it. If T is low, it acts like hill climbing.
# If stuck, T might be too low.
# I can reset T if no improvement for X steps.

def construct():
    N = 32
    MAX_VAL = 512
    start_time = time.time()
    DEADLINE = 145.0
    
    def score_set(S):
        diffs = set()
        for x in S:
            for y in S:
                if x != y:
                    z = 2 * y - x
                    if z in S:
                        d = y - x
                        if d > 0:
                            diffs.add(d)
        return len(diffs)

    current_S = set(random.sample(range(1, MAX_VAL + 1), N))
    current_score = score_set(current_S)
    best_S = set(current_S)
    best_score = current_score
    
    T = 10.0
    cooling_rate = 0.9995
    min_T = 0.01
    no_improve = 0
    kick_threshold = 2000
    
    while time.time() < DEADLINE:
        candidate_S = set(current_S)
        
        # Remove
        elem_to_remove = random.choice(list(candidate_S))
        candidate_S.remove(elem_to_remove)
        
        # Add
        while True:
            new_elem = random.randint(1, MAX_VAL)
            if new_elem not in candidate_S:
                break
        candidate_S.add(new_elem)
        
        new_score = score_set(candidate_S)
        
        if new_score > current_score:
            current_S = candidate_S
            current_score = new_score
            no_improve = 0
            if new_score > best_score:
                best_S = set(candidate_S)
                best_score = new_score
        else:
            delta = new_score - current_score
            if delta < 0:
                prob = 2.718281828 ** (delta / T)
                if random.random() < prob:
                    current_S = candidate_S
                    current_score = new_score
                    no_improve = 0
                else:
                    no_improve += 1
            else:
                # Should not happen if logic is correct, but handle equal score
                current_S = candidate_S
                current_score = new_score
                no_improve += 1
        
        # Kick if stuck
        if no_improve > kick_threshold:
            T = max(T, 5.0)
            no_improve = 0
        else:
            T *= cooling_rate
            if T < min_T:
                T = min_T
                
    return sorted(best_S)
