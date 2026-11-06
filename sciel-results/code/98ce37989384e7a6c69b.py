import time
import random
from itertools import combinations

def is_sidon_with_diagonals(subset):
    """
    Check if a subset of integers has all unordered pair sums (including diagonals) unique.
    """
    sums = set()
    k = len(subset)
    for i in range(k):
        for j in range(i, k):
            s = subset[i] + subset[j]
            if s in sums:
                return False
            sums.add(s)
    return True

def max_sidon_size(candidate_set):
    """
    Find the size of the largest subset of candidate_set that is Sidon with diagonals.
    candidate_set is a list of 12 distinct integers.
    """
    n = len(candidate_set)
    # Optimization: Check from largest k down
    for k in range(n, 0, -1):
        found = False
        for comb in combinations(candidate_set, k):
            if is_sidon_with_diagonals(comb):
                found = True
                break
        if found:
            return k
    return 0

def construct():
    start_time = time.time()
    deadline = 88.0  # Safety margin
    
    # Initial candidates with high collision density (Arithmetic Progressions)
    # These are likely to have small max Sidon subsets.
    candidates = [
        list(range(12)),
        [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22],
        [0, 3, 6, 9, 12, 15, 18, 21, 24, 27, 30, 33],
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 20],
        [0, 1, 2, 3, 4, 5, 16, 17, 18, 19, 20, 21],
    ]
    
    best_set = None
    best_score = 13  # Max possible is 12, so 13 is infinity
    
    # Evaluate initial candidates
    for cand in candidates:
        if len(set(cand)) != 12:
            continue
        if any(x < 0 or x > 128 for x in cand):
            continue
        score = max_sidon_size(cand)
        if score < best_score:
            best_score = score
            best_set = cand[:]
            
    if best_set is None:
        best_set = list(range(12))
        best_score = max_sidon_size(best_set)
        
    # If we already found a score of 3, it's likely optimal. 
    # (Theoretical min for n=12 is debated, but 3 is the target for improvement)
    # If best_score is 4, we try to find 3.
    
    current_set = best_set[:]
    current_score = best_score
    
    # Simulated Annealing / Local Search
    # We want to minimize the max Sidon size.
    # State: current_set
    # Objective: max_sidon_size(current_set)
    
    temperature = 1.0
    cooling_rate = 0.999
    
    while time.time() < deadline:
        # Create a new candidate by perturbing current_set
        new_set = current_set[:]
        
        # Randomly replace 1 or 2 elements
        num_replaces = random.choice([1, 2])
        indices = random.sample(range(12), num_replaces)
        
        for idx in indices:
            # Strategy: Try to create collisions.
            # Pick a value close to existing ones or random.
            if random.random() < 0.5:
                # Near existing
                ref = random.choice(current_set)
                new_val = ref + random.randint(-5, 5)
            else:
                new_val = random.randint(0, 128)
                
            # Ensure distinctness
            attempts = 0
            while (new_val in new_set or new_val < 0 or new_val > 128) and attempts < 10:
                new_val = random.randint(0, 128)
                attempts += 1
            
            if new_val in new_set or new_val < 0 or new_val > 128:
                # Fallback: just pick a random unused value
                available = [v for v in range(129) if v not in new_set]
                if available:
                    new_val = random.choice(available)
                else:
                    continue # Should not happen
            
            new_set[idx] = new_val
        
        if len(set(new_set)) != 12:
            continue
            
        new_set.sort()
        new_score = max_sidon_size(new_set)
        
        # Acceptance criteria
        if new_score < current_score:
            # Always accept improvement
            current_set = new_set[:]
            current_score = new_score
            
            if current_score < best_score:
                best_score = current_score
                best_set = new_set[:]
                
            if best_score <= 3:
                break # Can't do better than 3 (theoretically)
                
        else:
            # Accept worse or equal with probability exp(-delta/T)
            delta = new_score - current_score
            if delta == 0:
                # Accept equal to explore
                if random.random() < 0.5:
                    current_set = new_set[:]
            else:
                prob = random.random()
                # Simple annealing
                if prob < (1.0 / (1.0 + delta)): # Simplified acceptance
                    current_set = new_set[:]
        
        # Cool down
        temperature *= cooling_rate
        if temperature < 0.01:
            temperature = 0.01

    return best_set
