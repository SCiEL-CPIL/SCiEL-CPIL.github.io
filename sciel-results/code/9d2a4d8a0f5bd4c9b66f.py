"""Optimized solver for minimizing max Sidon subset size in a 12-element set."""
import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 55  # Safety margin before 60s kill

    def max_sidon_size(arr):
        """Exact largest Sidon subset size using backtracking."""
        n = len(arr)
        arr_sorted = sorted(arr)
        
        best_size = 0
        
        def backtrack(idx, current, current_sums):
            nonlocal best_size
            k = len(current)
            if k > best_size:
                best_size = k
            
            if idx >= n:
                return
            
            # Pruning: can't beat best_size
            if k + (n - idx) <= best_size:
                return
            
            for i in range(idx, n):
                elem = arr_sorted[i]
                # Check if adding elem keeps Sidon property
                # New sums: elem+elem, and elem+current[j] for each j in current
                new_sums = [elem + elem]
                for j in current:
                    new_sums.append(elem + j)
                
                # Check for collisions within new_sums
                if len(set(new_sums)) != len(new_sums):
                    continue
                # Check collisions with existing sums
                if any(s in current_sums for s in new_sums):
                    continue
                
                backtrack(i + 1, current + [elem], current_sums | set(new_sums))
        
        backtrack(0, [], set())
        return best_size

    def evaluate(arr):
        """Returns the max Sidon size of the given array."""
        return max_sidon_size(arr)

    # Phase 1: Generate diverse starting points
    random.seed(42)
    candidates = []
    
    # 1. The seed itself
    candidates.append(list(range(12)))
    
    # 2. Small range random sets (high collision probability)
    # Smaller range -> more sum collisions -> smaller max Sidon subset
    for _ in range(20):
        # Try ranges up to 40
        max_val = random.randint(15, 40)
        # Ensure we can pick 12 distinct ints
        if max_val < 11:
            continue
        cand = random.sample(range(max_val + 1), 12)
        candidates.append(cand)
        
    # 3. Structured sets
    candidates.append(list(range(12))) # 0..11
    candidates.append(list(range(0, 24, 2))) # 0,2,4..22
    candidates.append([0, 1, 2, 3, 4, 5, 10, 11, 12, 13, 14, 15])
    candidates.append([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11])
    candidates.append([0, 2, 4, 6, 8, 10, 20, 22, 24, 26, 28, 30])
    candidates.append([0, 1, 3, 4, 6, 7, 10, 11, 13, 14, 16, 17])
    
    # 4. Symmetric sets around a center
    for center in [10, 20, 30]:
        # Pick 6 pairs (c-d, c+d)
        offsets = random.sample(range(1, 30), 6)
        cand = []
        for off in offsets:
            cand.append(center - off)
            cand.append(center + off)
        if len(set(cand)) == 12 and all(0 <= x <= 128 for x in cand):
            candidates.append(cand)

    # Phase 2: Local Search
    best_overall = None
    best_overall_score = 12 # Initialize to max possible
    
    # First, evaluate all initial candidates to find a good starting point
    for cand in candidates:
        if time.time() > deadline:
            break
        if len(set(cand)) != 12 or any(x < 0 or x > 128 for x in cand):
            continue
        
        score = evaluate(cand)
        if score < best_overall_score:
            best_overall_score = score
            best_overall = cand[:]
            
        # If we found size 3, we are very good. If size 2, excellent.
        if score <= 3:
            # We can still try to improve to 2, but 3 is a huge jump from 4
            pass

    if best_overall is None:
        best_overall = list(range(12))
        best_overall_score = evaluate(best_overall)
        
    current = best_overall[:]
    current_score = best_overall_score
    
    # Local Search Loop
    no_improve_count = 0
    max_no_improve = 500
    
    while time.time() < deadline - 5: # Keep 5s buffer
        if no_improve_count > max_no_improve:
            # Restart with a new random candidate
            new_cand = random.sample(range(129), 12)
            new_score = evaluate(new_cand)
            if new_score < current_score:
                current = new_cand
                current_score = new_score
                no_improve_count = 0
            else:
                # Try a small perturbation of the best found so far
                current = best_overall[:]
                current_score = best_overall_score
                no_improve_count = 0
            continue
            
        # Pick a random index to replace
        idx = random.randint(0, 11)
        old_val = current[idx]
        
        # Strategy for new value:
        # 1. Try values close to existing ones to maintain density
        # 2. Try random values in a small range
        # 3. Try random values in [0, 128]
        
        if random.random() < 0.5:
            # Local move: value near one of the existing elements
            anchor = random.choice(current)
            new_val = anchor + random.randint(-5, 5)
        else:
            # Global move: random in small range to keep density
            new_val = random.randint(0, 40)
            
        if new_val < 0 or new_val > 128:
            continue
        if new_val in current and new_val != old_val:
            continue
            
        trial = current[:]
        trial[idx] = new_val
        trial.sort()
        
        trial_score = evaluate(trial)
        
        if trial_score < current_score:
            current = trial
            current_score = trial_score
            no_improve_count = 0
            
            if current_score < best_overall_score:
                best_overall_score = current_score
                best_overall = current[:]
                
            # If we hit 3, keep searching for 2
            # If we hit 2, we can stop or keep polishing
        else:
            no_improve_count += 1
            
    return best_overall

# EVOLVE-BLOCK-END
