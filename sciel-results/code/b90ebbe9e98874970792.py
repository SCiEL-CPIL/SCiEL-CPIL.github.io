import time
import random
import math

def calculate_score(U):
    if not U or 0 not in U:
        return 0.0
    U_set = set(U)
    if len(U_set) != len(U):
        return 0.0
    
    max_u = max(U)
    if max_u == 0:
        return 1.0 # log(1)/log(1) is undefined, but limit is 1? 
                   # Actually for U={0}, |U-U|=1, |U+U|=1, ratio 1, log 0. Score 1.
    
    sums = set()
    diffs = set()
    
    # Optimize: use list comprehension or loops
    # N is likely small (< 50) for best scores?
    # If N is large, M is likely large, score drops.
    # Let's assume N < 100.
    
    for u1 in U_set:
        for u2 in U_set:
            sums.add(u1 + u2)
            diffs.add(u1 - u2)
            
    sum_size = len(sums)
    diff_size = len(diffs)
    
    if sum_size == 0:
        return 0.0
        
    ratio = diff_size / sum_size
    if ratio <= 1.0:
        # Score is 1 + negative/zero. 
        # We want to maximize, so > 1 is better.
        # But we must return the actual value.
        pass
        
    denom = math.log(2 * max_u + 1)
    if denom == 0:
        return 1.0
        
    score = 1.0 + math.log(ratio) / denom
    return score

def construct():
    start_time = time.time()
    deadline = start_time + 100.0  # 20s safety margin
    
    # Initial candidates
    candidates = [
        [0, 1, 3, 7],
        [0, 1, 3],
        [0, 1, 2, 4, 8],
        [0, 1, 4, 10],
        [0, 1, 3, 7, 15],
        [0, 1, 2, 5, 11],
    ]
    
    # Add some random small sets
    for _ in range(20):
        n = random.randint(3, 8)
        max_val = random.randint(5, 30)
        # Ensure 0 is included
        s = {0}
        while len(s) < n:
            s.add(random.randint(1, max_val))
        candidates.append(sorted(list(s)))
        
    best_set = candidates[0]
    best_score = calculate_score(best_set)
    
    # Local Search
    # We will iterate through moves
    # To avoid infinite loops, we track visited states or just rely on time
    # Given the small search space, simple hill climbing might work.
    
    current_set = list(best_set)
    current_score = best_score
    
    # We need a way to generate neighbors efficiently
    # Neighbors:
    # 1. Add x
    # 2. Remove x
    # 3. Replace x with y
    
    # Since M affects the denominator strongly, we should be careful with adding large numbers.
    # Let's limit the search range for new numbers.
    
    while time.time() < deadline:
        improved = False
        current_set_sorted = sorted(current_set)
        max_u = max(current_set_sorted) if current_set_sorted else 0
        
        # 1. Try adding a new element
        # Range for new element: 1 to 2*max_u + 5
        # We don't want to go too far out.
        max_search_val = 2 * max_u + 10 if max_u > 0 else 20
        
        # Candidate additions
        add_candidates = []
        for x in range(1, max_search_val + 1):
            if x not in current_set:
                add_candidates.append(x)
                
        # To keep it fast, sample if too many
        if len(add_candidates) > 50:
            add_candidates = random.sample(add_candidates, 50)
            
        for x in add_candidates:
            new_set = current_set + [x]
            s = calculate_score(new_set)
            if s > current_score + 1e-9:
                current_set = new_set
                current_score = s
                if s > best_score:
                    best_set = list(current_set)
                    best_score = s
                improved = True
                break # Restart loop with new state
        
        if improved:
            continue
            
        # 2. Try removing an element
        # Never remove 0
        remove_candidates = [x for x in current_set if x != 0]
        if remove_candidates:
            if len(remove_candidates) > 10:
                remove_candidates = random.sample(remove_candidates, 10)
            
            for x in remove_candidates:
                new_set = [y for y in current_set if y != x]
                s = calculate_score(new_set)
                if s > current_score + 1e-9:
                    current_set = new_set
                    current_score = s
                    if s > best_score:
                        best_set = list(current_set)
                        best_score = s
                    improved = True
                    break
        
        if improved:
            continue
        
        # 3. Try swapping
        # Pick a random element (not 0) and replace with a random new value
        swap_candidates = [x for x in current_set if x != 0]
        if swap_candidates:
            x = random.choice(swap_candidates)
            # Generate new value
            new_val = random.randint(1, 2 * max_u + 10)
            if new_val != x and new_val not in current_set:
                new_set = [y if y != x else new_val for y in current_set]
                s = calculate_score(new_set)
                if s > current_score + 1e-9:
                    current_set = new_set
                    current_score = s
                    if s > best_score:
                        best_set = list(current_set)
                        best_score = s
                    improved = True

        if not improved:
            # Stuck. Perturb.
            # Remove a random non-zero element and add a random new one
            if len(current_set) > 1:
                x = random.choice([y for y in current_set if y != 0])
                current_set.remove(x)
                new_val = random.randint(1, 2 * max(current_set) + 20)
                current_set.append(new_val)
                current_score = calculate_score(current_set)
                if current_score > best_score:
                    best_set = list(current_set)
                    best_score = current_score
            else:
                # Add a random element
                new_val = random.randint(1, 20)
                if new_val not in current_set:
                    current_set.append(new_val)
                    current_score = calculate_score(current_set)
                    if current_score > best_score:
                        best_set = list(current_set)
                        best_score = current_score

    return sorted(best_set)
