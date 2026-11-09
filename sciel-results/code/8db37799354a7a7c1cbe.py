"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random

# EVOLVE-BLOCK-START

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 120s limit
    
    N = 32
    MAX_VAL = 512
    
    def list_to_mask(lst):
        mask = 0
        for v in lst:
            mask |= (1 << v)
        return mask

    def score_mask(mask):
        """
        Calculate the number of distinct d > 0 such that a, a+d, a+2d are all in the set.
        """
        valid_diffs = set()
        # d can range from 1 to 255 (since a+2d <= 512)
        for d in range(1, 256):
            if (mask & (mask >> d) & (mask >> 2 * d)) != 0:
                valid_diffs.add(d)
        return len(valid_diffs)

    def score_list(lst):
        mask = list_to_mask(lst)
        return score_mask(mask)

    def greedy_improve(lst, max_time=110.0):
        """
        Greedy repair: iteratively try to replace an element to improve the score.
        Uses bitmask for fast evaluation.
        """
        current_lst = lst[:]
        current_mask = list_to_mask(current_lst)
        current_score = score_mask(current_mask)
        best_lst = current_lst[:]
        best_score = current_score
        
        # Track which elements have been "tried" recently to avoid redundant checks
        # or just iterate through all pairs (i, new_val)
        
        # To make it faster, we can pre-calculate the set of values in the list
        current_set = set(current_lst)
        
        no_improvement_count = 0
        max_no_improvement = 5000 # Reset counter if we do many checks
        
        while time.time() < max_time:
            improved = False
            # Shuffle indices to randomize the order of attempts
            indices = list(range(N))
            random.shuffle(indices)
            
            for i in indices:
                if time.time() > max_time:
                    break
                    
                old_val = current_lst[i]
                
                # Try replacing current_lst[i] with a random new value
                # We can bias towards values that might complete progressions
                # But random sampling is a good baseline for the "repair" phase
                
                # Optimization: Instead of trying all 512 values, try a subset
                # Or try values that are "close" to existing ones to form small d
                # Let's try a mix: some random, some structured
                
                # Structured candidates:
                # 1. Values that form a 3-term AP with two existing elements
                # 2. Random values
                
                # To keep it simple and fast, we'll just do random replacement first.
                # If we want to be smarter, we can check if replacing old_val with new_val
                # increases the score.
                
                # To speed up, we can limit the number of random trials per index
                for _ in range(20):
                    if time.time() > max_time:
                        break
                    
                    new_val = random.randint(1, MAX_VAL)
                    if new_val in current_set:
                        continue
                    if new_val == old_val:
                        continue
                    
                    # Tentatively swap
                    current_set.remove(old_val)
                    current_set.add(new_val)
                    current_lst[i] = new_val
                    
                    new_mask = list_to_mask(current_lst)
                    new_score = score_mask(new_mask)
                    
                    if new_score > best_score:
                        best_score = new_score
                        best_lst = current_lst[:]
                        current_score = new_score
                        improved = True
                        no_improvement_count = 0
                    else:
                        # Revert
                        current_lst[i] = old_val
                        current_set.remove(new_val)
                        current_set.add(old_val)
                        no_improvement_count += 1
            
            if not improved and no_improvement_count > max_no_improvement:
                # If we've tried many replacements without improvement, 
                # we might be stuck in a local optimum.
                # We can try a "kick": randomly change 2-3 elements to escape.
                kick_size = random.randint(2, 3)
                kick_indices = random.sample(range(N), kick_size)
                for k in kick_indices:
                    old_val = current_lst[k]
                    new_val = random.randint(1, MAX_VAL)
                    while new_val in current_set or new_val == old_val:
                        new_val = random.randint(1, MAX_VAL)
                    current_set.remove(old_val)
                    current_set.add(new_val)
                    current_lst[k] = new_val
                
                # Update mask and score
                current_mask = list_to_mask(current_lst)
                current_score = score_mask(current_mask)
                
                # If the kick improved it, great. If not, we continue from this new state
                # (simulated annealing-like behavior without explicit temperature)
                no_improvement_count = 0

        return best_lst, best_score

    # Strategy:
    # 1. Generate a diverse set of initial candidates using algebraic/structured methods.
    # 2. Run greedy_improve on each.
    # 3. Keep the best.
    
    best_overall_lst = None
    best_overall_score = 0
    
    # Initial candidate 1: Simple range (baseline)
    init1 = list(range(1, 33))
    s1 = score_list(init1)
    best_overall_score = s1
    best_overall_lst = init1[:]
    
    # Initial candidate 2: Sparse AP
    # To get 32 elements in 1..512, average spacing is ~16.
    # Let's try a base AP with small step, then perturb.
    init2 = []
    base = random.randint(1, 10)
    step = random.randint(15, 18) # 32 * 16 = 512
    for i in range(N):
        v = base + i * step
        if v <= MAX_VAL:
            init2.append(v)
    while len(init2) < N:
        v = random.randint(1, MAX_VAL)
        if v not in init2:
            init2.append(v)
    init2 = init2[:N]
    
    # Initial candidate 3: Two interleaved APs
    init3 = []
    base1 = random.randint(1, 20)
    base2 = random.randint(20, 40)
    step = random.randint(15, 18)
    for i in range(16):
        v1 = base1 + i * step
        v2 = base2 + i * step
        if v1 <= MAX_VAL and v1 not in init3:
            init3.append(v1)
        if v2 <= MAX_VAL and v2 not in init3:
            init3.append(v2)
    while len(init3) < N:
        v = random.randint(1, MAX_VAL)
        if v not in init3:
            init3.append(v)
    init3 = init3[:N]
    
    # Initial candidate 4: Random
    init4 = random.sample(range(1, MAX_VAL + 1), N)
    
    candidates = [init1, init2, init3, init4]
    
    # Run greedy improvement on each candidate
    # We divide the time budget among them
    time_per_candidate = (deadline - time.time()) / len(candidates)
    
    for idx, cand in enumerate(candidates):
        if time.time() > deadline - 5.0:
            break
            
        # Ensure we have a valid list
        cand = list(cand)
        # Clean up duplicates/range issues just in case
        seen = set()
        clean_cand = []
        for v in cand:
            if 1 <= v <= MAX_VAL and v not in seen:
                seen.add(v)
                clean_cand.append(v)
        while len(clean_cand) < N:
            v = random.randint(1, MAX_VAL)
            if v not in seen:
                seen.add(v)
                clean_cand.append(v)
        clean_cand = clean_cand[:N]
        
        improved_lst, improved_score = greedy_improve(clean_cand, max_time=time_per_candidate)
        
        if improved_score > best_overall_score:
            best_overall_score = improved_score
            best_overall_lst = improved_lst[:]
            
    # Final verification and cleanup
    result = best_overall_lst[:]
    seen = set()
    clean_result = []
    for v in result:
        if 1 <= v <= MAX_VAL and v not in seen:
            seen.add(v)
            clean_result.append(v)
    
    while len(clean_result) < N:
        v = random.randint(1, MAX_VAL)
        if v not in seen:
            seen.add(v)
            clean_result.append(v)
            
    clean_result = clean_result[:N]
    
    return sorted(clean_result)

# EVOLVE-BLOCK-END
