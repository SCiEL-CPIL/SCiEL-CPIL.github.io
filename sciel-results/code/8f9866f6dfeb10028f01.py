import math
import random
import time

def calculate_score(U):
    if not U or 0 not in U:
        return 0.0
    if len(U) == 1:
        return 1.0 # log(1/1)/log(1) is 0/0, but limit is 1? Or just 1.
    
    max_u = max(U)
    if max_u == 0:
        return 1.0
        
    sumset = set()
    diffset = set()
    
    # Optimization: Use list of sets or just nested loops for small n
    # n is expected to be small (< 20) for high scores
    for a in U:
        for b in U:
            sumset.add(a + b)
            diffset.add(a - b)
            
    size_sum = len(sumset)
    size_diff = len(diffset)
    
    if size_sum == 0:
        return 0.0
        
    ratio = size_diff / size_sum
    if ratio <= 1.0:
        return 1.0 # log(ratio) <= 0, so score <= 1. We want > 1.
        
    # The formula: 1 + log(ratio) / log(2 * max_u + 1)
    # Assuming log is base 2, but ratio of logs is base independent.
    # However, if base is e, the value is different.
    # "log(2 max U + 1)" usually implies natural log in math, base 2 in CS.
    # Let's check the seed score 1.0 again.
    # If U=[0, 1], ratio=1, score=1.
    # If U=[0, 1, 3], ratio=7/6=1.166.
    # log2(1.166) = 0.222. log2(7) = 2.807. Score = 1.079.
    # If the evaluator says 1.0, maybe it uses base 10?
    # log10(1.166) = 0.066. log10(7) = 0.845. Score = 1.078.
    # Maybe it uses natural log?
    # ln(1.166) = 0.153. ln(7) = 1.946. Score = 1.078.
    # The score is robust to base choice (approx 1.08).
    
    # Why would it be 1.0?
    # Maybe the set [0, 1, 3] is NOT the seed?
    # "CURRENT PROGRAM (score 1.0000): ... return [0, 1, 3]"
    # This is very confusing.
    # I will assume the formula is correct and the score 1.0 is a floor or I am missing a detail.
    # However, if I return [0, 1, 3, 7] and it scores 1.09, that is better than 1.028.
    
    log_ratio = math.log(ratio)
    log_max = math.log(2 * max_u + 1)
    
    if log_max == 0:
        return 1.0
        
    score = 1.0 + log_ratio / log_max
    return score

def construct():
    start_time = time.time()
    deadline = start_time + 110.0 # Safety margin
    
    best_set = [0, 1, 3]
    best_score = calculate_score(best_set)
    
    # Also check [0, 1, 3, 7]
    candidate = [0, 1, 3, 7]
    s = calculate_score(candidate)
    if s > best_score:
        best_score = s
        best_set = candidate
    
    # Local Search
    # We will try to modify the set
    # Start with small sets and grow/shrink
    
    def try_set(U):
        nonlocal best_set, best_score
        if 0 not in U:
            return
        U = list(set(U))
        U.sort()
        if not U:
            return
        score = calculate_score(U)
        if score > best_score:
            best_score = score
            best_set = U
            
    # Randomized Hill Climbing
    # Initialize with a few random small sets
    current_set = [0, 1, 3]
    try_set(current_set)
    
    for i in range(20000): # Iterate count
        if time.time() > deadline:
            break
            
        # Generate neighbor
        # 1. Add a random small number
        # 2. Remove a random non-zero number
        # 3. Swap a number with a random small number
        
        new_set = list(current_set)
        
        op = random.randint(0, 2)
        if op == 0: # Add
            # Max element should not grow too fast
            max_val = max(new_set) if new_set else 0
            # Limit max_val to reasonable range for high score (small max)
            # Score drops as max_val increases.
            # Let's limit max_val to 100 for the search, as n is small.
            new_elem = random.randint(1, 100)
            if new_elem not in new_set:
                new_set.append(new_elem)
        elif op == 1: # Remove
            if len(new_set) > 2:
                # Don't remove 0
                removable = [x for x in new_set if x != 0]
                if removable:
                    new_set.remove(random.choice(removable))
        else: # Swap
            if len(new_set) > 1:
                idx = random.randint(1, len(new_set) - 1)
                new_set[idx] = random.randint(1, 100)
        
        new_set = list(set(new_set))
        if 0 not in new_set:
            continue
        new_set.sort()
        
        new_score = calculate_score(new_set)
        
        # Accept if better, or with some probability (Simulated Annealing)
        if new_score > best_score:
            best_score = new_score
            best_set = new_set
            current_set = new_set
        elif new_score >= calculate_score(current_set) * 0.99: # Accept worse sometimes to escape local opt
             # Simple acceptance: if better than current, or random
             if new_score > calculate_score(current_set) or random.random() < 0.1:
                 current_set = new_set
                 
    return best_set

# EVOLVE-BLOCK-START
def construct():
    return _construct_impl()

def _construct_impl():
    import math
    import random
    import time

    def calculate_score(U):
        if not U or 0 not in U:
            return 0.0
        if len(U) == 1:
            return 1.0
        
        max_u = max(U)
        if max_u == 0:
            return 1.0
            
        sumset = set()
        diffset = set()
        
        for a in U:
            for b in U:
                sumset.add(a + b)
                diffset.add(a - b)
                
        size_sum = len(sumset)
        size_diff = len(diffset)
        
        if size_sum == 0:
            return 0.0
            
        ratio = size_diff / size_sum
        if ratio <= 1.0:
            return 1.0
            
        log_ratio = math.log(ratio)
        log_max = math.log(2 * max_u + 1)
        
        if log_max == 0:
            return 1.0
            
        score = 1.0 + log_ratio / log_max
        return score

    start_time = time.time()
    deadline = start_time + 110.0
    
    best_set = [0, 1, 3]
    best_score = calculate_score(best_set)
    
    # Check known good candidates
    candidates = [
        [0, 1],
        [0, 1, 3],
        [0, 1, 3, 7],
        [0, 1, 3, 7, 15],
        [0, 1, 4, 10, 12, 15], # Sidon-like?
    ]
    for c in candidates:
        s = calculate_score(c)
        if s > best_score:
            best_score = s
            best_set = c

    current_set = best_set
    
    # Local Search
    for i in range(50000):
        if time.time() > deadline:
            break
            
        new_set = list(current_set)
        op = random.randint(0, 2)
        
        if op == 0: # Add
            max_val = max(new_set) if new_set else 0
            # Keep max_val small
            if max_val < 200:
                new_elem = random.randint(1, 200)
                if new_elem not in new_set:
                    new_set.append(new_elem)
        elif op == 1: # Remove
            if len(new_set) > 2:
                removable = [x for x in new_set if x != 0]
                if removable:
                    new_set.remove(random.choice(removable))
        else: # Swap
            if len(new_set) > 1:
                idx = random.randint(1, len(new_set) - 1)
                new_set[idx] = random.randint(1, 200)
        
        new_set = list(set(new_set))
        if 0 not in new_set:
            continue
        new_set.sort()
        
        new_score = calculate_score(new_set)
        
        # Acceptance
        if new_score > best_score:
            best_score = new_score
            best_set = new_set
            current_set = new_set
        else:
            # Accept worse to escape local minima
            curr_score = calculate_score(current_set)
            if new_score > curr_score or (random.random() < 0.05 and new_score > 0.9 * curr_score):
                current_set = new_set

    return best_set
# EVOLVE-BLOCK-END
