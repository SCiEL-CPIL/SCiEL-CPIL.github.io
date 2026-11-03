import time
import random
import math

def score_set(U):
    if not U:
        return 0
    U = list(set(U))
    if 0 not in U:
        return 0
    M = max(U)
    if M == 0:
        return 1.0 # log(1)/log(1) is undefined, but limit is 1? 
        # Actually if U={0}, U-U={0}, U+U={0}. Ratio 1. log(1)=0. Score 1.
    
    # Calculate sumset and diffset
    # Using sets for simplicity, as sizes are small
    sum_set = set()
    diff_set = set()
    
    # Optimization: if U is small, nested loops are fine
    for a in U:
        for b in U:
            sum_set.add(a + b)
            diff_set.add(a - b)
            
    n_sum = len(sum_set)
    n_diff = len(diff_set)
    
    if n_sum == 0:
        return 0
        
    ratio = n_diff / n_sum
    if ratio <= 1:
        # Score would be <= 1
        # But we want to maximize, so we track this
        pass
        
    denom = math.log(2 * M + 1)
    if denom == 0:
        return 1.0
        
    val = 1.0 + math.log(ratio) / denom
    return val

def construct():
    start_time = time.time()
    deadline = start_time + 110  # 10s safety margin
    
    # Initial candidate
    best_U = [0, 1, 3]
    best_score = score_set(best_U)
    
    # Local search
    # We will try to improve the set by adding, removing, or swapping elements.
    # We restrict the search space for new elements to [0, 100] to keep it bounded.
    # If we find a good set with large M, we might want to allow larger M, 
    # but usually small M is better for the denominator.
    
    max_val = 50
    max_size = 10
    
    current_U = list(best_U)
    current_score = best_score
    
    # Random restarts or continuous local search?
    # Continuous local search with occasional random jumps.
    
    while time.time() < deadline:
        # Generate a move
        move_type = random.random()
        
        if move_type < 0.3:
            # Add element
            if len(current_U) < max_size:
                # Try adding a random element
                new_elem = random.randint(1, max_val)
                if new_elem not in current_U:
                    candidate = current_U + [new_elem]
                    s = score_set(candidate)
                    if s > current_score:
                        current_U = candidate
                        current_score = s
                        if current_score > best_score:
                            best_score = current_score
                            best_U = current_U
        elif move_type < 0.6:
            # Remove element
            if len(current_U) > 1:
                # Remove a random non-zero element
                removable = [x for x in current_U if x != 0]
                if removable:
                    rem_elem = random.choice(removable)
                    candidate = [x for x in current_U if x != rem_elem]
                    s = score_set(candidate)
                    if s > current_score:
                        current_U = candidate
                        current_score = s
                        if current_score > best_score:
                            best_score = current_score
                            best_U = current_U
        else:
            # Swap element
            if len(current_U) > 1:
                removable = [x for x in current_U if x != 0]
                if removable:
                    rem_elem = random.choice(removable)
                    new_elem = random.randint(1, max_val)
                    if new_elem not in current_U:
                        candidate = [x for x in current_U if x != rem_elem]
                        candidate.append(new_elem)
                        s = score_set(candidate)
                        if s > current_score:
                            current_U = candidate
                            current_score = s
                            if current_score > best_score:
                                best_score = current_score
                                best_U = current_U
                                
        # Occasionally restart from a random small set to escape local optima
        if random.random() < 0.01:
             # Random restart
             size = random.randint(2, 5)
             new_U = [0]
             for _ in range(size - 1):
                 new_U.append(random.randint(1, max_val))
             new_U = list(set(new_U))
             if len(new_U) > 0:
                 s = score_set(new_U)
                 if s > current_score:
                     current_U = new_U
                     current_score = s
                     if current_score > best_score:
                         best_score = current_score
                         best_U = current_U

    return best_U
