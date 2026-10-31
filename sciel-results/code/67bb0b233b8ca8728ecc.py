import time
import random
import math

def calculate_score(U):
    U = set(U)
    if 0 not in U:
        return -1000.0
        
    # Normalize to min 0
    min_u = min(U)
    if min_u != 0:
        U = {x - min_u for x in U}
        
    if len(U) == 1:
        # U={0}, ratio 1, log(1)=0, log(1)=0. Undefined.
        # Return 1.0 as limit?
        return 1.0

    max_u = max(U)
    val = 2 * max_u + 1
    if val <= 1:
        return 1.0

    sum_set = set()
    diff_set = set()
    u_list = list(U)
    n = len(u_list)
    
    for i in range(n):
        for j in range(n):
            sum_set.add(u_list[i] + u_list[j])
            diff_set.add(u_list[i] - u_list[j])
            
    num = len(diff_set)
    den = len(sum_set)
    
    if den == 0:
        return -1000.0
        
    ratio = num / den
    
    # If ratio <= 1, score <= 1
    # We want to maximize
    score = 1.0 + math.log(ratio) / math.log(val)
    return score

def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    
    # Seed
    best_set = [0, 1, 3]
    best_score = calculate_score(best_set)
    
    def local_search(U, time_limit):
        U = set(U)
        min_u = min(U)
        if min_u != 0:
            U = {x - min_u for x in U}
            
        current_score = calculate_score(list(U))
        max_val = max(U) if U else 0
        
        while time.time() < time_limit:
            # Choose a move
            move_type = random.randint(0, 2)
            new_U = set(U)
            
            if move_type == 0:
                # Change existing element
                if not U: continue
                u_list = list(U)
                idx = random.randint(0, len(u_list)-1)
                old_val = u_list[idx]
                # Try a new value
                # Keep max_val somewhat controlled
                new_val = random.randint(0, max_val + 10)
                if new_val == old_val:
                    continue
                new_U.remove(old_val)
                new_U.add(new_val)
                
            elif move_type == 1:
                # Add element
                if len(U) > 30: # Cap size
                    continue
                new_val = random.randint(0, max_val + 10)
                if new_val in new_U:
                    continue
                new_U.add(new_val)
                
            else:
                # Remove element
                if len(U) <= 1:
                    continue
                u_list = list(U)
                idx = random.randint(0, len(u_list)-1)
                old_val = u_list[idx]
                if old_val == 0:
                    continue # Must keep 0
                new_U.remove(old_val)
                
            # Normalize
            if not new_U:
                continue
            min_u = min(new_U)
            if min_u != 0:
                new_U = {x - min_u for x in new_U}
            
            new_score = calculate_score(list(new_U))
            
            # Accept if better, or with small prob if worse
            if new_score > current_score or (random.random() < 0.1 and new_score > current_score - 0.05):
                U = new_U
                current_score = new_score
                max_val = max(U) if U else 0
                
        return list(U), current_score

    # Search loop
    # Try different initial sizes
    for n in range(3, 20):
        if time.time() > deadline:
            break
            
        for restart in range(20):
            if time.time() > deadline:
                break
            
            # Initialize
            # Use a geometric or random start
            if restart == 0:
                # Start with AP-like
                U_init = list(range(n))
            else:
                # Random
                max_init = n * 5
                candidates = random.sample(range(1, max_init), n-1)
                U_init = [0] + candidates
            
            # Time per restart: remaining time / remaining restarts
            remaining_time = deadline - time.time()
            time_per_restart = remaining_time / 20
            limit = time.time() + max(0.5, time_per_restart)
            
            U_cand, score_cand = local_search(U_init, limit)
            
            if score_cand > best_score:
                best_score = score_cand
                best_set = U_cand
                
    return best_set
