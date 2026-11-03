import time
import math
import random

def calculate_score(U):
    if not U:
        return 1.0
    max_u = max(U)
    if max_u == 0:
        return 1.0
    
    sums = set()
    diffs = set()
    
    # Convert to list for faster iteration
    U_list = list(U)
    n = len(U_list)
    
    for i in range(n):
        u1 = U_list[i]
        for j in range(n):
            u2 = U_list[j]
            sums.add(u1 + u2)
            diffs.add(u1 - u2)
            
    ratio = len(diffs) / len(sums)
    if ratio <= 1:
        return 1.0
        
    denom = math.log(2 * max_u + 1)
    if denom == 0:
        return 1.0
        
    return 1.0 + math.log(ratio) / denom

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    # Initial set
    current_U = {0}
    current_score = calculate_score(current_U)
    best_U = set(current_U)
    best_score = current_score
    
    # Greedy Initialization
    # Try to build a good initial set
    max_val_limit = 100
    while time.time() < deadline - 30:  # Leave time for SA
        best_add = None
        best_add_score = current_score
        current_max = max(current_U)
        
        # Try adding numbers slightly larger than current max
        # and also some smaller ones if gaps exist
        candidates = []
        # Range to search
        search_max = min(current_max + 20, max_val_limit)
        for k in range(1, search_max + 1):
            if k not in current_U:
                candidates.append(k)
        
        if not candidates:
            break
            
        for k in candidates:
            new_U = current_U | {k}
            s = calculate_score(new_U)
            if s > best_add_score:
                best_add_score = s
                best_add = k
                
        if best_add is None:
            break
            
        current_U.add(best_add)
        current_score = best_add_score
        
        if current_score > best_score:
            best_score = current_score
            best_U = set(current_U)
            
        # If we haven't improved in a few steps, break?
        # Or just continue.
        
    # Simulated Annealing
    T = 0.1
    T_min = 0.001
    cooling = 0.995
    
    current_U = set(best_U)
    current_score = best_score
    
    while time.time() < deadline:
        if T < T_min:
            T = T_min
            
        # Generate move
        move_type = random.random()
        new_U = set(current_U)
        
        if move_type < 0.4:
            # Add
            current_max = max(current_U) if current_U else 0
            # Search range around current max
            search_max = min(current_max + 15, 200)
            k = random.randint(1, search_max)
            if k not in new_U:
                new_U.add(k)
            else:
                continue
        elif move_type < 0.7:
            # Remove
            removable = [x for x in current_U if x != 0]
            if removable:
                k = random.choice(removable)
                new_U.discard(k)
            else:
                continue
        else:
            # Replace
            removable = [x for x in current_U if x != 0]
            if removable:
                k = random.choice(removable)
                current_max = max(current_U)
                search_max = min(current_max + 15, 200)
                y = random.randint(1, search_max)
                if y != k:
                    new_U.discard(k)
                    new_U.add(y)
                else:
                    continue
            else:
                continue
                
        new_score = calculate_score(new_U)
        
        # Acceptance
        delta = new_score - current_score
        if delta > 0 or random.random() < math.exp(delta / T):
            current_U = new_U
            current_score = new_score
            
            if current_score > best_score:
                best_score = current_score
                best_U = set(current_U)
                
        T *= cooling
        
    return sorted(list(best_U))
