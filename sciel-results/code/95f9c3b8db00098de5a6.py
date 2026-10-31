import numpy as np
import random
import time

def score_set(U):
    if not U or 0 not in U:
        return 0.0
    U = set(U)
    n = len(U)
    if n < 2:
        return 1.0
    max_u = max(U)
    # Calculate sumset and diffset sizes
    # For small n, nested loops are fine
    sum_set = set()
    diff_set = set()
    for a in U:
        for b in U:
            sum_set.add(a + b)
            diff_set.add(a - b)
    
    s_size = len(sum_set)
    d_size = len(diff_set)
    
    if s_size == 0:
        return 0.0
        
    ratio = d_size / s_size
    # Avoid log(0) or negative
    if ratio <= 1:
        # If ratio is 1, log is 0, score is 1
        # If ratio < 1, log is negative, score < 1
        # We want to maximize, so < 1 is bad, but possible?
        # Usually d_size >= s_size for sets containing 0?
        # Not necessarily.
        pass
        
    denom = np.log(2 * max_u + 1)
    if denom == 0:
        return 1.0
        
    val = 1 + np.log(ratio) / denom
    return val

def construct():
    start_time = time.time()
    deadline = start_time + 110  # 10s margin
    
    best_score = 0.0
    best_set = {0}
    
    # Initial candidates
    candidates = [
        {0, 1, 3},
        {0, 1, 4, 6},
        {0, 1, 2, 4},
        {0, 1, 3, 4},
        {0, 2, 3, 5},
        {0, 1, 5, 7},
        {0, 1, 4, 7},
        {0, 1, 2, 5, 8},
        {0, 1, 3, 6, 10},
        {0, 1, 4, 6, 9},
    ]
    
    for c in candidates:
        s = score_set(c)
        if s > best_score:
            best_score = s
            best_set = set(c)
            
    # Local Search
    current_set = set(best_set)
    current_score = best_score
    
    max_m = 60
    no_improve_count = 0
    temp = 1.0
    min_temp = 0.01
    
    while time.time() < deadline:
        if no_improve_count > 100:
            break
            
        # Generate a move
        move_type = random.choice(['add', 'remove', 'swap'])
        new_set = set(current_set)
        
        if move_type == 'add':
            # Add a random element
            candidates_add = [i for i in range(max_m + 1) if i not in new_set]
            if candidates_add:
                new_set.add(random.choice(candidates_add))
        elif move_type == 'remove':
            # Remove a random non-zero element
            candidates_remove = [i for i in new_set if i != 0]
            if candidates_remove:
                new_set.discard(random.choice(candidates_remove))
        elif move_type == 'swap':
            if len(new_set) > 1:
                candidates_remove = [i for i in new_set if i != 0]
                if candidates_remove:
                    old = random.choice(candidates_remove)
                    new_set.discard(old)
                    candidates_add = [i for i in range(max_m + 1) if i not in new_set]
                    if candidates_add:
                        new_set.add(random.choice(candidates_add))
        
        if new_set == current_set:
            continue
            
        new_score = score_set(new_set)
        
        if new_score > current_score:
            current_set = new_set
            current_score = new_score
            no_improve_count = 0
            if current_score > best_score:
                best_score = current_score
                best_set = set(current_set)
        else:
            no_improve_count += 1
            # Simulated annealing step
            delta = new_score - current_score
            if delta < 0:
                prob = np.exp(delta / temp)
                if random.random() < prob:
                    current_set = new_set
                    current_score = new_score
            temp = max(min_temp, temp * 0.95)
            
        # Periodic restart
        if no_improve_count > 200:
            current_set = set(best_set)
            current_score = best_score
            no_improve_count = 0
            temp = 1.0
            
    return sorted(best_set)

if __name__ == "__main__":
    print(construct())
