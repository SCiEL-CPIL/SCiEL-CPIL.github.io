import time
import math
import random

def _score_set(U):
    U_set = sorted(set(U))
    n = len(U_set)
    if n == 0:
        return 0.0
    max_u = max(U_set)
    if max_u == 0:
        return 1.0
    
    sumset = set()
    diffset = set()
    for a in U_set:
        for b in U_set:
            sumset.add(a + b)
            diffset.add(a - b)
            
    if not sumset:
        return 0.0
        
    ratio = len(diffset) / len(sumset)
    if ratio <= 0:
        return 0.0
        
    log_ratio = math.log(ratio)
    log_denom = math.log(2 * max_u + 1)
    
    if log_denom == 0:
        return 1.0
        
    return 1.0 + log_ratio / log_denom

def _local_search(U, deadline):
    U = sorted(set(U))
    best_score = _score_set(U)
    best_U = U[:]
    
    # We will iterate until no improvement or deadline
    # To avoid getting stuck, we can randomize the order or use simulated annealing?
    # Simple hill climbing is fine for now.
    
    while time.time() < deadline:
        improved = False
        max_u = max(best_U)
        # Search range: 1 to max_u + 10
        # If max_u is small, we might want to go higher?
        # But increasing max_u increases denominator.
        # So usually we stay close.
        search_upper = max_u + 10
        if search_upper < 20:
            search_upper = 20
            
        for i in range(len(best_U)):
            if best_U[i] == 0:
                continue
            old_val = best_U[i]
            # Create a set of current values excluding U[i] for fast lookup
            current_vals = set(best_U)
            current_vals.remove(old_val)
            
            for new_val in range(1, search_upper + 1):
                if new_val in current_vals:
                    continue
                new_U = best_U[:]
                new_U[i] = new_val
                new_U.sort()
                
                s = _score_set(new_U)
                if s > best_score + 1e-9:
                    best_score = s
                    best_U = new_U
                    improved = True
                    break
            if improved:
                break
        if not improved:
            break
            
    return sorted(set(best_U)), best_score

def construct():
    start_time = time.time()
    deadline = start_time + 100
    
    seeds = [
        [0, 1, 3, 7, 15],
        [0, 1, 3, 7, 15, 31],
        [0, 1, 2, 4, 8],
        [0, 1, 2, 4, 8, 16],
        [0, 1, 4, 10, 12],
        [0, 1, 3, 4],
        [0, 1, 2, 5],
        [0, 1, 3, 6],
        [0, 1, 3, 7],
    ]
    
    best_U = seeds[0]
    best_score = _score_set(best_U)
    
    # Run local search on each seed
    for seed in seeds:
        if time.time() > deadline:
            break
        opt_U, opt_score = _local_search(seed, deadline)
        if opt_score > best_score:
            best_score = opt_score
            best_U = opt_U
            
    # If time remains, try random perturbations of the best set
    if time.time() < deadline:
        random.seed(42)
        for _ in range(100):
            if time.time() > deadline:
                break
            perturbed = best_U[:]
            i = random.randint(1, len(perturbed)-1)
            old = perturbed[i]
            new = random.randint(1, max(perturbed) + 10)
            if new == old or new in perturbed:
                continue
            perturbed[i] = new
            perturbed.sort()
            s = _score_set(perturbed)
            if s > best_score:
                best_score = s
                best_U = perturbed
                # Restart local search?
                opt_U, opt_score = _local_search(best_U, deadline)
                if opt_score > best_score:
                    best_score = opt_score
                    best_U = opt_U
                    
    return sorted(set(best_U))
