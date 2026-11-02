"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math
import random
import numpy as np

def _score_set(U):
    """Compute the score for a finite set U containing 0."""
    U = list(set(U))
    if not U:
        return 0.0
    n = len(U)
    if n == 0:
        return 0.0
    max_u = max(U)
    if max_u == 0:
        return 1.0
    
    U_arr = np.array(U, dtype=np.int64)
    
    # Compute sumset and difference set using numpy
    # This is efficient for n < 200
    sums = np.add.outer(U_arr, U_arr)
    diffs = np.subtract.outer(U_arr, U_arr)
    
    sumset_size = len(np.unique(sums))
    diffset_size = len(np.unique(diffs))
    
    if sumset_size == 0:
        return 0.0
        
    ratio = diffset_size / sumset_size
    if ratio <= 0:
        return 0.0
    if ratio == 1.0:
        return 1.0
        
    log_ratio = math.log(ratio)
    log_denom = math.log(2 * max_u + 1)
    
    if log_denom == 0:
        return 1.0
        
    return 1.0 + log_ratio / log_denom

def construct():
    start_time = time.time()
    deadline = start_time + 115  # Safety margin
    
    best_U = [0, 1, 3]
    best_score = _score_set(best_U)
    
    # Generate diverse seeds
    seeds = []
    
    # 1. Random small sets
    for _ in range(2000):
        if time.time() > deadline - 20: break
        n = random.randint(3, 15)
        # Random set with 0, values up to 200
        U = [0] + sorted(random.sample(range(1, 200), n-1))
        seeds.append(U)
        
    # 2. Structured seeds: AP with outlier
    for k in range(3, 12):
        for M in range(k+1, 100):
            U = list(range(k)) + [M]
            if len(set(U)) == len(U):
                seeds.append(U)
                
    # 3. Structured seeds: Powers of 2 (large gaps, good for diffset?)
    for k in range(3, 10):
        U = [2**i for i in range(k)]
        seeds.append(U)
        
    # 4. Structured seeds: Random Golomb-like
    for _ in range(100):
        if time.time() > deadline - 20: break
        n = random.randint(4, 12)
        U = [0]
        for i in range(1, n):
            U.append(random.randint(1, 300))
        U = sorted(set(U))
        if 0 in U:
            seeds.append(U)
            
    # Evaluate seeds and pick top ones
    scored_seeds = []
    for U in seeds:
        if time.time() > deadline - 10: break
        s = _score_set(U)
        scored_seeds.append((s, U))
        
    scored_seeds.sort(key=lambda x: x[0], reverse=True)
    
    # Run Simulated Annealing from top seeds
    # We can run SA in parallel? No, single thread.
    # We will run SA for the top 10 seeds, allocating time to each.
    num_sa_runs = 10
    time_per_run = (deadline - time.time()) / max(1, num_sa_runs)
    
    for i in range(min(num_sa_runs, len(scored_seeds))):
        if time.time() > deadline - 2: break
        
        s, U = scored_seeds[i]
        current_U = U[:]
        current_score = s
        
        if current_score > best_score:
            best_score = current_score
            best_U = current_U[:]
            
        T = 0.05  # Initial temperature
        T_min = 1e-6
        cooling = 0.999  # Slow cooling for more exploration
        
        # Run SA until time runs out for this seed
        while time.time() < deadline - 2 and T > T_min:
            # Generate neighbor
            new_U = current_U[:]
            move_type = random.random()
            
            max_val = max(new_U) if new_U else 0
            
            if move_type < 0.4:
                # Move an element (local)
                if len(new_U) > 1:
                    idx = random.randint(1, len(new_U)-1)
                    # Small perturbation
                    new_val = max(1, new_U[idx] + random.randint(-10, 10))
                    new_U[idx] = new_val
            elif move_type < 0.6:
                # Move an element (global jump)
                if len(new_U) > 1:
                    idx = random.randint(1, len(new_U)-1)
                    new_val = random.randint(1, max_val + 50)
                    new_U[idx] = new_val
            elif move_type < 0.85:
                # Add an element
                new_val = random.randint(1, max_val + 50)
                new_U.append(new_val)
            else:
                # Remove an element
                if len(new_U) > 3:
                    idx = random.randint(1, len(new_U)-1)
                    new_U.pop(idx)
                    
            new_U = sorted(set(new_U))
            if 0 not in new_U: continue
            
            new_score = _score_set(new_U)
            delta = new_score - current_score
            
            if delta > 0 or random.random() < math.exp(delta / T):
                current_U = new_U
                current_score = new_score
                
                if current_score > best_score:
                    best_score = current_score
                    best_U = current_U[:]
            
            T *= cooling
            
    return sorted(set(best_U))
# EVOLVE-BLOCK-END
