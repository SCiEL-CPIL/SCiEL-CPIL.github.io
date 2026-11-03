import time
import random
import itertools
from math import log
import numpy as np

def score_set(U):
    U = sorted(set(U))
    if 0 not in U:
        return -1
    n = len(U)
    if n < 2:
        return -1
        
    U_arr = np.array(U, dtype=np.int64)
    max_u = int(U_arr[-1])
    
    # Use boolean arrays for sums and differences
    # Sums range: 0 to 2*max_u
    sum_size = 2 * max_u + 1
    # Diffs range: -(max_u) to max_u, offset by max_u
    diff_size = 2 * max_u + 1
    
    sums_arr = U_arr[:, None] + U_arr[None, :]
    diff_arr = U_arr[:, None] - U_arr[None, :]
    
    # Count unique sums using boolean array
    sum_mask = np.zeros(sum_size, dtype=np.uint8)
    sum_mask[sums_arr] = 1
    n_sums = int(sum_mask.sum())
    
    # Count unique diffs using boolean array
    diff_mask = np.zeros(diff_size, dtype=np.uint8)
    diff_mask[diff_arr + max_u] = 1
    n_diffs = int(diff_mask.sum())
    
    if n_sums == 0:
        return -1
        
    ratio = n_diffs / n_sums
    if ratio <= 1:
        return -1
        
    D = 2 * max_u + 1
    if D <= 1:
        return -1
        
    return 1 + log(ratio) / log(D)

def local_search(start_set, deadline, max_iter=5000):
    current = sorted(set(start_set))
    if 0 not in current:
        current = sorted(set(current + [0]))
    current_score = score_set(current)
    if current_score < 0:
        return current, -1
    
    best_set = current[:]
    best_score = current_score
    
    # Simulated annealing with adaptive temperature
    T = 0.08
    cooling = 0.995
    min_T = 0.001
    
    while time.time() < deadline - 2 and max_iter > 0:
        max_iter -= 1
        if len(current) <= 1:
            break
            
        # Randomly choose an operation: move, add, remove
        op = random.random()
        
        if op < 0.75:
            # Move an element
            i = random.randint(1, len(current) - 1)
            old_val = current[i]
            
            if random.random() < 0.4:
                # Very small perturbation
                delta = random.choice([-2, -1, 1, 2])
                new_val = old_val + delta
            elif random.random() < 0.7:
                # Small perturbation
                delta = random.randint(-5, 5)
                new_val = old_val + delta
            else:
                # Larger jump
                new_val = random.randint(1, current[-1] + 15)
                
            if new_val <= 0:
                continue
                
            new_set = current[:]
            new_set[i] = new_val
            new_set = sorted(set(new_set))
            
            if len(new_set) != len(current):
                continue
                
        elif op < 0.90:
            # Add an element
            if len(current) >= 18:
                continue
            # Try to add near gaps or extend range
            if random.random() < 0.5:
                new_val = random.randint(1, current[-1] + 25)
            else:
                # Try to fill a gap
                if len(current) >= 2:
                    gap_start, gap_end = random.choice([(current[i], current[i+1]) for i in range(len(current)-1) if current[i+1] - current[i] > 1])
                    if gap_end - gap_start > 1:
                        new_val = random.randint(gap_start + 1, gap_end - 1)
                    else:
                        new_val = random.randint(1, current[-1] + 25)
                else:
                    new_val = random.randint(1, current[-1] + 25)
            if new_val in current:
                continue
            new_set = sorted(current + [new_val])
            
        else:
            # Remove an element (not 0)
            if len(current) <= 3:
                continue
            i = random.randint(1, len(current) - 1)
            new_set = current[:i] + current[i+1:]
            
        new_score = score_set(new_set)
        
        if new_score > best_score:
            best_score = new_score
            best_set = new_set[:]
            
        # Accept if better or with probability
        if new_score > current_score:
            current = new_set
            current_score = new_score
        else:
            delta = new_score - current_score
            if delta < 0:
                prob = np.exp(delta / T)
            else:
                prob = 1.0
                
            if random.random() < prob:
                current = new_set
                current_score = new_score
                
        T = max(T * cooling, min_T)
        
    return best_set, best_score

def generate_geometric_seeds():
    """Generate geometric progression based seeds."""
    seeds = []
    # Try various bases and lengths
    for base in range(2, 8):
        for length in range(3, 12):
            if time.time() > 100:  # quick check
                break
            U = [0]
            val = 1
            for i in range(length):
                if val > 500:
                    break
                U.append(val)
                val *= base
            if len(U) >= 3:
                seeds.append(sorted(set(U)))
    return seeds

def construct():
    deadline = time.time() + 110  # Safety margin for 120s limit
    
    best_set = [0, 1, 4]
    best_score = score_set(best_set)
    
    # Seeds: mix of known good sets, geometric progressions, and random starts
    seeds = [
        [0, 1, 3],
        [0, 1, 4, 9, 11],
        [0, 1, 3, 7, 12],
        [0, 1, 4, 10, 12, 17],
        [0, 1, 4, 10, 18, 23, 31],
        [0, 1, 4, 10, 12, 17, 25],
        [0, 1, 2, 4, 8],
        [0, 1, 2, 5, 11],
        [0, 1, 3, 7, 14],
        [0, 1, 3, 8, 13],
        [0, 2, 3, 7, 12],
        [0, 1, 4, 11, 13],
        [0, 1, 4, 9, 13],
        [0, 1, 4],
        [0, 1, 4, 16],
        [0, 1, 3, 9, 27],
    ]
    
    # Add geometric progression seeds
    geo_seeds = generate_geometric_seeds()
    seeds.extend(geo_seeds)
    
    # Add some random seeds
    random.seed(42)
    for _ in range(15):
        n = random.randint(4, 10)
        max_val = random.randint(20, 120)
        U = [0]
        other_elements = random.sample(range(1, max_val), min(n - 1, max_val - 1))
        U.extend(other_elements)
        seeds.append(sorted(set(U)))
    
    # Deduplicate seeds
    seen = set()
    unique_seeds = []
    for s in seeds:
        key = tuple(s)
        if key not in seen:
            seen.add(key)
            unique_seeds.append(s)
    seeds = unique_seeds
    
    # Phase 1: Quick local search from all seeds
    for seed in seeds:
        if time.time() > deadline - 30:
            break
        local_best, local_score = local_search(seed, deadline, max_iter=2000)
        if local_score > best_score:
            best_score = local_score
            best_set = local_best[:]
    
    # Phase 2: Focused local search from best few
    if time.time() < deadline - 15:
        # Do a deeper search from the current best
        local_best, local_score = local_search(best_set, deadline, max_iter=5000)
        if local_score > best_score:
            best_score = local_score
            best_set = local_best[:]
    
    # Phase 3: Try a few more random restarts if time permits
    if time.time() < deadline - 10:
        random.seed(123)
        for _ in range(5):
            if time.time() > deadline - 5:
                break
            n = random.randint(5, 12)
            max_val = random.randint(30, 150)
            U = [0]
            other_elements = random.sample(range(1, max_val), min(n - 1, max_val - 1))
            U.extend(other_elements)
            seed = sorted(set(U))
            local_best, local_score = local_search(seed, deadline, max_iter=3000)
            if local_score > best_score:
                best_score = local_score
                best_set = local_best[:]
    
    return best_set
