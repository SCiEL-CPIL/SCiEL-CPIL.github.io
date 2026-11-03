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
        
    # Use NumPy for faster computation
    U_arr = np.array(U, dtype=np.int64)
    
    # Compute all pairwise sums and differences
    sums_arr = U_arr[:, None] + U_arr[None, :]
    unique_sums = np.unique(sums_arr)
    
    diffs_arr = U_arr[:, None] - U_arr[None, :]
    unique_diffs = np.unique(diffs_arr)
    
    ratio = len(unique_diffs) / len(unique_sums)
    if ratio <= 1:
        return -1
        
    max_u = U_arr[-1]
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
    
    # Adaptive simulated annealing with time-based cooling
    T = 0.08
    min_T = 0.005
    cooling = 0.995
    
    iter_count = 0
    no_improve = 0
    patience = 500
    
    while time.time() < deadline - 5:
        iter_count += 1
        if len(current) <= 1:
            break
        
        # Randomly choose an operation: move, add, remove
        op = random.random()
        
        if op < 0.75:
            # Move an element
            i = random.randint(1, len(current) - 1)
            old_val = current[i]
            
            if random.random() < 0.4:
                # Very small perturbation for fine-tuning
                delta = random.randint(-2, 2)
                new_val = old_val + delta
            elif random.random() < 0.7:
                # Small perturbation
                delta = random.randint(-5, 5)
                new_val = old_val + delta
            else:
                # Jump
                new_val = random.randint(1, current[-1] + 15)
                
            if new_val <= 0:
                continue
                
            new_set = current[:]
            new_set[i] = new_val
            new_set = sorted(set(new_set))
            
            if len(new_set) != len(current):
                continue
                
        elif op < 0.88:
            # Add an element
            if len(current) >= 12: # Cap size to keep search fast
                continue
            # Try values near existing elements or gaps
            if random.random() < 0.5:
                # Near existing element
                base = random.choice(current[1:])
                new_val = base + random.randint(-3, 3)
            else:
                # Random in range
                new_val = random.randint(1, current[-1] + 25)
            if new_val <= 0 or new_val in current:
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
            no_improve = 0
        else:
            no_improve += 1
            
        # Accept if better or with probability
        if new_score > current_score:
            current = new_set
            current_score = new_score
        else:
            delta = new_score - current_score
            if delta < 0:
                prob = np.exp(delta / max(T, min_T))
            else:
                prob = 1.0
                
            if random.random() < prob:
                current = new_set
                current_score = new_score
        
        # Adaptive cooling: cool faster when stuck
        if no_improve > patience:
            T = max(T * 0.95, min_T)
            no_improve = 0
        else:
            T = max(T * cooling, min_T)
        
        # Periodic restart if stuck for too long
        if no_improve > patience * 3:
            # Restart from best with small perturbation
            current = best_set[:]
            current_score = best_score
            no_improve = 0
            T = 0.05
        
        if iter_count >= max_iter:
            break
    
    # Final polish: try small moves around best
    polish_deadline = min(deadline - 2, time.time() + 10)
    polish_current = best_set[:]
    polish_score = best_score
    
    for _ in range(200):
        if time.time() > polish_deadline:
            break
        if len(polish_current) <= 1:
            break
        
        i = random.randint(1, len(polish_current) - 1)
        old_val = polish_current[i]
        delta = random.randint(-2, 2)
        new_val = old_val + delta
        
        if new_val <= 0:
            continue
        
        new_set = polish_current[:]
        new_set[i] = new_val
        new_set = sorted(set(new_set))
        
        if len(new_set) != len(polish_current):
            continue
        
        new_score = score_set(new_set)
        if new_score > polish_score:
            polish_score = new_score
            polish_current = new_set
            if new_score > best_score:
                best_score = new_score
                best_set = new_set[:]
    
    return best_set, best_score

def construct():
    deadline = time.time() + 115  # Safety margin for 150s limit
    
    best_set = [0, 1, 4]
    best_score = score_set(best_set)
    
    # Seeds: mix of known good sets and random starts
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
        [0, 1, 2, 4, 7],
        [0, 1, 3, 6, 10],
        [0, 2, 4, 7, 12],
        [0, 1, 4, 7, 11],
    ]
    
    # Add some random seeds
    random.seed(42)
    for _ in range(15):
        n = random.randint(3, 8)
        max_val = random.randint(15, 60)
        U = [0]
        other_elements = random.sample(range(1, max_val), n - 1)
        U.extend(other_elements)
        seeds.append(sorted(U))
        
    # Local search from each seed with adaptive time allocation
    seeds_sorted = sorted(seeds, key=lambda s: score_set(s), reverse=True)
    
    for i, seed in enumerate(seeds_sorted):
        if time.time() > deadline - 15:
            break
        
        # Allocate more time to promising seeds
        if i < 5:
            max_iter = 5000
        elif i < 10:
            max_iter = 3000
        else:
            max_iter = 1500
            
        local_best, local_score = local_search(seed, deadline, max_iter=max_iter)
        if local_score > best_score:
            best_score = local_score
            best_set = local_best[:]
            
    # Final focused search from best found
    if time.time() < deadline - 20:
        final_best, final_score = local_search(best_set, deadline, max_iter=8000)
        if final_score > best_score:
            best_score = final_score
            best_set = final_best[:]
            
    return best_set
