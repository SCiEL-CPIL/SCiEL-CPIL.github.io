import time
import random
import numpy as np
from math import log

def score_set_fast(U):
    """
    Fast vectorized scoring for a list of integers U.
    """
    U = sorted(set(U))
    if 0 not in U:
        return -1
    n = len(U)
    if n < 2:
        return -1
        
    U_arr = np.array(U, dtype=np.int64)
    
    # Compute all pairwise sums (i <= j to handle symmetry, but set handles duplicates)
    # To be efficient, we can compute full outer sum and diff, then count unique.
    # For n up to ~100, n^2 = 10,000 operations which is very fast in numpy.
    
    sums = U_arr[:, None] + U_arr[None, :]
    diffs = U_arr[:, None] - U_arr[None, :]
    
    # Flatten and get unique counts
    # Using np.unique is efficient for small arrays
    sum_count = len(np.unique(sums))
    diff_count = len(np.unique(diffs))
    
    if sum_count == 0:
        return -1
        
    ratio = diff_count / sum_count
    if ratio <= 1:
        return -1
        
    max_u = U_arr.max()
    D = 2 * max_u + 1
    if D <= 1:
        return -1
        
    return 1 + log(ratio) / log(D)

def construct():
    start_time = time.time()
    deadline = start_time + 145  # Safety margin for 150s limit
    
    # Initial diverse seeds
    seeds = [
        [0, 1, 4, 9, 16, 25, 36, 49, 64, 81, 100, 121, 144, 169, 196, 225, 256, 289, 324, 361],
        [0, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768],
        [0, 1, 3, 9, 27, 81, 243, 729, 2187, 6561, 19683, 59049],
        [0, 1, 10, 100, 1000, 10000, 100000],
        list(range(0, 50, 2)) # Even numbers
    ]
    
    best_set = seeds[0]
    best_score = score_set_fast(best_set)
    
    for s in seeds[1:]:
        sc = score_set_fast(s)
        if sc > best_score:
            best_score = sc
            best_set = s
            
    # Simulated Annealing
    # Parameters
    T = 1.0
    cooling = 0.9995
    min_T = 0.001
    max_val_limit = 50000
    max_n_limit = 100
    min_n_limit = 5
    
    current_set = list(best_set)
    current_score = best_score
    
    # Pre-compute some random candidates for efficiency? 
    # Or just generate on the fly.
    
    while time.time() < deadline:
        if T < min_T:
            # Reset temperature if stuck for a while? Or just break if no progress.
            # Let's do a hard restart with a new random seed if we've been stuck.
            # Simple heuristic: if T is low, we are converged. Restart with a perturbed best.
            current_set = list(best_set)
            # Perturb
            if len(current_set) > 1:
                idx = random.randint(1, len(current_set)-1)
                current_set[idx] = max(1, current_set[idx] + random.randint(-100, 100))
                current_set = sorted(set(current_set))
                current_score = score_set_fast(current_set)
            T = 1.0
            
        # Choose a move
        move_type = random.random()
        new_set = None
        
        if move_type < 0.4:
            # Add an element
            if len(current_set) < max_n_limit:
                # Pick a random value in a wide range, or near existing elements
                if random.random() < 0.5:
                    # Near existing
                    base = random.choice(current_set)
                    delta = random.randint(-100, 100)
                    val = base + delta
                else:
                    # Random anywhere
                    val = random.randint(1, max_val_limit)
                
                if val > 0 and val not in current_set:
                    new_set = sorted(current_set + [val])
        
        elif move_type < 0.7:
            # Remove an element
            if len(current_set) > min_n_limit:
                idx = random.randint(1, len(current_set)-1)
                new_set = [x for i, x in enumerate(current_set) if i != idx]
                new_set = sorted(new_set)
        
        else:
            # Replace an element
            if len(current_set) > 1:
                idx = random.randint(1, len(current_set)-1)
                old_val = current_set[idx]
                # New value: near old or random
                if random.random() < 0.5:
                    delta = random.randint(-50, 50)
                    val = old_val + delta
                else:
                    val = random.randint(1, max_val_limit)
                
                if val > 0 and val not in current_set:
                    new_set = current_set.copy()
                    new_set[idx] = val
                    new_set = sorted(set(new_set))
        
        if new_set is None:
            T *= cooling
            continue
            
        new_score = score_set_fast(new_set)
        
        # Acceptance criterion
        delta_score = new_score - current_score
        if delta_score > 0:
            current_set = new_set
            current_score = new_score
            if current_score > best_score:
                best_score = current_score
                best_set = list(current_set)
        else:
            # Accept with probability exp(-delta/T)
            # If T is very small, this is unlikely
            if T > 0:
                prob = np.exp(-delta_score / T)
                if random.random() < prob:
                    current_set = new_set
                    current_score = new_score
        
        T *= cooling
        
    # Final polish: simple local search on best_set to ensure no obvious improvements
    # This is cheap and safe
    improved = True
    while improved and time.time() < deadline - 5:
        improved = False
        n = len(best_set)
        # Try adding/removing/replacing small deltas
        for i in range(1, n):
            if time.time() > deadline - 5: break
            old = best_set[i]
            # Try removing
            if n > 5:
                cand = [x for j, x in enumerate(best_set) if j != i]
                sc = score_set_fast(cand)
                if sc > best_score:
                    best_set = cand
                    best_score = sc
                    improved = True
                    break
            # Try small replacements
            for d in range(-5, 6):
                if d == 0: continue
                new_val = old + d
                if new_val <= 0: continue
                if new_val in best_set: continue
                cand = best_set.copy()
                cand[i] = new_val
                cand = sorted(set(cand))
                if len(cand) != n: continue
                sc = score_set_fast(cand)
                if sc > best_score:
                    best_set = cand
                    best_score = sc
                    improved = True
                    break
            if improved: break
        
        # Try adding a small value
        if not improved and n < max_n_limit:
            max_v = max(best_set)
            for v in range(1, max_v + 10):
                if v in best_set: continue
                cand = sorted(best_set + [v])
                sc = score_set_fast(cand)
                if sc > best_score:
                    best_set = cand
                    best_score = sc
                    improved = True
                    break

    return best_set

if __name__ == "__main__":
    # Quick test
    U = construct()
    print("Result:", U)
    print("Score:", score_set_fast(U))
