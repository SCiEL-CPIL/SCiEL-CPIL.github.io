import numpy as np
import time
import random
import sys

def _score_set(U):
    """
    Calculate the score for a set U.
    Score = 1 + log2(|U-U| / |U+U|) / log2(2 * max(U) + 1)
    """
    if not U:
        return 1.0
    
    # Ensure 0 is in U? The task says construct() returns a set containing 0.
    # We assume input is valid.
    U = sorted(set(U))
    n = len(U)
    if n < 2:
        return 1.0
        
    max_u = max(U)
    
    # If max_u is 0, U={0}. |U+U|=1, |U-U|=1. Ratio 1. Score 1.
    if max_u == 0:
        return 1.0

    # Create indicator array
    # Size: 2 * max_u + 1 is sufficient for sums in [0, 2*max_u]
    # and diffs in [-max_u, max_u] -> shifted to [0, 2*max_u]
    size = 2 * max_u + 1
    ind = np.zeros(size, dtype=np.int8)
    
    # Fill indicator
    # Using a loop is slow for large N, but N is likely < 1000
    # We can use ind[U] = 1 if U is an array of indices
    ind[U] = 1
    
    # Sumset: Convolution
    # ind * ind gives counts of sums
    conv = np.convolve(ind, ind, mode='full')
    sumset_size = np.sum(conv > 0)
    
    # Diffset: Correlation
    # np.correlate(ind, ind, 'full') gives counts of differences
    # The result is centered.
    corr = np.correlate(ind, ind, mode='full')
    diffset_size = np.sum(corr > 0)
    
    if sumset_size == 0 or diffset_size == 0:
        return 1.0
        
    ratio = diffset_size / sumset_size
    
    # If ratio is 1, score is 1.
    # If ratio < 1, score < 1.
    # We want to maximize.
    
    denom = np.log2(2 * max_u + 1)
    if denom < 1e-9:
        return 1.0
        
    val = 1.0 + np.log2(ratio) / denom
    
    return val

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 140 seconds
    
    # Initial best
    best_set = [0, 1, 3]
    best_score = _score_set(best_set)
    
    # Try a few random initializations to find a good starting point
    # and maybe a better best_score immediately
    for _ in range(10):
        if time.time() > deadline: break
        size = random.randint(5, 20)
        rand_set = set()
        rand_set.add(0)
        while len(rand_set) < size:
            rand_set.add(random.randint(1, 50))
        rand_list = list(rand_set)
        s = _score_set(rand_list)
        if s > best_score:
            best_score = s
            best_set = rand_list
            
    current_set = list(best_set)
    current_score = best_score
    
    T = 1.0
    cooling = 0.9995
    
    # Local Search / Simulated Annealing
    while time.time() < deadline:
        # Generate a move
        move_type = random.randint(0, 3)
        new_set = list(current_set)
        
        if move_type == 0: # Add
            if new_set:
                max_val = max(new_set)
                # Add a number. Range: up to 2*max_val or random
                # To keep max_u controlled, we don't go too high
                limit = max(100, max_val * 2)
                x = random.randint(0, limit)
                if x not in new_set:
                    new_set.append(x)
            else:
                new_set.append(0)
                
        elif move_type == 1: # Remove
            if len(new_set) > 1:
                # Don't remove 0
                if new_set[0] == 0:
                    idx = random.randint(1, len(new_set) - 1)
                else:
                    idx = random.randint(0, len(new_set) - 1)
                new_set.pop(idx)
                
        elif move_type == 2: # Swap
            if len(new_set) > 1:
                if new_set[0] == 0:
                    idx = random.randint(1, len(new_set) - 1)
                else:
                    idx = random.randint(0, len(new_set) - 1)
                max_val = max(new_set) if new_set else 10
                limit = max(100, max_val * 2)
                new_val = random.randint(0, limit)
                if new_val not in new_set:
                    new_set[idx] = new_val
                else:
                    # Collision, skip move
                    continue
                    
        else: # Perturb
            if len(new_set) > 1:
                if new_set[0] == 0:
                    idx = random.randint(1, len(new_set) - 1)
                else:
                    idx = random.randint(0, len(new_set) - 1)
                delta = random.choice([-1, 1, -2, 2, -5, 5])
                new_val = new_set[idx] + delta
                if new_val >= 0 and new_val not in new_set:
                    new_set[idx] = new_val
                else:
                    continue
        
        # Ensure 0 is in set
        if 0 not in new_set:
            new_set.append(0)
            
        # Evaluate
        new_score = _score_set(new_set)
        
        delta = new_score - current_score
        
        if delta > 0 or random.random() < np.exp(delta / T):
            current_set = new_set
            current_score = new_score
            if current_score > best_score:
                best_score = current_score
                best_set = list(current_set)
                
        T *= cooling
        if T < 0.001:
            T = 0.001
            # Optional: Reheat and restart from best
            if random.random() < 0.1:
                T = 1.0
                current_set = list(best_set)
                current_score = best_score

    return best_set
