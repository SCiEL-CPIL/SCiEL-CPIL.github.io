import math
import numpy as np
import time
import random

def score_set(U):
    """
    Calculate the score for a given set U.
    Score = 1 + log(|U-U|/|U+U|) / log(2*max(U) + 1)
    """
    if not U:
        return 0
    U = list(set(U))
    if 0 not in U:
        return 0
    max_u = max(U)
    if max_u == 0:
        return 1.0
    
    # Vectorized calculation for speed
    U_arr = np.array(U, dtype=np.int64)
    # Using outer sum and diff
    sums = np.add.outer(U_arr, U_arr).flatten()
    diffs = np.subtract.outer(U_arr, U_arr).flatten()
    
    # Count unique elements
    # np.unique sorts, which takes time. 
    # For small sets, this is fine. For larger sets, we might need a faster unique count.
    # However, n is likely small (under 20-30) in optimal solutions for this specific metric 
    # because max(U) grows and the log denominator penalizes large max(U).
    
    size_sum = len(np.unique(sums))
    size_diff = len(np.unique(diffs))
    
    if size_sum == 0:
        return 0
        
    ratio = size_diff / size_sum
    if ratio <= 0:
        return 0
        
    denom = math.log(2 * max_u + 1)
    if denom == 0:
        return 0
        
    return 1 + math.log(ratio) / denom

def construct():
    start_time = time.time()
    deadline = start_time + 145  # 5s margin
    
    # Initialize with a good seed. 
    # The paper mentions constructions of size 2003 and 54265. 
    # However, the score formula has a denominator log(2*max(U)+1). 
    # If max(U) is huge, the log term is large, reducing the impact of the ratio.
    # Let's start with a small set that has a decent ratio.
    # Sets like {0, 1, 3, 7, 12, ...} (Golomb-like) often have good difference sets.
    
    # Let's start with a random set or a known good small set.
    # Try {0, 1, 3, 7}
    best_set = [0, 1, 3, 7]
    best_score = score_set(best_set)
    
    current_set = best_set[:]
    current_score = best_score
    
    # Simulated Annealing
    T = 0.5
    T_min = 0.01
    cooling_rate = 0.999
    
    # We will try to add/remove elements and move elements.
    # The size of the set can vary.
    # Let's allow size to range from 3 to 20.
    
    while time.time() < deadline:
        if T < T_min:
            break
            
        # Generate a neighbor
        neighbor = current_set[:]
        random.shuffle(neighbor) # Not strictly necessary if we pick index, but ensures randomness
        
        move_type = random.random()
        
        if move_type < 0.3:
            # Move an element
            if len(neighbor) > 1:
                idx = random.randint(1, len(neighbor) - 1)
                old_val = neighbor[idx]
                # Try to move to a new value
                # Range: 1 to max(neighbor) + 10
                max_val = max(neighbor)
                new_val = random.randint(1, max_val + 10)
                if new_val not in neighbor:
                    neighbor[idx] = new_val
                    neighbor.sort()
        
        elif move_type < 0.6:
            # Add an element
            if len(neighbor) < 25:
                max_val = max(neighbor)
                new_val = random.randint(1, max_val + 20)
                if new_val not in neighbor:
                    neighbor.append(new_val)
                    neighbor.sort()
        
        elif move_type < 0.8:
            # Remove an element
            if len(neighbor) > 3:
                # Don't remove 0
                removable_indices = [i for i in range(1, len(neighbor))]
                idx = random.choice(removable_indices)
                neighbor.pop(idx)
        
        else:
            # Random restart (small probability)
            # Generate a random set of size 3-8
            n = random.randint(3, 8)
            max_u = random.randint(10, 50)
            vals = random.sample(range(1, max_u + 1), n - 1)
            neighbor = [0] + sorted(vals)
        
        if neighbor == current_set:
            continue
            
        neighbor_score = score_set(neighbor)
        
        if neighbor_score > current_score:
            current_set = neighbor
            current_score = neighbor_score
            if current_score > best_score:
                best_score = current_score
                best_set = current_set[:]
        else:
            # Accept worse solution with probability exp((new - old) / T)
            delta = neighbor_score - current_score
            if delta < 0:
                prob = math.exp(delta / T)
                if random.random() < prob:
                    current_set = neighbor
                    current_score = neighbor_score
        
        T *= cooling_rate
        
    return best_set
