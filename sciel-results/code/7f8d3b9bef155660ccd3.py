import time
import random
from math import log
import numpy as np

def score_set(U):
    U = sorted(set(U))
    if 0 not in U:
        return -1
    n = len(U)
    if n < 2:
        return 0.0
    
    # Use NumPy for faster sumset and difference set calculation
    U_arr = np.array(U)
    
    # Calculate all pairwise sums
    sums = np.unique(U_arr[:, None] + U_arr[None, :])
    
    # Calculate all pairwise differences
    diffs = np.unique(U_arr[:, None] - U_arr[None, :])
    
    ratio = len(diffs) / len(sums)
    if ratio <= 1:
        return -1
    max_u = max(U)
    D = 2 * max_u + 1
    if D <= 1:
        return -1
    return 1 + log(ratio) / log(D)

def geometric_set(b, m):
    """Construct U = {floor(b^k) for k=0,...,m}"""
    U = [0]
    for k in range(1, m + 1):
        val = int(b ** k)
        if val <= 0:
            val = 1
        if val not in U:
            U.append(val)
    return sorted(U)

def score_geometric(b, m):
    """Score a geometric progression set."""
    U = geometric_set(b, m)
    if len(U) < 2:
        return -1
    return score_set(U)

def local_search(U, deadline, max_iter=5000):
    current = sorted(set(U))
    current_score = score_set(current)
    if current_score < 0:
        return current, current_score
    
    temperature = 0.05
    cooling_rate = 0.99
    
    iteration = 0
    while time.time() < deadline and iteration < max_iter:
        iteration += 1
        n = len(current)
        
        move_type = random.randint(1, 3)
        new_set = current
        
        if move_type == 1:
            i = random.randint(1, n - 1)
            old_val = current[i]
            # Perturb value
            if random.random() < 0.5:
                delta = random.randint(-10, 10)
            else:
                delta = random.randint(-int(old_val*0.5), int(old_val*0.5))
            
            new_val = old_val + delta
            if new_val <= 0:
                continue
            new_set = [current[j] if j != i else new_val for j in range(n)]
            new_set = sorted(set(new_set))
            if len(new_set) != n:
                continue
                
        elif move_type == 2:
            # Add element
            if n >= 10:
                continue
            max_val = max(current)
            if random.random() < 0.5:
                new_val = random.randint(max_val + 1, max_val + 10)
            else:
                new_val = random.randint(1, max_val + 10)
            
            if new_val in current:
                continue
            new_set = sorted(current + [new_val])
            
        else:
            # Remove element
            if n <= 3:
                continue
            i = random.randint(1, n - 1)
            new_set = [current[j] for j in range(n) if j != i]

        new_score = score_set(new_set)
        
        if new_score > current_score:
            current = new_set
            current_score = new_score
        else:
            delta = current_score - new_score
            if random.random() < np.exp(-delta / max(temperature, 1e-9)):
                current = new_set
                current_score = new_score
        
        temperature *= cooling_rate

    return current, current_score

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin
    
    best_set = [0, 1, 4]
    best_score = score_set(best_set)
    
    # Phase 1: Search geometric progressions
    for b in range(2, 20):
        for m in range(2, 10):
            if time.time() > deadline - 20:
                break
            U = geometric_set(b, m)
            s = score_set(U)
            if s > best_score:
                best_score = s
                best_set = U
    
    # Phase 2: Local search from best GP
    if time.time() < deadline - 10:
        refined, refined_score = local_search(best_set, deadline - 5)
        if refined_score > best_score:
            best_score = refined_score
            best_set = refined
    
    # Phase 3: Try some specific small sets
    candidates = [
        [0, 1, 2, 5],
        [0, 1, 3, 8],
        [0, 1, 4, 9],
        [0, 1, 2, 4, 8],
        [0, 1, 3, 9, 27]
    ]
    for U in candidates:
        if time.time() > deadline - 10:
            break
        s = score_set(U)
        if s > best_score:
            best_score = s
            best_set = U
            # Local search from this new best
            if time.time() < deadline - 5:
                refined, refined_score = local_search(best_set, deadline - 2)
                if refined_score > best_score:
                    best_score = refined_score
                    best_set = refined

    return best_set
