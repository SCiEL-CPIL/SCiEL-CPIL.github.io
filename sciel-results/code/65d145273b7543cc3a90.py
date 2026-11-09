# EVOLVE-BLOCK-START
import time
import random
import numpy as np
from math import log

KNOWN_GOLOMB = {
    2: [0, 1],
    3: [0, 1, 3],
    4: [0, 1, 4, 6],
    5: [0, 1, 4, 10, 12],
    6: [0, 1, 4, 10, 18, 23],
    7: [0, 1, 7, 12, 18, 23, 31],
    8: [0, 1, 7, 12, 18, 25, 34, 41],
    9: [0, 1, 7, 12, 18, 25, 34, 44, 51],
    10: [0, 1, 7, 12, 18, 25, 34, 44, 52, 62],
    11: [0, 1, 7, 12, 18, 25, 34, 44, 52, 62, 70],
    12: [0, 1, 7, 12, 18, 25, 34, 44, 52, 62, 70, 79],
}

def compute_score(U):
    """Compute the raw score for a given set U."""
    U = sorted(set(U))
    if 0 not in U:
        return 0.0
    n = len(U)
    if n == 0:
        return 0.0
    max_u = max(U)
    if max_u == 0:
        return 0.0
    
    # Compute sumset U+U
    sumset = set()
    for a in U:
        for b in U:
            sumset.add(a + b)
    
    # Compute difference set U-U
    diffset = set()
    for a in U:
        for b in U:
            diffset.add(a - b)
    
    ratio = len(diffset) / len(sumset)
    if ratio <= 0:
        return 0.0
    
    denominator = log(2 * max_u + 1)
    if denominator <= 0:
        return 0.0
    
    return 1.0 + log(ratio) / denominator

def compute_score_fft(U):
    """Compute score using FFT for O(M log M) complexity."""
    U = sorted(set(U))
    if 0 not in U:
        return 0.0
    n = len(U)
    if n == 0:
        return 0.0
    max_u = max(U)
    if max_u == 0:
        return 0.0
    
    # For sumset: use FFT convolution
    # Create indicator array
    size = 2 * max_u + 1
    # Next power of 2 for efficient FFT
    fft_size = 1
    while fft_size < size:
        fft_size *= 2
    
    indicator = np.zeros(fft_size)
    for u in U:
        indicator[u] = 1.0
    
    # Sumset via convolution
    fft_ind = np.fft.fft(indicator)
    conv = np.fft.ifft(fft_ind * fft_ind)
    sumset_count = int(np.sum(np.round(np.abs(conv)) > 0.5))
    
    # Difference set: |U-U| = number of distinct differences
    # U-U contains values from -(max_u) to max_u
    # Count distinct differences
    diffset = set()
    for a in U:
        for b in U:
            diffset.add(a - b)
    diffset_count = len(diffset)
    
    ratio = diffset_count / sumset_count
    if ratio <= 0:
        return 0.0
    
    denominator = log(2 * max_u + 1)
    if denominator <= 0:
        return 0.0
    
    return 1.0 + log(ratio) / denominator

def simulated_annealing(U, time_limit=100, max_value=100):
    """Simulated annealing to optimize set U."""
    start_time = time.time()
    U = sorted(set(U))
    best_U = list(U)
    best_score = compute_score(U)
    current_U = list(U)
    current_score = best_score
    
    # Temperature schedule
    T_start = 1.0
    T_end = 0.001
    cooling_rate = 0.999
    
    # Max value for new elements
    max_val = max_value
    
    n_steps = 0
    while (time.time() - start_time) < time_limit:
        T = T_start * (cooling_rate ** n_steps)
        if T < T_end:
            break
        
        n_steps += 1
        n = len(current_U)
        
        # Choose a perturbation type
        pert_type = random.random()
        
        if pert_type < 0.4:
            # Shift an existing element
            if n < 2:
                continue
            i = random.randint(1, n - 1)  # Keep 0 fixed
            old_val = current_U[i]
            # Try shifting to a nearby value
            delta = random.randint(-10, 10)
            new_val = old_val + delta
            if new_val < 0 or new_val == old_val or new_val in current_U:
                continue
            if new_val > max_val:
                continue
            
            new_U = list(current_U)
            new_U[i] = new_val
            new_U.sort()
            
        elif pert_type < 0.7:
            # Add a new element
            if n >= 15:  # Limit size
                continue
            new_val = random.randint(1, max_val)
            if new_val in current_U:
                continue
            
            new_U = list(current_U)
            new_U.append(new_val)
            new_U.sort()
            
        elif pert_type < 0.9:
            # Remove an element (keep 0)
            if n < 3:
                continue
            i = random.randint(1, n - 1)
            new_U = list(current_U)
            new_U.pop(i)
            
        else:
            # Random restart with small set
            new_U = [0]
            for _ in range(random.randint(2, 6)):
                v = random.randint(1, max_val)
                if v not in new_U:
                    new_U.append(v)
            new_U.sort()
        
        if len(new_U) < 2:
            continue
        if new_U[0] != 0:
            continue
        
        new_score = compute_score(new_U)
        
        # Acceptance criterion
        if new_score > current_score:
            current_U = new_U
            current_score = new_score
            if new_score > best_score:
                best_score = new_score
                best_U = list(new_U)
        else:
            # Accept with probability exp(-(current_score - new_score) / T)
            delta = current_score - new_score
            if delta < 10 * T:  # Avoid overflow
                prob = np.exp(-delta / max(T, 1e-10))
                if random.random() < prob:
                    current_U = new_U
                    current_score = new_score
    
    return best_U, best_score

def construct():
    start_time = time.time()
    deadline = start_time + 145  # Leave 5s safety margin
    
    best_U = [0, 1, 3]
    best_score = compute_score(best_U)
    
    # Try known Golomb rulers
    for n in range(2, 13):
        if time.time() > deadline:
            break
        if n in KNOWN_GOLOMB:
            U = list(KNOWN_GOLOMB[n])
            score = compute_score(U)
            if score > best_score:
                best_score = score
                best_U = U
    
    # Simulated annealing from multiple starting points
    starting_sets = [
        [0, 1, 3],
        [0, 1, 4, 6],
        [0, 1, 4, 10, 12],
        [0, 1, 4, 9, 11],
        [0, 1, 3, 6, 9],
        [0, 2, 3, 7, 10],
        [0, 1, 2, 4, 7, 11],
        [0, 1, 3, 7, 12, 18],
    ]
    
    # Also try the best Golomb ruler
    best_golomb = None
    best_golomb_score = 0
    for n in range(2, 13):
        if n in KNOWN_GOLOMB:
            U = list(KNOWN_GOLOMB[n])
            score = compute_score(U)
            if score > best_golomb_score:
                best_golomb_score = score
                best_golomb = U
    
    if best_golomb and best_golomb not in starting_sets:
        starting_sets.append(best_golomb)
    
    for start_set in starting_sets:
        if time.time() > deadline - 10:
            break
        if len(start_set) < 2:
            continue
        U = list(start_set)
        U.sort()
        if U[0] != 0:
            continue
        
        score = compute_score(U)
        if score > best_score:
            best_score = score
            best_U = U
        
        # Run simulated annealing with remaining time
        remaining = deadline - time.time() - 5
        if remaining > 5:
            improved_U, improved_score = simulated_annealing(
                U, 
                time_limit=remaining,
                max_value=80
            )
            if improved_score > best_score:
                best_score = improved_score
                best_U = improved_U
    
    # Final polish: try local search on the best solution
    if time.time() < deadline - 5:
        remaining = deadline - time.time() - 2
        if remaining > 2:
            improved_U, improved_score = simulated_annealing(
                best_U,
                time_limit=remaining,
                max_value=60
            )
            if improved_score > best_score:
                best_score = improved_score
                best_U = improved_U
    
    return best_U
# EVOLVE-BLOCK-END
