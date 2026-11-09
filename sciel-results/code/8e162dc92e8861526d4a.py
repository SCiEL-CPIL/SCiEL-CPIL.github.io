import time
import numpy as np
from math import log

def compute_score(U):
    """Compute the score for a given set U."""
    U = sorted(set(U))
    if 0 not in U:
        U = sorted(set(U) | {0})
    n = len(U)
    if n == 0:
        return 0.0
    max_u = max(U)
    if max_u == 0:
        return 0.0
    
    # Compute sumset and difference set using FFT for speed
    arr = np.zeros(2 * max_u + 1, dtype=np.float64)
    for x in U:
        arr[x] = 1.0
    
    # Sumset: convolution of arr with itself
    sumset_fft = np.fft.fft(arr, n=4 * max_u + 4)
    sumset_conv = np.real(np.fft.ifft(sumset_fft * sumset_fft))
    sumset_count = np.sum(sumset_conv > 0.5)
    
    # Difference set: convolution of arr with reversed arr
    arr_rev = arr[::-1]
    diffset_fft = np.fft.fft(arr_rev, n=4 * max_u + 4)
    diffset_conv = np.real(np.fft.ifft(diffset_fft * sumset_fft))
    diffset_count = np.sum(diffset_conv > 0.5)
    
    ratio = diffset_count / sumset_count
    if ratio <= 1:
        return 1.0
    
    denominator = log(2 * max_u + 1)
    if denominator <= 0:
        return 0.0
    
    return 1.0 + log(ratio) / denominator

def construct():
    start_time = time.time()
    deadline = start_time + 145  # Safety margin before 150s limit
    
    # Start with a good initial set - use squares which have good properties
    # {0, 1, 4, 9, 16, 25, 36, 49} scored well
    best_set = [0, 1, 4, 9, 16, 25, 36, 49]
    best_score = compute_score(best_set)
    
    # Simulated annealing
    current_set = list(best_set)
    current_score = best_score
    
    T = 1.0  # Temperature
    cooling_rate = 0.999
    
    while time.time() < deadline:
        # Generate a perturbation
        import random
        action = random.choice(['add', 'remove', 'shift', 'swap'])
        
        new_set = list(current_set)
        max_val = max(new_set) if new_set else 0
        
        if action == 'add' and len(new_set) < 500:
            # Add a new element
            if random.random() < 0.5:
                # Add near existing elements
                base = random.choice(new_set)
                offset = random.randint(1, max(1, max_val // 10))
                new_elem = base + offset
            else:
                # Add random element
                new_elem = random.randint(1, max_val * 2 + 10)
            if new_elem not in new_set and new_elem >= 0:
                new_set.append(new_elem)
        
        elif action == 'remove' and len(new_set) > 2:
            # Remove a random element (never remove 0)
            removable = [x for x in new_set if x != 0]
            if removable:
                new_set.remove(random.choice(removable))
        
        elif action == 'shift':
            # Shift a random element
            shiftable = [x for x in new_set if x != 0]
            if shiftable:
                idx = new_set.index(random.choice(shiftable))
                new_val = max(1, new_set[idx] + random.randint(-max(1, max_val//20), max(1, max_val//20)))
                if new_val not in new_set:
                    new_set[idx] = new_val
        
        elif action == 'swap':
            # Swap two elements
            if len(new_set) >= 3:
                i, j = random.sample(range(len(new_set)), 2)
                if new_set[i] != 0 or new_set[j] != 0:
                    new_set[i], new_set[j] = new_set[j], new_set[i]
        
        if new_set == current_set:
            continue
        
        new_set = sorted(set(new_set))
        if 0 not in new_set:
            continue
        
        new_score = compute_score(new_set)
        
        # Accept if better, or with probability based on temperature
        if new_score > current_score:
            current_set = new_set
            current_score = new_score
            if new_score > best_score:
                best_score = new_score
                best_set = list(new_set)
        else:
            delta = new_score - current_score
            if delta < 0 and random.random() < np.exp(delta / T):
                current_set = new_set
                current_score = new_score
        
        T *= cooling_rate
        if T < 0.001:
            T = 0.001
    
    return sorted(set(best_set))
