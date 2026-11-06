import numpy as np
import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 50.0  # Safety margin for 60s limit
    
    n = 128
    # A is a boolean array of length n+1 (0 to 128)
    # We want to find A such that for every s in [128, 256], there exists a, b in A with a+b=s.
    # Minimize max multiplicity.
    
    # Efficient cost function using precomputed contributions
    # We maintain:
    # counts[s] = number of unordered pairs (a, b) with a <= b, a+b=s, a,b in A.
    # We can update this incrementally when toggling an element x.
    
    # Let's initialize with A = [0, 128] (all ones)
    A = np.ones(n + 1, dtype=bool)
    
    # Precompute the contribution of each element x to the counts array
    # If we add x to A:
    # For each y in A (y != x), the pair (min(x,y), max(x,y)) adds 1 to counts[x+y].
    # The pair (x, x) adds 1 to counts[2x].
    # If we remove x from A:
    # We subtract these contributions.
    
    # counts[s] for s in 0..256
    counts = np.zeros(257, dtype=np.int64)
    
    def recalculate_counts():
        # Full recalculation for initialization
        nonlocal counts
        counts = np.zeros(257, dtype=np.int64)
        for i in range(n + 1):
            if A[i]:
                for j in range(i, n + 1):
                    if A[j]:
                        counts[i + j] += 1
    
    recalculate_counts()
    
    def get_peak_and_invalid():
        # Find max count in [128, 256] and check for zeros
        target_counts = counts[128:257]
        peak = np.max(target_counts)
        invalid_count = np.sum(target_counts == 0)
        return peak, invalid_count
    
    current_peak, current_invalid = get_peak_and_invalid()
    current_score = current_peak * 1000 + current_invalid
    
    best_A = A.copy()
    best_score = current_score
    best_peak, best_invalid = current_peak, current_invalid
    
    temperature = 10.0
    cooling = 0.999
    
    while time.time() < deadline:
        # Pick a random element to toggle
        idx = random.randint(0, n)
        
        if A[idx]:
            # Remove idx
            # For each y in A (y != idx), subtract 1 from counts[idx+y]
            # Subtract 1 from counts[2*idx]
            for y in range(n + 1):
                if A[y] and y != idx:
                    counts[idx + y] -= 1
            counts[2 * idx] -= 1
            A[idx] = False
        else:
            # Add idx
            # For each y in A (y != idx), add 1 to counts[idx+y]
            # Add 1 to counts[2*idx]
            for y in range(n + 1):
                if A[y] and y != idx:
                    counts[idx + y] += 1
            counts[2 * idx] += 1
            A[idx] = True
            
        new_peak, new_invalid = get_peak_and_invalid()
        new_score = new_peak * 1000 + new_invalid
        
        delta = new_score - current_score
        
        if delta < 0 or (temperature > 0 and random.random() < np.exp(-delta / temperature)):
            current_peak, current_invalid = new_peak, new_invalid
            current_score = new_score
            
            if new_score < best_score:
                best_score = new_score
                best_peak, best_invalid = new_peak, new_invalid
                best_A = A.copy()
        else:
            # Revert
            if A[idx]:
                # It was added, so remove it
                for y in range(n + 1):
                    if A[y] and y != idx:
                        counts[idx + y] -= 1
                counts[2 * idx] -= 1
                A[idx] = False
            else:
                # It was removed, so add it back
                for y in range(n + 1):
                    if A[y] and y != idx:
                        counts[idx + y] += 1
                counts[2 * idx] += 1
                A[idx] = True
        
        temperature *= cooling
        if temperature < 0.01:
            break
            
    # Ensure we return a valid set
    if best_invalid > 0:
        # Fallback
        return list(range(n + 1))
    
    return [i for i in range(n + 1) if best_A[i]]
