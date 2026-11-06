import time
import random
import numpy as np

def _compute_peak(A):
    """Compute the maximum representation multiplicity for targets [128,256]."""
    A_set = set(A)
    A_sorted = sorted(A)
    reps = np.zeros(257, dtype=np.int32)
    for i in range(len(A_sorted)):
        for j in range(i, len(A_sorted)):
            s = A_sorted[i] + A_sorted[j]
            if 128 <= s <= 256:
                reps[s] += 1
    # Check coverage
    if np.any(reps[128:257] == 0):
        return float('inf')
    return int(np.max(reps[128:257]))

def construct():
    start_time = time.time()
    deadline = start_time + 80  # 10s safety margin
    
    random.seed(42)
    
    # Initial candidate: try to construct a good set
    # Start with a structured approach, then local search
    
    best_A = None
    best_peak = float('inf')
    
    # Strategy: start with a random-ish set of moderate size, then local search
    # Try different sizes
    
    for size in [45, 50, 55, 60]:
        if time.time() > deadline:
            break
        
        # Create initial set: spread elements across [0,128]
        # Use a lattice-like structure
        step = 128.0 / (size - 1) if size > 1 else 128
        A = sorted(set(int(round(i * step)) for i in range(size)))
        # Ensure 0 and 128 are potentially included or adjust
        # Actually, let's use a different initialization
        
        # Better init: random subset with some structure
        A = sorted(random.sample(range(0, 129), size))
        
        peak = _compute_peak(A)
        if peak < best_peak:
            best_peak = peak
            best_A = list(A)
        
        # Local search: try swapping elements
        for iteration in range(5000):
            if time.time() > deadline:
                break
            
            # Pick a random element to replace
            idx = random.randint(0, len(A) - 1)
            old_val = A[idx]
            
            # Try replacing with a random value
            new_val = random.randint(0, 128)
            if new_val == old_val:
                continue
            
            # Check if new_val is already in A
            if new_val in A:
                continue
            
            # Create new set
            A_new = A[:idx] + [new_val] + A[idx+1:]
            A_new_sorted = sorted(A_new)
            
            new_peak = _compute_peak(A_new_sorted)
            
            if new_peak < peak:
                peak = new_peak
                A = A_new_sorted
                if peak < best_peak:
                    best_peak = peak
                    best_A = list(A)
            elif new_peak == peak and random.random() < 0.1:
                # Accept equal with small probability for diversity
                A = A_new_sorted
    
    if best_A is None:
        # Fallback to seed
        return list(range(64, 129))
    
    return best_A
