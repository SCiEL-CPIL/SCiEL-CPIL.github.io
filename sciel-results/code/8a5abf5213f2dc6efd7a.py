import numpy as np
import time
import random

N = 128
LO, HI = 128, 256

def compute_stats(A):
    arr = np.zeros(N+1, dtype=np.int64)
    for a in A:
        if 0 <= a <= N:
            arr[a] = 1
    conv = np.convolve(arr, arr, mode='full')
    peak = 0
    uncovered = 0
    Aset = set(A)
    for s in range(LO, HI+1):
        ordered = int(conv[s]) if s < len(conv) else 0
        diag = 1 if (s % 2 == 0 and (s//2) in Aset) else 0
        unord = (ordered + diag) // 2
        if unord == 0:
            uncovered += 1
        else:
            if unord > peak:
                peak = unord
    return peak, uncovered

def construct():
    start = time.time()
    deadline = start + 55
    
    # Initial state: Random subset or heuristic seed
    # Let's try a random subset of size ~60-80
    # Or use the previous seed but randomize it
    A = set()
    # Start with a random set of size 70
    candidates = list(range(N+1))
    random.shuffle(candidates)
    for x in candidates[:70]:
        A.add(x)
        
    best_A = set(A)
    best_cost, best_uncovered = compute_stats(best_A)
    current_cost, current_uncovered = best_cost, best_uncovered
    
    # Simulated Annealing parameters
    T = 10.0
    cooling = 0.995
    min_T = 0.01
    
    while time.time() < deadline and T > min_T:
        # Generate a neighbor
        move_type = random.random()
        A_new = set(A)
        
        if move_type < 0.33:
            # Remove
            if A:
                a_to_remove = random.choice(list(A))
                A_new.remove(a_to_remove)
        elif move_type < 0.66:
            # Add
            a_to_add = random.randint(0, N)
            A_new.add(a_to_add)
        else:
            # Swap
            if A:
                a_to_remove = random.choice(list(A))
                a_to_add = random.randint(0, N)
                if a_to_add != a_to_remove:
                    A_new.remove(a_to_remove)
                    A_new.add(a_to_add)
            else:
                A_new.add(random.randint(0, N))
                
        if not A_new:
            continue
            
        new_peak, new_uncovered = compute_stats(A_new)
        new_cost = new_peak + 100 * new_uncovered
        
        delta = new_cost - current_cost
        
        if delta <= 0 or random.random() < np.exp(-delta / T):
            A = A_new
            current_cost = new_cost
            current_uncovered = new_uncovered
            
            if new_uncovered == 0 and new_peak < best_cost:
                best_cost = new_peak
                best_A = set(A)
                
        T *= cooling
        
    # Final check: if best_A is not covered, we might need to repair
    # But SA should have found a covered one if it exists.
    # If best_A is not covered, fall back to a valid construction
    _, best_unc = compute_stats(best_A)
    if best_unc > 0:
        # Fallback: use the previous program's logic or a simple valid set
        # The set [64, 128] is valid.
        return list(range(64, N+1))
        
    return list(best_A)
