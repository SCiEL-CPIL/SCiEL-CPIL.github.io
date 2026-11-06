import numpy as np
import time
import random

N = 128
LO, HI = 128, 256

def compute_stats(A):
    # A: set of ints in [0,128]
    arr = np.zeros(N+1, dtype=np.int64)
    for a in A:
        if 0 <= a <= N:
            arr[a] = 1
    conv = np.convolve(arr, arr, mode='full')
    peak = 0
    covered = True
    Aset = set(A)
    for s in range(LO, HI+1):
        if s >= len(conv):
            ordered = 0
        else:
            ordered = int(conv[s])
        diag = 1 if (s % 2 == 0 and (s//2) in Aset) else 0
        unord = (ordered + diag) // 2
        if unord == 0:
            covered = False
        if unord > peak:
            peak = unord
    return peak, covered

def construct():
    start = time.time()
    deadline = start + 55  # leave 5s margin
    
    # Initial: Use a dense set to ensure coverage, then anneal
    # The seed uses range(64, 129). Let's start there.
    A = set(range(64, N+1))
    
    # Ensure it's valid
    peak, covered = compute_stats(A)
    if not covered:
        # Fallback to full set if somehow not covered (unlikely for 64..128)
        A = set(range(N+1))
        peak, covered = compute_stats(A)
        
    best_A = set(A)
    best_peak = peak
    current_peak = peak
    current_covered = covered
    current_cost = current_peak if current_covered else 1000 + current_peak
    
    # Simulated Annealing parameters
    T = 10.0
    T_min = 0.01
    alpha = 0.99
    
    while T > T_min and time.time() < deadline:
        # Generate a neighbor
        # Choose move type: 0=remove, 1=add, 2=swap
        move_type = random.randint(0, 2)
        A_new = set(A)
        
        if move_type == 0:
            if A:
                a = random.choice(list(A))
                A_new.remove(a)
        elif move_type == 1:
            candidates = [i for i in range(N+1) if i not in A]
            if candidates:
                a = random.choice(candidates)
                A_new.add(a)
        else: # swap
            if A:
                a = random.choice(list(A))
                candidates = [i for i in range(N+1) if i not in A]
                if candidates:
                    b = random.choice(candidates)
                    A_new.remove(a)
                    A_new.add(b)
                    
        if not A_new:
            continue
            
        p_new, c_new = compute_stats(A_new)
        cost_new = p_new if c_new else 1000 + p_new
        
        # Acceptance criterion
        delta = cost_new - current_cost
        if delta < 0 or (random.random() < np.exp(-delta / T)):
            A = A_new
            current_peak = p_new
            current_covered = c_new
            current_cost = cost_new
            
            if current_covered and current_peak < best_peak:
                best_peak = current_peak
                best_A = set(A)
                
        T *= alpha
        
    if not best_A:
        return list(range(64, N+1))
    return list(best_A)
