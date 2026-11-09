import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = 115.0
    
    N = 51975
    
    # Get divisors
    divisors = []
    for d in range(2, N + 1):
        if N % d == 0:
            divisors.append(d)
    divisors.sort()
    
    # State
    counts = np.zeros(N, dtype=np.int32)
    uncovered = N
    residues = {m: 0 for m in divisors} # m -> r
    
    def update_add(m, r):
        nonlocal uncovered
        idx = np.arange(r, N, m)
        dec = np.sum(counts[idx] == 0)
        counts[idx] += 1
        uncovered -= dec
        
    def update_remove(m, r):
        nonlocal uncovered
        idx = np.arange(r, N, m)
        inc = np.sum(counts[idx] == 1)
        counts[idx] -= 1
        uncovered += inc
        
    # Initialize counts with random residues
    for m in divisors:
        r = random.randint(0, m-1)
        residues[m] = r
        update_add(m, r)
        
    best_uncovered = uncovered
    best_residues = residues.copy()
    
    # Coordinate Descent
    # We will run multiple passes. In each pass, we iterate through moduli (shuffled)
    # and for each modulus, we find the best residue.
    
    max_passes = 5000 # Limit by time instead
    while time.time() < deadline - 5:
        improved = False
        # Shuffle order to randomize descent direction
        order = divisors.copy()
        random.shuffle(order)
        
        for m in order:
            if time.time() > deadline - 5:
                break
                
            current_r = residues[m]
            best_r = current_r
            best_local_uncovered = uncovered
            
            # Remove current
            update_remove(m, current_r)
            
            # Try all residues
            # Optimization: If m is large, iterating all r is expensive.
            # But sum of m is small? No, m is the modulus.
            # Number of residues to try is m.
            # Cost is m * (N/m) = N.
            # So it is constant per modulus!
            for r in range(m):
                if r == current_r:
                    continue
                update_add(m, r)
                if uncovered < best_local_uncovered:
                    best_local_uncovered = uncovered
                    best_r = r
                update_remove(m, r)
            
            # Restore best
            if best_r != current_r:
                update_add(m, best_r)
                residues[m] = best_r
                improved = True
            else:
                update_add(m, current_r)
            
            # Update global best
            if uncovered < best_uncovered:
                best_uncovered = uncovered
                best_residues = residues.copy()
                
        if not improved:
            # Local minimum reached. Restart with random perturbation?
            # Or just break if we've done many passes without improvement?
            # Let's do a random perturbation: pick 2-3 moduli, randomize residues, and continue.
            if best_uncovered == 0:
                break
            
            # Perturb
            num_perturb = random.randint(1, 3)
            for _ in range(num_perturb):
                m = random.choice(divisors)
                new_r = random.randint(0, m-1)
                update_remove(m, residues[m])
                update_add(m, new_r)
                residues[m] = new_r
            
            # Reset improved flag to continue loop
            improved = True
            
    # Reconstruct best classes
    classes = []
    for m in divisors:
        classes.append([m, best_residues[m]])
        
    return classes
