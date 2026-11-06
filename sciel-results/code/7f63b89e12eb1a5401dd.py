"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import random
import math

def construct():
    # Generate primes <= 1000
    primes = [p for p in range(2, 1001) if all(p % q for q in range(2, int(p ** 0.5) + 1))]
    n_primes = len(primes)
    primes_arr = np.array(primes, dtype=np.int64)
    
    # Use a sufficiently large window to capture the true L, 
    # but not so large that updates are too slow.
    # The goal is L >= 1000. Current best is likely in the thousands.
    # Let's use W = 50000 to be safe, but optimize the loop.
    W = 50000
    
    # Initialize residues randomly
    residues = np.array([random.randint(0, p - 1) for p in primes], dtype=np.int64)
    
    def compute_initial_coverage(residues):
        cov = np.zeros(W + 1, dtype=np.int32)
        for i in range(n_primes):
            p = primes_arr[i]
            r = residues[i]
            # Positions x in [1, W] such that x % p == r
            # x = r + k*p
            # First x >= 1
            if r == 0:
                start = p
            else:
                start = r
            if start <= W:
                cov[start:W+1:p] += 1
        return cov

    def get_max_L(cov):
        # Find largest L such that coverage[1:L+1] all >= 2
        # We look for the first index in 1..W where cov < 2
        bad = np.where(cov[1:W+1] < 2)[0]
        if len(bad) == 0:
            return W
        return bad[0] + 1

    cov = compute_initial_coverage(residues)
    current_L = get_max_L(cov)
    best_L = current_L
    best_residues = residues.copy()
    
    # Simulated Annealing Parameters
    T = 10.0
    T_min = 0.01
    cooling_rate = 0.999
    max_iters = 20000
    random.seed(42)
    
    for it in range(max_iters):
        if T < T_min:
            break
            
        # Pick a random prime to modify
        i = random.randint(0, n_primes - 1)
        p = primes_arr[i]
        old_r = residues[i]
        
        # Pick a new random residue
        new_r = random.randint(0, p - 1)
        if new_r == old_r:
            continue
            
        # Update coverage array
        # Remove old coverage
        old_start = old_r if old_r >= 1 else p
        if old_start <= W:
            cov[old_start:W+1:p] -= 1
            
        # Add new coverage
        new_start = new_r if new_r >= 1 else p
        if new_start <= W:
            cov[new_start:W+1:p] += 1
            
        # Compute new L
        new_L = get_max_L(cov)
        
        # Acceptance criterion
        delta_L = new_L - current_L
        if delta_L > 0:
            accept = True
        else:
            # Accept with probability exp(delta_L / T)
            # delta_L is negative
            prob = math.exp(delta_L / T)
            accept = random.random() < prob
            
        if accept:
            residues[i] = new_r
            current_L = new_L
            if current_L > best_L:
                best_L = current_L
                best_residues = residues.copy()
        else:
            # Revert coverage
            if new_start <= W:
                cov[new_start:W+1:p] -= 1
            if old_start <= W:
                cov[old_start:W+1:p] += 1
                
        # Cool down
        T *= cooling_rate
        
    return [int(r) for r in best_residues]
# EVOLVE-BLOCK-END
