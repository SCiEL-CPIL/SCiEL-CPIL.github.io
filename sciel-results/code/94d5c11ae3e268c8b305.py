"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import random
import time
import math

def construct():
    # Generate primes <= 1000
    primes = [p for p in range(2, 1001) if all(p % q for q in range(2, int(p ** 0.5) + 1))]
    n_primes = len(primes)
    primes_arr = np.array(primes, dtype=np.int64)
    
    # Use a larger window to allow L to grow beyond 5000
    W = 20000
    
    def compute_initial_coverage(residues):
        cov = np.zeros(W + 1, dtype=np.int32)
        for i in range(n_primes):
            p = primes_arr[i]
            r = residues[i]
            start = r if r >= 1 else p
            if start <= W:
                cov[start:W+1:p] += 1
        return cov

    def get_max_L(cov):
        # Find largest L such that coverage[1:L+1] all >= 2
        # coverage[0] is unused
        # We check indices 1 to W
        bad = np.where(cov[1:W+1] < 2)[0]
        if len(bad) == 0:
            return W
        first_bad = bad[0] + 1
        return first_bad - 1

    # Initial Heuristic: r_p = 1 for all p gives L=1 (x=1 covered by all).
    # But we can do better. 
    # Let's start with random residues and use SA.
    # Or start with r_p = 0? 
    # Let's try a few random initializations and keep the best.
    
    best_L = 0
    best_residues = None
    
    # Simulated Annealing Parameters
    T_start = 10.0
    T_end = 0.1
    cooling = 0.9995
    max_iters = 100000
    restart_threshold = 5000 # No improvement for this many steps -> restart
    
    random.seed(42)
    
    # We will run multiple SA runs with different initializations
    num_runs = 5
    
    for run in range(num_runs):
        # Initialization
        if run == 0:
            residues = np.ones(n_primes, dtype=np.int64)
        elif run == 1:
            residues = np.zeros(n_primes, dtype=np.int64)
        else:
            residues = np.array([random.randint(0, p-1) for p in primes], dtype=np.int64)
        
        cov = compute_initial_coverage(residues)
        current_L = get_max_L(cov)
        
        if current_L > best_L:
            best_L = current_L
            best_residues = residues.copy()
            
        if current_L >= W:
            break
            
        T = T_start
        no_improve_count = 0
        best_run_L = current_L
        best_run_residues = residues.copy()
        
        for it in range(max_iters):
            # Pick a random prime to change
            i = random.randint(0, n_primes - 1)
            p = primes_arr[i]
            old_r = residues[i]
            
            # Pick a new random residue
            new_r = random.randint(0, p - 1)
            if new_r == old_r:
                continue
                
            # Update coverage
            old_start = old_r if old_r >= 1 else p
            if old_start <= W:
                cov[old_start:W+1:p] -= 1
            new_start = new_r if new_r >= 1 else p
            if new_start <= W:
                cov[new_start:W+1:p] += 1
                
            new_L = get_max_L(cov)
            
            # Acceptance Criterion
            if new_L >= current_L:
                # Always accept improvement or equal
                residues[i] = new_r
                current_L = new_L
                no_improve_count = 0
                
                if current_L > best_run_L:
                    best_run_L = current_L
                    best_run_residues = residues.copy()
                    
                if current_L > best_L:
                    best_L = current_L
                    best_residues = residues.copy()
                    
                if current_L >= W:
                    break
            else:
                # Worse move, accept with probability exp((new_L - current_L) / T)
                delta = new_L - current_L # Negative
                prob = math.exp(delta / T)
                if random.random() < prob:
                    residues[i] = new_r
                    current_L = new_L
                    no_improve_count += 1
                else:
                    # Revert
                    if new_start <= W:
                        cov[new_start:W+1:p] -= 1
                    if old_start <= W:
                        cov[old_start:W+1:p] += 1
                    no_improve_count += 1
            
            # Cool down
            T *= cooling
            if T < T_end:
                T = T_end
                
            # Restart logic within the run
            if no_improve_count > restart_threshold:
                # Perturb the best solution found in this run
                residues = best_run_residues.copy()
                # Randomly change a few residues
                for _ in range(10):
                    i_pert = random.randint(0, n_primes - 1)
                    p_pert = primes_arr[i_pert]
                    residues[i_pert] = random.randint(0, p_pert - 1)
                
                cov = compute_initial_coverage(residues)
                current_L = get_max_L(cov)
                T = T_start * 0.5 # Reheat slightly
                no_improve_count = 0
                best_run_L = current_L
                best_run_residues = residues.copy()
                
                if current_L > best_L:
                    best_L = current_L
                    best_residues = residues.copy()
                
                if current_L >= W:
                    break
                    
            # Time check
            if time.time() - start_time > 100:
                break

    return [int(r) for r in best_residues]

start_time = time.time()
# EVOLVE-BLOCK-END
