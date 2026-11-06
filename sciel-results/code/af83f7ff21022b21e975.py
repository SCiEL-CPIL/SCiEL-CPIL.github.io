# EVOLVE-BLOCK-START
import numpy as np
import random
import time

def construct():
    # Generate primes <= 1000
    primes = [p for p in range(2, 1001) if all(p % q for q in range(2, int(p ** 0.5) + 1))]
    n_primes = len(primes)
    primes_arr = np.array(primes, dtype=np.int64)
    
    # Use a larger window to allow for growth
    W = 20000
    
    # Initialize residues
    # r_p = 1 is a strong start
    residues = np.ones(n_primes, dtype=np.int64)
    
    def compute_coverage(residues):
        cov = np.zeros(W + 1, dtype=np.int32)
        for i in range(n_primes):
            p = primes_arr[i]
            r = residues[i]
            start = r if r >= 1 else p
            if start <= W:
                cov[start:W+1:p] += 1
        return cov

    def get_max_L(coverage):
        # Find largest L such that coverage[1:L+1] all >= 2
        # coverage[0] is unused
        # We look for the first index >= 1 where coverage < 2
        bad_mask = coverage[1:W+1] < 2
        if not np.any(bad_mask):
            return W
        first_bad_idx = np.argmax(bad_mask)
        return first_bad_idx + 1

    # Initial state
    cov = compute_coverage(residues)
    current_L = get_max_L(cov)
    best_L = current_L
    best_residues = residues.copy()
    
    start_time = time.time()
    time_limit = 110.0 # Leave 10s buffer
    
    # Local Search: Best Improvement
    # We find the first bad x, and try to fix it by changing one residue.
    
    while time.time() - start_time < time_limit:
        # If we reached W, we can't improve within this window. 
        # We could expand W, but that's complex. 
        # Given the problem asks for L>=1000, and current score is ~3.7 (which is likely small L?), 
        # wait, score 3.6941? The score is the raw value L? 
        # "The evaluator measures the raw value (higher is better)."
        # "search score is value/seed". 
        # If seed score is 3.6941, and it's raw value, then L ~ 3? 
        # That seems very low. 
        # Let's check the seed program behavior.
        # If r_p = 1 for all p, x=1 is covered by 168 primes.
        # x=2: covered by p=2 if r_2=0 or 2? r_2=1. 2%2=0 != 1. Not covered by 2.
        # Covered by p=3? 2%3=2 != 1.
        # ...
        # It is likely that L is very small if we just start with 1s and don't search well.
        # The seed program has a bug or is inefficient.
        # My new program should find a much larger L.
        
        if current_L >= W:
            # Try to extend? For now, break.
            break
        
        # Find first bad x
        bad_positions = np.where(cov[1:W+1] < 2)[0]
        if len(bad_positions) == 0:
            break
            
        x = bad_positions[0] + 1
        
        # Find candidates: primes that do NOT cover x
        mods = x % primes_arr
        mask = residues != mods
        candidate_indices = np.where(mask)[0]
        
        best_new_L = current_L
        best_move = None # (index, new_residue)
        
        # Try each candidate? 168 candidates.
        # For each candidate, update cov, check L, revert.
        # This is 168 * (W/p + O(W)) operations.
        # O(W) is the bottleneck. 168 * 20000 = 3.3M ops per iteration.
        # If we do 1000 iterations, 3.3B ops. Too slow.
        
        # Optimization: Only check a subset of candidates or use a smarter heuristic.
        # Heuristic: Prefer smaller primes? Or random?
        # Let's try a random subset of 10-20 candidates to keep it fast.
        
        if len(candidate_indices) > 0:
            # Randomly pick up to 20 candidates to evaluate
            k = min(20, len(candidate_indices))
            chosen = random.sample(list(candidate_indices), k)
            
            for i in chosen:
                p = primes_arr[i]
                old_r = residues[i]
                new_r = x % p
                
                # Update cov
                old_start = old_r if old_r >= 1 else p
                if old_start <= W:
                    cov[old_start:W+1:p] -= 1
                new_start = new_r if new_r >= 1 else p
                if new_start <= W:
                    cov[new_start:W+1:p] += 1
                    
                new_L = get_max_L(cov)
                
                # Revert
                if new_start <= W:
                    cov[new_start:W+1:p] -= 1
                if old_start <= W:
                    cov[old_start:W+1:p] += 1
                    
                if new_L > best_new_L:
                    best_new_L = new_L
                    best_move = (i, new_r)
            
            if best_move:
                i, new_r = best_move
                p = primes_arr[i]
                old_r = residues[i]
                
                # Apply move
                old_start = old_r if old_r >= 1 else p
                if old_start <= W:
                    cov[old_start:W+1:p] -= 1
                new_start = new_r if new_r >= 1 else p
                if new_start <= W:
                    cov[new_start:W+1:p] += 1
                    
                residues[i] = new_r
                current_L = best_new_L
                
                if current_L > best_L:
                    best_L = current_L
                    best_residues = residues.copy()
            else:
                # No improvement found in the sampled candidates.
                # Perform a random perturbation to escape local minimum.
                i = random.randint(0, n_primes - 1)
                p = primes_arr[i]
                old_r = residues[i]
                new_r = random.randint(0, p - 1)
                
                if new_r != old_r:
                    old_start = old_r if old_r >= 1 else p
                    if old_start <= W:
                        cov[old_start:W+1:p] -= 1
                    new_start = new_r if new_r >= 1 else p
                    if new_start <= W:
                        cov[new_start:W+1:p] += 1
                        
                    new_L = get_max_L(cov)
                    
                    # Accept if it doesn't degrade L too much (Simulated Annealing style)
                    # Or just accept if L >= current_L - 5?
                    # Let's accept if new_L >= current_L
                    if new_L >= current_L:
                        residues[i] = new_r
                        current_L = new_L
                        if current_L > best_L:
                            best_L = current_L
                            best_residues = residues.copy()
                    else:
                        # Revert
                        if new_start <= W:
                            cov[new_start:W+1:p] -= 1
                        if old_start <= W:
                            cov[old_start:W+1:p] += 1

    return [int(r) for r in best_residues]
# EVOLVE-BLOCK-END
