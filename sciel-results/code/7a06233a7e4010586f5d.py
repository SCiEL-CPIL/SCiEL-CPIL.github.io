import numpy as np
import time
import random

def construct():
    start_time = time.time()
    deadline = 145.0
    
    # Generate primes <= 1000
    primes = [p for p in range(2, 1001) if all(p % q for q in range(2, int(p ** 0.5) + 1))]
    n_primes = len(primes)
    primes_arr = np.array(primes, dtype=np.int64)
    
    W = 50000  # working window
    
    def compute_coverage(residues):
        cov = np.zeros(W + 1, dtype=np.int32)
        for i in range(n_primes):
            p = primes_arr[i]
            r = residues[i]
            start = r if r >= 1 else p
            if start <= W:
                cov[start:W+1:p] += 1
        return cov
    
    def get_max_L(cov):
        if cov[1] < 2:
            return 0
        bad = np.where(cov[1:W+1] < 2)[0]
        if len(bad) == 0:
            return W
        return bad[0]

    # Phase 1: Smart greedy initialization
    residues = np.zeros(n_primes, dtype=np.int64)
    
    # Simple greedy: for each prime, pick residue that covers 1, 2, 3... as much as possible
    # Or just random start and let local search fix it.
    # Let's use a heuristic: r_p = 1 for all? No.
    # Let's use random init.
    for i in range(n_primes):
        residues[i] = random.randint(0, primes_arr[i] - 1)
    
    cov = compute_coverage(residues)
    current_L = get_max_L(cov)
    best_L = current_L
    best_residues = residues.copy()
    
    # Local Search: Targeted Extension
    max_iters = 1000000
    it = 0
    stuck_count = 0
    
    while it < max_iters and time.time() < deadline - 2:
        if current_L >= W:
            break
            
        target = current_L + 1
        improved = False
        
        # Try to find a safe single move to cover 'target'
        # For each prime i, the required residue is target % p_i
        # Check if changing i to this residue is safe (doesn't break 1..current_L)
        
        # To speed up, we can shuffle primes order
        indices = list(range(n_primes))
        random.shuffle(indices)
        
        for i in indices:
            p_i = primes_arr[i]
            r_new = target % p_i
            r_old = residues[i]
            
            if r_new == r_old:
                continue
                
            # Check safety:
            # Positions covered by i with r_old in [1, current_L]
            # These are x such that x % p_i == r_old % p_i (handling 0)
            # If r_old == 0, positions are p_i, 2p_i, ...
            # If r_old > 0, positions are r_old, r_old+p_i, ...
            
            # We need to check if any such x has cov[x] == 2
            # If cov[x] == 2, removing i makes it 1 -> Unsafe
            
            # Iterate through x in [1, current_L] covered by i
            start_old = r_old if r_old >= 1 else p_i
            if start_old <= current_L:
                unsafe = False
                # Iterate x = start_old, start_old + p_i, ... <= current_L
                for x in range(start_old, current_L + 1, p_i):
                    if cov[x] == 2:
                        unsafe = True
                        break
                if unsafe:
                    continue
            
            # If safe, apply the move
            # Update cov
            start_old = r_old if r_old >= 1 else p_i
            start_new = r_new if r_new >= 1 else p_i
            
            if start_old <= W:
                cov[start_old:W+1:p_i] -= 1
            if start_new <= W:
                cov[start_new:W+1:p_i] += 1
            residues[i] = r_new
            
            # Recompute L (it should increase)
            new_L = get_max_L(cov)
            
            if new_L > current_L:
                current_L = new_L
                if current_L > best_L:
                    best_L = current_L
                    best_residues = residues.copy()
                improved = True
                stuck_count = 0
                it += 1
                break # Restart loop with new L
            else:
                # Revert (shouldn't happen if safety check was correct, but just in case)
                if start_new <= W:
                    cov[start_new:W+1:p_i] -= 1
                if start_old <= W:
                    cov[start_old:W+1:p_i] += 1
                residues[i] = r_old
                it += 1
        
        if not improved:
            stuck_count += 1
            # If stuck, try random perturbation
            # Pick a random prime and random residue
            i = random.randint(0, n_primes - 1)
            p = primes_arr[i]
            old_r = residues[i]
            new_r = random.randint(0, p - 1)
            if new_r == old_r:
                it += 1
                continue
                
            old_start = old_r if old_r >= 1 else p
            new_start = new_r if new_r >= 1 else p
            
            if old_start <= W:
                cov[old_start:W+1:p] -= 1
            if new_start <= W:
                cov[new_start:W+1:p] += 1
            residues[i] = new_r
            
            new_L = get_max_L(cov)
            
            # Accept if L increases or stays same (to explore)
            # If L decreases, only accept with small probability or if stuck a lot
            if new_L >= current_L:
                current_L = new_L
                if current_L > best_L:
                    best_L = current_L
                    best_residues = residues.copy()
                stuck_count = 0
            else:
                # L decreased
                # Accept if stuck_count > 100 to escape local opt
                if stuck_count > 100 and random.random() < 0.1:
                    current_L = new_L
                else:
                    # Revert
                    if new_start <= W:
                        cov[new_start:W+1:p] -= 1
                    if old_start <= W:
                        cov[old_start:W+1:p] += 1
                    residues[i] = old_r
            
            it += 1
            
        # Periodic full recompute to fix drift
        if it % 1000 == 0:
            cov = compute_coverage(residues)
            current_L = get_max_L(cov)
            if current_L > best_L:
                best_L = current_L
                best_residues = residues.copy()
                
    return [int(r) for r in best_residues]
