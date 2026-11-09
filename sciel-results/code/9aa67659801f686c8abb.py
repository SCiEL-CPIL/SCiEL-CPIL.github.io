import time
import numpy as np
import random

def construct():
    start_time = time.time()
    deadline = 115.0  # Safety margin
    
    # 1. Generate primes <= 1000
    primes = []
    for p in range(2, 1001):
        is_prime = True
        for q in range(2, int(p**0.5) + 1):
            if p % q == 0:
                is_prime = False
                break
        if is_prime:
            primes.append(p)
    
    n_primes = len(primes)
    
    # Evaluation range: We want to find L. 
    # If L can go up to 1000+, we need to check further.
    # Let's set L_max to 2000 to be safe for scoring logic, 
    # but the score is capped at 10^5 so 2000 is plenty for optimization focus.
    L_MAX = 2000
    
    def compute_L(cov):
        # cov is 0-indexed, cov[i] is count for integer i+1
        # Find first index where cov < 2
        # If all >= 2, L = L_MAX
        fail_indices = np.where(cov < 2)[0]
        if len(fail_indices) == 0:
            return L_MAX
        return fail_indices[0]
    
    def get_score(cov):
        return compute_L(cov)
        
    best_L = 0
    best_residues = None
    
    # Initial heuristic from seed
    seed_residues = np.array([((i + 2) // 2) % p for i, p in enumerate(primes)])
    
    # Evaluate seed
    cov = np.zeros(L_MAX, dtype=int)
    for i, p in enumerate(primes):
        r = seed_residues[i]
        # r is 0 <= r < p
        # x = r, r+p, r+2p...
        # In 0-indexed cov, integer x corresponds to index x-1
        # So we want indices j such that (j+1) % p == r
        # (j+1) = r + k*p => j = r - 1 + k*p
        # If r == 0, j = p-1 + k*p
        # If r > 0, j = r-1 + k*p
        
        if r == 0:
            start_idx = p - 1
        else:
            start_idx = r - 1
            
        if start_idx < L_MAX:
            cov[start_idx::p] += 1
            
    L_seed = get_score(cov)
    best_L = L_seed
    best_residues = seed_residues.copy()
    
    # Simulated Annealing
    # We will run multiple restarts.
    # Each restart runs SA for a fixed time or steps.
    
    def run_sa(initial_residues, time_limit):
        residues = initial_residues.copy()
        cov = np.zeros(L_MAX, dtype=int)
        
        # Build initial coverage
        for i, p in enumerate(primes):
            r = residues[i]
            if r == 0:
                start_idx = p - 1
            else:
                start_idx = r - 1
            if start_idx < L_MAX:
                cov[start_idx::p] += 1
                
        current_L = get_score(cov)
        best_local_L = current_L
        best_local_residues = residues.copy()
        
        T = 5.0 # Initial temperature
        T_min = 0.01
        cooling = 0.999
        
        steps = 0
        start_t = time.time()
        
        while time.time() - start_t < time_limit:
            # Check global deadline
            if time.time() > deadline:
                break
                
            # Pick a random prime to change
            i = np.random.randint(n_primes)
            p = primes[i]
            old_r = residues[i]
            
            # Pick a new residue
            # To make moves more effective, maybe pick residues that are "close" 
            # or just random. Random is standard for SA.
            new_r = np.random.randint(p)
            
            if new_r == old_r:
                continue
                
            # Calculate delta in coverage
            # Remove old
            if old_r == 0:
                old_start = p - 1
            else:
                old_start = old_r - 1
                
            if old_start < L_MAX:
                cov[old_start::p] -= 1
                
            # Add new
            if new_r == 0:
                new_start = p - 1
            else:
                new_start = new_r - 1
                
            if new_start < L_MAX:
                cov[new_start::p] += 1
                
            new_L = get_score(cov)
            
            # Acceptance criterion
            if new_L > current_L:
                # Better, accept
                current_L = new_L
                residues[i] = new_r
                if current_L > best_local_L:
                    best_local_L = current_L
                    best_local_residues = residues.copy()
            else:
                # Worse or equal
                # Use a soft score to allow moves that improve potential
                # If L is same, check if min coverage in [L, L+10] increased?
                # For simplicity, standard SA on L
                delta = current_L - new_L
                if delta == 0:
                    # Equal L, accept with some probability to explore
                    if np.random.rand() < 0.5:
                        residues[i] = new_r
                else:
                    # Worse
                    if np.random.rand() < np.exp(-delta / T):
                        residues[i] = new_r
                        current_L = new_L
            
            # Cool down
            T = max(T * cooling, T_min)
            steps += 1
            
            # If T is very low and we are stuck, maybe reheat?
            # Or just let it run.
            
        return best_local_L, best_local_residues

    # Multi-start
    # We have ~100s budget.
    # Let's do 5 restarts of 20s each? Or dynamic.
    
    # Restart 1: Seed
    # Already done
    
    # Restart 2-5: Random or Perturbed
    for restart in range(5):
        if time.time() > deadline:
            break
            
        # Initialize with perturbation of best found so far
        init_res = best_residues.copy()
        # Perturb 10% of residues
        num_perturb = n_primes // 10
        indices_to_perturb = np.random.choice(n_primes, num_perturb, replace=False)
        for i in indices_to_perturb:
            p = primes[i]
            init_res[i] = np.random.randint(p)
            
        # Run SA for a portion of remaining time
        # Let's allocate 15s per restart
        time_limit = 15.0
        
        L_local, res_local = run_sa(init_res, time_limit)
        
        if L_local > best_L:
            best_L = L_local
            best_residues = res_local
            
    # Final Polish:
    # Do a hill-climbing pass on the best result to squeeze out last bits
    # Only accept if L improves
    if time.time() < deadline - 5:
        residues = best_residues.copy()
        cov = np.zeros(L_MAX, dtype=int)
        for i, p in enumerate(primes):
            r = residues[i]
            if r == 0:
                start_idx = p - 1
            else:
                start_idx = r - 1
            if start_idx < L_MAX:
                cov[start_idx::p] += 1
                
        current_L = get_score(cov)
        
        # Hill climbing: try all single changes, accept if L increases
        improved = True
        while improved and time.time() < deadline - 2:
            improved = False
            for i in range(n_primes):
                if time.time() > deadline - 2:
                    break
                p = primes[i]
                old_r = residues[i]
                
                # Try a few random new residues
                for _ in range(5):
                    new_r = np.random.randint(p)
                    if new_r == old_r:
                        continue
                        
                    # Temporarily apply
                    if old_r == 0:
                        old_start = p - 1
                    else:
                        old_start = old_r - 1
                    if old_start < L_MAX:
                        cov[old_start::p] -= 1
                        
                    if new_r == 0:
                        new_start = p - 1
                    else:
                        new_start = new_r - 1
                    if new_start < L_MAX:
                        cov[new_start::p] += 1
                        
                    new_L = get_score(cov)
                    
                    if new_L > current_L:
                        # Accept
                        residues[i] = new_r
                        current_L = new_L
                        improved = True
                        # Revert the "temporary" logic by keeping the new state
                        # The cov array is already updated to new state
                        # We need to break inner loop and continue outer?
                        # Actually, if we accept, we stay with new_r.
                        break
                    else:
                        # Revert
                        if new_r == 0:
                            new_start = p - 1
                        else:
                            new_start = new_r - 1
                        if new_start < L_MAX:
                            cov[new_start::p] -= 1
                            
                        if old_r == 0:
                            old_start = p - 1
                        else:
                            old_start = old_r - 1
                        if old_start < L_MAX:
                            cov[old_start::p] += 1
                        
                # If we found an improvement, we break the inner loop for this i
                # and continue to next i.
                
        if current_L > best_L:
            best_L = current_L
            best_residues = residues

    return [int(r) for r in best_residues]
