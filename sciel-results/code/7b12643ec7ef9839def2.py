import numpy as np
import time
import random

def construct():
    start_time = time.time()
    deadline = 145.0  # 5s safety margin before 150s kill
    
    # Generate primes <= 1000
    primes = [p for p in range(2, 1001) if all(p % q for q in range(2, int(p ** 0.5) + 1))]
    n_primes = len(primes)
    primes_arr = np.array(primes, dtype=np.int64)
    
    W = 50000  # larger window for better search
    
    def compute_coverage(residues):
        """Compute coverage array for positions 1..W"""
        cov = np.zeros(W + 1, dtype=np.int32)
        for i in range(n_primes):
            p = primes_arr[i]
            r = residues[i]
            start = r if r >= 1 else p
            if start <= W:
                cov[start:W+1:p] += 1
        return cov
    
    def get_max_L(cov):
        """Largest L such that cov[1..L] all >= 2"""
        if cov[1] < 2:
            return 0
        bad = np.where(cov[1:W+1] < 2)[0]
        if len(bad) == 0:
            return W
        return bad[0]
    
    # Phase 1: Smart initialization
    # Strategy: assign residues to maximize coverage of small integers
    # For each prime, try all residues and pick the one that gives best marginal coverage
    residues = np.zeros(n_primes, dtype=np.int64)
    cov = np.zeros(W + 1, dtype=np.int32)
    
    # Sort primes by size (larger primes have more residue options)
    # Process smaller primes first as they have fewer options
    order = list(range(n_primes))
    
    for idx in order:
        i = idx
        p = primes_arr[i]
        best_r = 0
        best_score = -1e18
        
        # Try a subset of residues for speed
        # Focus on residues that cover positions 1..min(p, 200)
        candidates = set()
        for target in range(1, min(p, 300) + 1):
            candidates.add(target % p)
        # Add some random ones
        for _ in range(10):
            candidates.add(random.randint(0, p - 1))
        
        for r in candidates:
            start = r if r >= 1 else p
            if start > W:
                continue
            # Count how many positions in [1, W] get additional coverage
            # and how many critical positions (cov==1) get fixed
            gain = 0
            for pos in range(start, W + 1, p):
                if cov[pos] == 1:
                    gain += 10  # fixing a gap is very valuable
                elif cov[pos] == 0:
                    gain += 1  # building up coverage
                else:
                    gain += 0  # already covered >= 2, less important
            # Small penalty for very large residues (less frequent coverage in [1,W])
            if start <= W:
                freq = (W - start) // p + 1
            else:
                freq = 0
            score = gain + freq * 0.01
            
            if score > best_score:
                best_score = score
                best_r = r
        
        residues[i] = best_r
        p = primes_arr[i]
        r = best_r
        start = r if r >= 1 else p
        if start <= W:
            cov[start:W+1:p] += 1
    
    current_L = get_max_L(cov)
    best_L = current_L
    best_residues = residues.copy()
    
    # Phase 2: Iterative repair with pairwise swaps
    # Focus on fixing the first gap
    max_phase2_iters = 100000
    it = 0
    
    while it < max_phase2_iters and time.time() < deadline - 10:
        if current_L >= W:
            break
        
        bad_pos = current_L + 1
        if bad_pos > W:
            break
        
        # Strategy A: Single residue change targeting the gap
        best_single = None
        best_single_score = -1e18
        
        # Only try a subset of primes for speed
        trial_primes = random.sample(range(n_primes), min(n_primes, 50))
        for i in trial_primes:
            p = primes_arr[i]
            old_r = residues[i]
            new_r = bad_pos % p
            if new_r == old_r:
                continue
            
            old_start = old_r if old_r >= 1 else p
            new_start = new_r if new_r >= 1 else p
            
            # Estimate impact
            damage = 0
            gain = 0
            if old_start <= W:
                for pos in range(old_start, bad_pos + 1, p):
                    if cov[pos] == 2:
                        damage += 1
            if new_start <= W:
                for pos in range(new_start, bad_pos + 1, p):
                    if cov[pos] == 1:
                        gain += 1
            
            score = gain * 10 - damage * 5 + 100  # 100 for covering bad_pos
            
            if score > best_single_score:
                best_single_score = score
                best_single = (i, new_r, old_start, new_start)
        
        # Strategy B: Pairwise swap
        best_swap = None
        best_swap_score = -1e18
        
        trial_pairs = random.sample(range(n_primes), min(n_primes, 30))
        for i in trial_pairs:
            p_i = primes_arr[i]
            old_r_i = residues[i]
            new_r_i = bad_pos % p_i
            if new_r_i == old_r_i:
                continue
            
            # Find a partner j to swap with
            for j in random.sample(range(n_primes), min(n_primes, 10)):
                if j == i:
                    continue
                p_j = primes_arr[j]
                old_r_j = residues[j]
                # Swap: i gets new_r_i, j gets old_r_i (or some other combo)
                # Actually, let's try: i gets new_r_i, j keeps its residue but we check if that helps
                # Better: try all combos where i covers bad_pos
                new_r_j = old_r_j  # j stays the same for now
                
                old_start_i = old_r_i if old_r_i >= 1 else p_i
                new_start_i = new_r_i if new_r_i >= 1 else p_i
                
                # Estimate: does this help?
                damage_i = 0
                gain_i = 0
                if old_start_i <= W:
                    for pos in range(old_start_i, bad_pos + 1, p_i):
                        if cov[pos] == 2:
                            damage_i += 1
                if new_start_i <= W:
                    for pos in range(new_start_i, bad_pos + 1, p_i):
                        if cov[pos] == 1:
                            gain_i += 1
                
                score = gain_i * 10 - damage_i * 5 + 50
                
                if score > best_swap_score:
                    best_swap_score = score
                    best_swap = (i, new_r_i, old_start_i, new_start_i)
        
        # Choose best action
        if best_single and best_single_score > 0:
            i, new_r, old_start, new_start = best_single
            p = primes_arr[i]
            old_r = residues[i]
            
            if old_start <= W:
                cov[old_start:W+1:p] -= 1
            if new_start <= W:
                cov[new_start:W+1:p] += 1
            
            residues[i] = new_r
            new_L = get_max_L(cov)
            
            if new_L >= current_L:
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
                residues[i] = old_r
        
        elif best_swap and best_swap_score > 0:
            i, new_r, old_start, new_start = best_swap
            p = primes_arr[i]
            old_r = residues[i]
            
            if old_start <= W:
                cov[old_start:W+1:p] -= 1
            if new_start <= W:
                cov[new_start:W+1:p] += 1
            
            residues[i] = new_r
            new_L = get_max_L(cov)
            
            if new_L >= current_L:
                current_L = new_L
                if current_L > best_L:
                    best_L = current_L
                    best_residues = residues.copy()
            else:
                if new_start <= W:
                    cov[new_start:W+1:p] -= 1
                if old_start <= W:
                    cov[old_start:W+1:p] += 1
                residues[i] = old_r
        else:
            # Stuck - try random perturbation to escape
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
            
            new_L = get_max_L(cov)
            
            # Accept with some probability even if worse (simulated annealing)
            if new_L >= current_L or random.random() < 0.1:
                current_L = new_L
                if current_L > best_L:
                    best_L = current_L
                    best_residues = residues.copy()
            else:
                if new_start <= W:
                    cov[new_start:W+1:p] -= 1
                if old_start <= W:
                    cov[old_start:W+1:p] += 1
        
        it += 1
        
        # Periodic recomputation
        if it % 1000 == 0:
            cov = compute_coverage(residues)
            current_L = get_max_L(cov)
            if current_L > best_L:
                best_L = current_L
                best_residues = residues.copy()
    
    # Phase 3: Final polish
    if time.time() < deadline - 3:
        for _ in range(20):
            if time.time() > deadline - 2:
                break
            for i in range(n_primes):
                if time.time() > deadline - 1:
                    break
                p = primes_arr[i]
                old_r = residues[i]
                target = current_L + 1
                if target <= W:
                    new_r = target % p
                    if new_r != old_r:
                        old_start = old_r if old_r >= 1 else p
                        new_start = new_r if new_r >= 1 else p
                        if old_start <= W:
                            cov[old_start:W+1:p] -= 1
                        if new_start <= W:
                            cov[new_start:W+1:p] += 1
                        new_L = get_max_L(cov)
                        if new_L > current_L:
                            residues[i] = new_r
                            current_L = new_L
                            if current_L > best_L:
                                best_L = current_L
                                best_residues = residues.copy()
                        else:
                            if new_start <= W:
                                cov[new_start:W+1:p] -= 1
                            if old_start <= W:
                                cov[old_start:W+1:p] += 1
    
    return [int(r) for r in best_residues]
