import numpy as np
import time
import random

def construct():
    start_time = time.time()
    deadline = 140.0  # 10s safety margin before 150s kill
    
    # Generate primes <= 1000
    primes = [p for p in range(2, 1001) if all(p % q for q in range(2, int(p ** 0.5) + 1))]
    n_primes = len(primes)
    primes_arr = np.array(primes, dtype=np.int64)
    
    # We'll work with a window W. The evaluator checks up to 10^5, but
    # we can only track coverage within our window.
    W = 20000
    
    # Smart initialization: for each prime, choose residue that covers position 1
    # r_p = 1 % p gives coverage at x=1, but we can do better by spreading
    # Actually, let's start with r_p = 1 for all, then improve greedily
    residues = np.ones(n_primes, dtype=np.int64)
    
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
        # Find first position where cov < 2
        bad = np.where(cov[1:W+1] < 2)[0]
        if len(bad) == 0:
            return W
        return bad[0]
    
    # Phase 1: Greedy initialization
    # Start with all residues = 1, then greedily improve
    cov = compute_coverage(residues)
    current_L = get_max_L(cov)
    best_L = current_L
    best_residues = residues.copy()
    
    # Greedy: repeatedly find the first gap, and for each prime, try to cover it
    # Priority: primes that can cover the gap with minimal disruption
    max_greedy_iters = 50
    for _ in range(max_greedy_iters):
        if current_L >= W:
            break
        if time.time() > deadline - 30:
            break
            
        # Find the first bad position
        bad_pos = current_L + 1  # first position with cov < 2
        if bad_pos > W:
            break
        
        # For each prime, compute what residue would cover bad_pos
        # and evaluate the impact
        best_prime_idx = -1
        best_new_r = -1
        best_score = -1e18
        
        for i in range(n_primes):
            p = primes_arr[i]
            old_r = residues[i]
            # Residue that covers bad_pos: bad_pos % p
            new_r = bad_pos % p
            if new_r == old_r:
                continue
            
            # Compute impact: how many positions in [1, bad_pos] would we lose/gain?
            # Removing old_r: positions old_r, old_r+p, ... <= bad_pos
            # Adding new_r: positions new_r, new_r+p, ... <= bad_pos
            
            # Count how many of the old positions had cov == 2 (would become 1)
            old_start = old_r if old_r >= 1 else p
            new_start = new_r if new_r >= 1 else p
            
            old_positions = []
            if old_start <= W:
                old_positions = list(range(old_start, bad_pos + 1, p))
            
            new_positions = []
            if new_start <= W:
                new_positions = list(range(new_start, bad_pos + 1, p))
            
            # Score: maximize coverage at bad_pos, minimize damage to earlier positions
            # Damage: positions where cov would drop below 2
            damage = 0
            for pos in old_positions:
                if pos <= bad_pos and cov[pos] == 2:
                    damage += 1
            
            # Gain: positions where cov was 1 and becomes 2
            gain = 0
            for pos in new_positions:
                if pos <= bad_pos and cov[pos] == 1:
                    gain += 1
            
            # Also check if new_r covers bad_pos (it should by construction)
            score = gain * 10 - damage * 5 + (1 if bad_pos in new_positions else 0) * 100
            
            if score > best_score:
                best_score = score
                best_prime_idx = i
                best_new_r = new_r
        
        if best_prime_idx == -1:
            break
        
        # Apply the change
        i = best_prime_idx
        p = primes_arr[i]
        old_r = residues[i]
        new_r = best_new_r
        
        old_start = old_r if old_r >= 1 else p
        new_start = new_r if new_r >= 1 else p
        
        if old_start <= W:
            cov[old_start:W+1:p] -= 1
        if new_start <= W:
            cov[new_start:W+1:p] += 1
        
        residues[i] = new_r
        new_L = get_max_L(cov)
        
        if new_L > best_L:
            best_L = new_L
            best_residues = residues.copy()
        
        current_L = new_L
    
    # Phase 2: Targeted local search
    # Focus on improving the frontier
    T = 5.0
    T_min = 0.1
    cooling = 0.9998
    
    it = 0
    max_iters = 200000
    
    while it < max_iters and time.time() < deadline - 5:
        # Pick a prime to modify
        i = random.randint(0, n_primes - 1)
        p = primes_arr[i]
        old_r = residues[i]
        
        # Strategy: 60% target the frontier, 40% random
        if current_L < W and random.random() < 0.6:
            bad_pos = current_L + 1
            if bad_pos <= W:
                new_r = bad_pos % p
            else:
                new_r = random.randint(0, p - 1)
        else:
            new_r = random.randint(0, p - 1)
        
        if new_r == old_r:
            it += 1
            continue
        
        # Update coverage incrementally
        old_start = old_r if old_r >= 1 else p
        new_start = new_r if new_r >= 1 else p
        
        if old_start <= W:
            cov[old_start:W+1:p] -= 1
        if new_start <= W:
            cov[new_start:W+1:p] += 1
        
        new_L = get_max_L(cov)
        
        # Acceptance: always accept improvements, occasionally accept neutral/slight worsenings
        delta = new_L - current_L
        if delta >= 0 or random.random() < np.exp(delta / T):
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
        
        it += 1
        T = max(T_min, T * cooling)
        
        # Periodic full recomputation to fix any drift
        if it % 500 == 0:
            cov = compute_coverage(residues)
            current_L = get_max_L(cov)
    
    # Phase 3: Final polish - try all primes against the frontier
    if time.time() < deadline - 2:
        for i in range(n_primes):
            if time.time() > deadline - 1:
                break
            p = primes_arr[i]
            old_r = residues[i]
            # Try residue that covers current_L + 1
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
                        # Revert
                        if new_start <= W:
                            cov[new_start:W+1:p] -= 1
                        if old_start <= W:
                            cov[old_start:W+1:p] += 1
    
    return [int(r) for r in best_residues]
