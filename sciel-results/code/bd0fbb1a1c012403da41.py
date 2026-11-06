"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import random
import time

def construct():
    # Generate primes <= 1000
    primes = [p for p in range(2, 1001) if all(p % q for q in range(2, int(p ** 0.5) + 1))]
    n_primes = len(primes)
    primes_arr = np.array(primes, dtype=np.int64)
    
    # Target window size. We want to maximize L.
    # Let's aim for W = 5000 initially. If we reach it, we can try to extend.
    W = 5000
    
    # Initialize residues
    # Heuristic: Set r_p = 1 for all p. This makes x=1 covered by all primes.
    # This is a very strong start for the beginning of the interval.
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
        bad = np.where(coverage[1:W+1] < 2)[0]
        if len(bad) == 0:
            return W
        first_bad = bad[0] + 1
        return first_bad - 1

    current_L = get_max_L(compute_coverage(residues))
    best_L = current_L
    best_residues = residues.copy()
    
    # Greedy Bottleneck Local Search
    # We iterate, finding the first uncovered (or singly covered) point,
    # and fixing it by adjusting a prime that doesn't cover it.
    
    max_iters = 50000
    random.seed(42)
    
    for it in range(max_iters):
        cov = compute_coverage(residues)
        current_L = get_max_L(cov)
        
        if current_L > best_L:
            best_L = current_L
            best_residues = residues.copy()
            
        if current_L >= W:
            # We covered the whole window. We can try to extend or break.
            # For now, break and return. The evaluator will check further if needed.
            # Actually, the evaluator computes L up to 10^5.
            # If we cover W=5000, we should try to extend the window logic?
            # The current structure fixes W. 
            # Let's just break and return best.
            break
        
        # Find the first bad x
        bad_positions = np.where(cov[1:W+1] < 2)[0]
        if len(bad_positions) == 0:
            break
        x = bad_positions[0] + 1
        
        # We need to increase coverage at x.
        # Find primes p such that x % p != residues[i]
        # If x % p == residues[i], the prime already covers x.
        # We want to pick a prime that does NOT cover x, and flip it to cover x.
        
        # Efficient way to find such primes:
        # For each i, check if residues[i] == x % primes_arr[i]
        # This is O(N).
        
        candidates = []
        for i in range(n_primes):
            p = primes_arr[i]
            r = residues[i]
            if r != x % p:
                candidates.append(i)
                
        if not candidates:
            # All primes cover x? Then coverage[x] should be n_primes = 168 >= 2.
            # So this case shouldn't happen if cov[x] < 2.
            # Unless there's a bug in coverage calculation.
            # Let's try a random perturbation to escape.
            i = random.randint(0, n_primes - 1)
            p = primes_arr[i]
            residues[i] = random.randint(0, p - 1)
            continue
        
        # Pick a random candidate
        i = random.choice(candidates)
        p = primes_arr[i]
        residues[i] = x % p
        
    # After local search, verify best_L
    # Note: The local search might oscillate or get stuck.
    # Let's do a few restarts with different seeds or initializations if time permits.
    # Given 120s limit, 50000 iters * 168 primes * O(W/p) updates...
    # Wait, compute_coverage is O(W * N/p_avg). W=5000, N=168.
    # Sum(5000/p) approx 5000 * log(log(1000)) ~ 5000 * 2 = 10000 operations per coverage calc.
    # 50000 iters * 10000 ops = 5 * 10^8 ops. This might be too slow in Python/NumPy if not optimized.
    # NumPy vectorization helps, but the loop inside compute_coverage is Python.
    
    # Optimization: Update coverage incrementally.
    
    # Let's rewrite the loop to use incremental updates.
    
    def compute_initial_coverage(residues):
        cov = np.zeros(W + 1, dtype=np.int32)
        for i in range(n_primes):
            p = primes_arr[i]
            r = residues[i]
            start = r if r >= 1 else p
            if start <= W:
                cov[start:W+1:p] += 1
        return cov

    residues = np.ones(n_primes, dtype=np.int64)
    cov = compute_initial_coverage(residues)
    current_L = get_max_L(cov)
    best_L = current_L
    best_residues = residues.copy()
    
    max_iters = 20000
    random.seed(42)
    
    for it in range(max_iters):
        if current_L > best_L:
            best_L = current_L
            best_residues = residues.copy()
            
        if current_L >= W:
            break
        
        bad_positions = np.where(cov[1:W+1] < 2)[0]
        if len(bad_positions) == 0:
            break
            
        x = bad_positions[0] + 1
        
        # Find candidates
        # We need primes where residues[i] != x % primes_arr[i]
        # Vectorized check:
        mods = x % primes_arr
        # residues is int64, mods is int64
        mask = residues != mods
        candidate_indices = np.where(mask)[0]
        
        if len(candidate_indices) == 0:
            # Perturbation
            i = random.randint(0, n_primes - 1)
            p = primes_arr[i]
            old_r = residues[i]
            new_r = random.randint(0, p - 1)
            if new_r == old_r: continue
            
            # Update cov
            old_start = old_r if old_r >= 1 else p
            if old_start <= W:
                cov[old_start:W+1:p] -= 1
            new_start = new_r if new_r >= 1 else p
            if new_start <= W:
                cov[new_start:W+1:p] += 1
            
            new_L = get_max_L(cov)
            if new_L < current_L:
                # Revert
                if new_start <= W:
                    cov[new_start:W+1:p] -= 1
                if old_start <= W:
                    cov[old_start:W+1:p] += 1
                # Keep old state
            else:
                residues[i] = new_r
                current_L = new_L
            continue
            
        i = candidate_indices[random.randint(0, len(candidate_indices)-1)]
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
        
        # Accept if it improves or maintains current_L?
        # Greedy: accept if new_L >= current_L
        # But we might want to accept slightly worse to escape?
        # Let's try greedy first.
        if new_L >= current_L:
            residues[i] = new_r
            current_L = new_L
        else:
            # Revert
            if new_start <= W:
                cov[new_start:W+1:p] -= 1
            if old_start <= W:
                cov[old_start:W+1:p] += 1
            # Try a random move to escape?
            # If stuck, do a random perturbation
            if it % 100 == 0:
                 i_rand = random.randint(0, n_primes - 1)
                 p_rand = primes_arr[i_rand]
                 old_r_rand = residues[i_rand]
                 new_r_rand = random.randint(0, p_rand - 1)
                 if new_r_rand != old_r_rand:
                     old_start_rand = old_r_rand if old_r_rand >= 1 else p_rand
                     if old_start_rand <= W:
                         cov[old_start_rand:W+1:p_rand] -= 1
                     new_start_rand = new_r_rand if new_r_rand >= 1 else p_rand
                     if new_start_rand <= W:
                         cov[new_start_rand:W+1:p_rand] += 1
                     residues[i_rand] = new_r_rand
                     current_L = get_max_L(cov)

    return [int(r) for r in best_residues]
# EVOLVE-BLOCK-END
