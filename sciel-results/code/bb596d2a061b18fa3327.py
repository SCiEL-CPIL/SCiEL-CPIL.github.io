import time
import numpy as np

def _sieve_primes(limit):
    if limit < 2:
        return []
    is_prime = np.ones(limit + 1, dtype=bool)
    is_prime[0] = is_prime[1] = False
    for i in range(2, int(limit ** 0.5) + 1):
        if is_prime[i]:
            is_prime[i*i::i] = False
    return [i for i in range(2, limit + 1) if is_prime[i]]

def _first_start(r, p):
    """First x >= 1 such that x ≡ r (mod p)"""
    if r == 0:
        return p
    return r

def construct():
    primes = _sieve_primes(1000)
    n = len(primes)
    
    # We will search for L up to a certain max.
    # The problem states L >= 1000 is the target.
    # Let's aim for higher.
    
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin
    
    # Initial M (max check range)
    M = 5000
    
    # Initialize residues
    # Strategy: Start with r_p = 1 for all p? Or random?
    # Let's try a few starts and pick the best.
    
    def _compute_counts(residues, max_check):
        counts = np.zeros(max_check + 1, dtype=np.int32)
        for r, p in zip(residues, primes):
            start = _first_start(r, p)
            if start <= max_check:
                counts[start::p] += 1
        return counts

    def _find_L(counts):
        # Find first x in 1..len(counts)-1 where counts[x] < 2
        # If all are >= 2, return len(counts)-1
        # Note: counts[0] is ignored
        below = np.where(counts[1:] < 2)[0]
        if len(below) == 0:
            return len(counts) - 1
        return below[0]

    def _update_counts(counts, old_r, new_r, p, max_check):
        old_start = _first_start(old_r, p)
        if old_start <= max_check:
            counts[old_start::p] -= 1
        new_start = _first_start(new_r, p)
        if new_start <= max_check:
            counts[new_start::p] += 1

    def _optimize(residues, max_check, time_limit):
        counts = _compute_counts(residues, max_check)
        best_L = _find_L(counts)
        
        # Local search
        improved = True
        no_improve_count = 0
        max_no_improve = 2000 # Allow some plateau search
        
        while improved and (time.time() < time_limit) and no_improve_count < max_no_improve:
            improved = False
            no_improve_count += 1
            
            # Find the current bottleneck
            # If best_L < max_check, the bottleneck is best_L + 1
            # If best_L == max_check, we might need to extend max_check, 
            # but for now, let's assume we are not at the boundary or we handle it outside.
            
            if best_L < max_check:
                target_x = best_L + 1
            else:
                # If we've covered up to max_check, we can't easily improve L 
                # without extending the range. 
                # For now, try to optimize within range to ensure robustness?
                # Actually, if L == max_check, the "gap" is at max_check+1 which we don't track.
                # We should extend M in the outer loop.
                # So if we hit the boundary, break and let outer loop extend M.
                break

            # Try to fix target_x
            # For each prime p, consider setting r_p = target_x % p
            # We want to pick the change that results in the largest new L.
            
            best_candidate_score = best_L
            best_candidate_p = -1
            best_candidate_new_r = -1
            best_candidate_old_r = -1
            
            # To make this faster, we can sample primes or prioritize.
            # Changing small primes has large side effects.
            # Changing large primes has small side effects.
            # Let's try all primes, but it's O(N * M/p) which is sum(M/p) ~ M log log N.
            # M=5000, N=168. Sum(1/p) ~ 2.5. So ~12500 ops per full sweep of candidates.
            # This is very fast.
            
            for i, p in enumerate(primes):
                old_r = residues[i]
                new_r = target_x % p
                
                if new_r == old_r:
                    continue
                
                # Apply
                _update_counts(counts, old_r, new_r, p, max_check)
                
                # Evaluate
                new_L = _find_L(counts)
                
                if new_L > best_candidate_score:
                    best_candidate_score = new_L
                    best_candidate_p = i
                    best_candidate_new_r = new_r
                    best_candidate_old_r = old_r
                    # We can't break here easily because we need to revert 
                    # or we can keep the best state. 
                    # But _find_L is O(M). 
                    # It's better to just track the best and revert at the end of the inner loop.
                
                # Revert
                _update_counts(counts, new_r, old_r, p, max_check)
            
            # Apply the best candidate found
            if best_candidate_p != -1:
                _update_counts(counts, best_candidate_old_r, best_candidate_new_r, primes[best_candidate_p], max_check)
                residues[best_candidate_p] = best_candidate_new_r
                best_L = best_candidate_score
                improved = True
                no_improve_count = 0
            else:
                # No improvement by fixing the immediate next gap with a single prime change.
                # This might be a local minimum.
                # Try random moves that keep L same or slightly worse? 
                # Or try fixing target_x by changing TWO primes?
                # For simplicity, let's try a random perturbation if stuck for a while.
                if no_improve_count > 100:
                    # Random restart of a few residues
                    for _ in range(5):
                        i = np.random.randint(0, n)
                        old_r = residues[i]
                        new_r = np.random.randint(0, primes[i])
                        _update_counts(counts, old_r, new_r, primes[i], max_check)
                        new_L = _find_L(counts)
                        if new_L >= best_L:
                            residues[i] = new_r
                            if new_L > best_L:
                                best_L = new_L
                                improved = True
                                no_improve_count = 0
                        else:
                            _update_counts(counts, new_r, old_r, primes[i], max_check)
        
        return residues, best_L

    # Main loop: Initialize and Optimize with increasing M
    
    # Start with a good initial guess
    # r_p = 1 is a strong candidate for small primes
    init_residues = [1] * n
    
    # Let's also try random init
    random_residues = [np.random.randint(0, p) for p in primes]
    
    # Evaluate initial scores with a small M to pick the best start
    M_init = 1000
    c1 = _compute_counts(init_residues, M_init)
    L1 = _find_L(c1)
    c2 = _compute_counts(random_residues, M_init)
    L2 = _find_L(c2)
    
    if L1 >= L2:
        current_residues = init_residues.copy()
        current_M = 1000
    else:
        current_residues = random_residues.copy()
        current_M = 1000

    # Iteratively increase M and optimize
    # Start M at 1000, then 2000, 5000, 10000...
    # But we should optimize at the current M first.
    
    while (time.time() < deadline):
        # If current_M is too small, increase it
        # We don't know L yet, so we can't be sure.
        # Let's just run optimize with current_M.
        # If L reaches current_M, we increase current_M.
        
        current_residues, current_L = _optimize(current_residues, current_M, deadline)
        
        if current_L >= current_M:
            # Extend the range
            # Double the range or add 5000
            current_M = min(current_M + 5000, 20000) # Cap at 20000 to avoid memory/time issues
            # Note: _optimize will re-compute counts for the new M.
            # We need to make sure _optimize handles the resize.
            # Currently _optimize takes max_check. 
            # If we call _optimize again with larger M, it recomputes counts from scratch.
            # This is O(N*M). 168 * 20000 = 3.3M ops. Fast.
        else:
            # If we are stuck at L < M, we might need to try harder or accept.
            # If we've been stuck for a while, break.
            # _optimize has a no_improve counter. If it returns without improving, 
            # and L < M, we are in a local minimum.
            # We could try random restarts of the whole vector, but that's expensive.
            # Let's just break if we can't improve.
            break

    return current_residues
