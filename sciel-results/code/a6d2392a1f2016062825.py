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
    if r == 0:
        return p
    return r

def construct():
    primes = _sieve_primes(1000)
    n = len(primes)
    
    start_time = time.time()
    deadline = start_time + 130  # 20s safety margin
    
    def _compute_counts(residues, max_check):
        counts = np.zeros(max_check + 1, dtype=np.int32)
        for r, p in zip(residues, primes):
            start = _first_start(r, p)
            if start <= max_check:
                counts[start::p] += 1
        return counts

    def _find_L(counts):
        below = np.where(counts[1:] < 2)[0]
        if len(below) == 0:
            return len(counts) - 1
        return int(below[0])

    def _update_counts(counts, old_r, new_r, p, max_check):
        old_start = _first_start(old_r, p)
        if old_start <= max_check:
            counts[old_start::p] -= 1
        new_start = _first_start(new_r, p)
        if new_start <= max_check:
            counts[new_start::p] += 1

    def _greedy_critical_optimize(residues, max_check, time_limit):
        """
        Greedy set-cover heuristic: repeatedly find the most "critical" position
        (first with count < 2) and try to fix it by changing one residue.
        Also tries to improve coverage of positions with count == 1.
        """
        residues = list(residues)
        counts = _compute_counts(residues, max_check)
        best_L = _find_L(counts)
        
        no_improve = 0
        max_no_improve = 300
        
        while time.time() < time_limit and no_improve < max_no_improve:
            if best_L >= max_check:
                break
            
            # Find the target: first position with count < 2
            target_x = best_L + 1
            
            # Try each prime to see if changing its residue helps
            best_delta = 0
            best_i = -1
            best_new_r = -1
            best_old_r = -1
            
            # Also consider improving positions with count == 1 near the frontier
            # to build a buffer
            frontier_start = max(1, target_x - 20)
            frontier_end = min(max_check, target_x + 20)
            
            for i, p in enumerate(primes):
                old_r = residues[i]
                
                # Primary candidate: cover target_x
                new_r_target = target_x % p
                candidates = set()
                if new_r_target != old_r:
                    candidates.add(new_r_target)
                
                # Secondary candidates: cover positions with count==1 in frontier
                for pos in range(frontier_start, frontier_end + 1):
                    if counts[pos] == 1:
                        cr = pos % p
                        if cr != old_r:
                            candidates.add(cr)
                
                # Tertiary: random
                if len(candidates) < 3:
                    for _ in range(2):
                        candidates.add(int(np.random.randint(0, p)))
                
                for new_r in candidates:
                    if new_r == old_r:
                        continue
                    
                    _update_counts(counts, old_r, new_r, p, max_check)
                    new_L = _find_L(counts)
                    _update_counts(counts, new_r, old_r, p, max_check)
                    
                    delta = new_L - best_L
                    if delta > best_delta:
                        best_delta = delta
                        best_i = i
                        best_new_r = new_r
                        best_old_r = old_r
            
            if best_i != -1:
                _update_counts(counts, best_old_r, best_new_r, primes[best_i], max_check)
                residues[best_i] = best_new_r
                best_L += best_delta
                no_improve = 0
            else:
                no_improve += 1
        
        return residues, best_L

    def _greedy_optimize(residues, max_check, time_limit):
        """Standard greedy hill-climbing."""
        residues = list(residues)
        counts = _compute_counts(residues, max_check)
        best_L = _find_L(counts)
        
        no_improve = 0
        max_no_improve = 500
        
        while time.time() < time_limit and no_improve < max_no_improve:
            if best_L >= max_check:
                break
            
            target_x = best_L + 1
            best_delta = 0
            best_i = -1
            best_new_r = -1
            best_old_r = -1
            
            for i, p in enumerate(primes):
                old_r = residues[i]
                new_r = target_x % p
                
                if new_r == old_r:
                    continue
                
                _update_counts(counts, old_r, new_r, p, max_check)
                new_L = _find_L(counts)
                _update_counts(counts, new_r, old_r, p, max_check)
                
                delta = new_L - best_L
                if delta > best_delta:
                    best_delta = delta
                    best_i = i
                    best_new_r = new_r
                    best_old_r = old_r
            
            if best_i != -1:
                _update_counts(counts, best_old_r, best_new_r, primes[best_i], max_check)
                residues[best_i] = best_new_r
                best_L += best_delta
                no_improve = 0
            else:
                no_improve += 1
        
        return residues, best_L

    def _random_restarts(residues, max_check, time_limit, num_restarts=8):
        best_residues = list(residues)
        best_L = _find_L(_compute_counts(residues, max_check))
        
        for _ in range(num_restarts):
            if time.time() >= time_limit:
                break
            
            random_residues = [int(np.random.randint(0, p)) for p in primes]
            
            # Alternate between the two optimization strategies
            if np.random.random() < 0.5:
                res, L = _greedy_critical_optimize(random_residues, max_check, time_limit)
            else:
                res, L = _greedy_optimize(random_residues, max_check, time_limit)
            
            if L > best_L:
                best_residues = res
                best_L = L
        
        return best_residues, best_L

    # Start with r_p = 1 for all primes
    init_residues = [1] * n
    current_M = 2000
    
    # Initial optimization with critical heuristic
    current_residues, current_L = _greedy_critical_optimize(init_residues, current_M, deadline)
    
    # Try to extend if we hit the boundary
    while time.time() < deadline:
        if current_L < current_M:
            # Stuck - try random restarts
            current_residues, current_L = _random_restarts(
                current_residues, current_M, deadline, num_restarts=5
            )
            if current_L < current_M:
                # Try one more round with standard greedy
                current_residues, current_L = _greedy_optimize(
                    current_residues, current_M, deadline
                )
                if current_L < current_M:
                    break
        
        # Extend the range
        current_M = min(current_M * 2, 10000)
        current_residues, current_L = _greedy_critical_optimize(current_residues, current_M, deadline)
    
    return [int(r) for r in current_residues]
