import numpy as np
from math import log, sqrt

def construct():
    # Generate primes up to 1000
    def sieve(n):
        is_prime = [True] * (n + 1)
        is_prime[0] = is_prime[1] = False
        for i in range(2, int(n ** 0.5) + 1):
            if is_prime[i]:
                is_prime[i*i::i] = [False] * len(is_prime[i*i::i])
        return [i for i, prime in enumerate(is_prime) if prime]
    
    primes = sieve(1000)
    n_primes = len(primes)
    
    MAX_L = 5000
    x = np.arange(1, MAX_L + 1)
    
    coverage = np.zeros(MAX_L, dtype=np.int32)
    residues = np.zeros(n_primes, dtype=np.int64)
    
    def compute_L(cov):
        bad = np.where(cov < 2)[0]
        if len(bad) == 0:
            return MAX_L
        return bad[0]
    
    def apply_residue(p, r, cov, x_arr):
        mask = (x_arr % p) == r
        cov[mask] += 1
    
    def remove_residue(p, r, cov, x_arr):
        mask = (x_arr % p) == r
        cov[mask] -= 1
    
    # Phase 1: Greedy assignment with deficit awareness
    # For each prime, choose residue to maximize coverage in the "frontier" region
    # Frontier = first few integers where coverage is low
    
    for i in range(n_primes):
        p = primes[i]
        
        # Find the current bottleneck
        bad = np.where(coverage < 2)[0]
        
        if len(bad) == 0:
            # All covered, just pick 0
            best_r = 0
        else:
            # Focus on the first few bad points to build up coverage there
            # Consider a window of the first 20 bad points
            window_size = min(20, len(bad))
            frontier = bad[:window_size]
            
            # Score each possible residue by how many frontier points it covers
            # Also consider points just after the frontier for extension potential
            
            best_r = 0
            best_score = -1
            
            # Try a subset of residues for speed
            # For small p, try all; for large p, try residues that cover frontier points
            if p <= 50:
                candidates = range(p)
            else:
                # Only try residues that cover at least one frontier point
                candidate_set = set()
                for idx in frontier:
                    val = x[idx]
                    r = val % p
                    candidate_set.add(r)
                # Also add 0 as a fallback
                candidate_set.add(0)
                candidates = list(candidate_set)
            
            for r in candidates:
                # Compute how many frontier points this residue covers
                covered = 0
                for idx in frontier:
                    if x[idx] % p == r:
                        covered += 1
                
                # Bonus for covering points just beyond the frontier
                # Look at points from frontier[-1]+1 to frontier[-1]+10
                ext_start = frontier[-1] + 1
                ext_end = min(MAX_L, ext_start + 10)
                ext_bonus = 0
                for idx in range(ext_start, ext_end):
                    if x[idx] % p == r:
                        ext_bonus += 1
                
                # Weighted score: frontier coverage is most important
                score = covered * 10 + ext_bonus
                
                if score > best_score:
                    best_score = score
                    best_r = r
        
        residues[i] = best_r
        apply_residue(p, best_r, coverage, x)
    
    # Phase 2: Targeted local search on bottleneck primes
    L = compute_L(coverage)
    
    improved = True
    max_iterations = 200
    
    while improved and L < MAX_L and max_iterations > 0:
        max_iterations -= 1
        improved = False
        
        x_fail = L + 1
        if x_fail > MAX_L:
            break
        
        # Find primes that could help cover x_fail
        # A prime p helps if changing its residue to x_fail % p is safe
        # Safe means: all points in [1, L] that p currently covers have coverage >= 3
        
        best_i = -1
        best_p = -1
        best_r_new = -1
        best_gain = 0
        
        for i in range(n_primes):
            p = primes[i]
            r_old = residues[i]
            r_new = x_fail % p
            
            if r_new == r_old:
                continue
            
            # Check safety: all points in [1, L] covered by (p, r_old) must have cov >= 3
            # Points covered by (p, r_old) in [1, L]: x = r_old + k*p, 1 <= x <= L
            
            if r_old == 0:
                k_start = 1  # x = p
            else:
                k_start = 0  # x = r_old
            
            k_end = (L - r_old) // p
            
            if k_start <= k_end:
                ks = np.arange(k_start, k_end + 1)
                x_vals = r_old + ks * p
                idxs = x_vals - 1
                
                if len(idxs) > 0 and np.min(coverage[idxs]) < 3:
                    continue  # Not safe
            
            # This change is safe. Estimate gain.
            # Gain = number of points in [L+1, L+20] that this change would cover
            gain = 0
            for dx in range(1, 21):
                xt = L + dx
                if xt <= MAX_L and xt % p == r_new:
                    gain += 1
            
            if gain > best_gain:
                best_gain = gain
                best_i = i
                best_p = p
                best_r_new = r_new
        
        if best_i >= 0:
            i = best_i
            p = best_p
            r_old = residues[i]
            r_new = best_r_new
            
            # Apply change
            remove_residue(p, r_old, coverage, x)
            apply_residue(p, r_new, coverage, x)
            residues[i] = r_new
            
            new_L = compute_L(coverage)
            
            if new_L > L:
                L = new_L
                improved = True
            else:
                # Revert
                remove_residue(p, r_new, coverage, x)
                apply_residue(p, r_old, coverage, x)
                residues[i] = r_old
    
    # Phase 3: Final pass - try to squeeze out more L by trying all primes for the bottleneck
    L = compute_L(coverage)
    
    for _ in range(50):
        if L >= MAX_L:
            break
        
        x_fail = L + 1
        if x_fail > MAX_L:
            break
        
        found = False
        for i in range(n_primes):
            p = primes[i]
            r_old = residues[i]
            r_new = x_fail % p
            
            if r_new == r_old:
                continue
            
            # Quick safety check
            if r_old == 0:
                k_start = 1
            else:
                k_start = 0
            
            k_end = (L - r_old) // p
            
            if k_start <= k_end:
                ks = np.arange(k_start, k_end + 1)
                x_vals = r_old + ks * p
                idxs = x_vals - 1
                
                if len(idxs) > 0 and np.min(coverage[idxs]) < 3:
                    continue
            
            # Apply
            remove_residue(p, r_old, coverage, x)
            apply_residue(p, r_new, coverage, x)
            residues[i] = r_new
            
            new_L = compute_L(coverage)
            
            if new_L > L:
                L = new_L
                found = True
                break
            else:
                # Revert
                remove_residue(p, r_new, coverage, x)
                apply_residue(p, r_old, coverage, x)
                residues[i] = r_old
        
        if not found:
            break
    
    return [int(r) for r in residues]
