"""Combined approach: greedy gap-closure + numpy coverage + structured local search."""
import time
import numpy as np

def construct():
    deadline = time.time() + 130  # safety margin before 150s limit
    
    # Generate primes up to 1000
    def sieve(n):
        is_prime = [True] * (n + 1)
        is_prime[0] = is_prime[1] = False
        for i in range(2, int(n**0.5) + 1):
            if is_prime[i]:
                for j in range(i*i, n + 1, i):
                    is_prime[j] = False
        return [i for i in range(2, n + 1) if is_prime[i]]
    
    primes = sieve(1000)
    n_primes = len(primes)  # 168
    primes_arr = np.array(primes)
    
    # Use a larger range to allow L to grow
    MAX_L = 10000
    
    # NumPy coverage array
    coverage = np.zeros(MAX_L + 1, dtype=np.int32)
    residues = np.zeros(n_primes, dtype=np.int32)
    
    def find_first_gap(cov):
        """Find the first index >= 1 where coverage < 2. Returns MAX_L+1 if none."""
        gaps = np.where(cov[1:] < 2)[0]
        if len(gaps) == 0:
            return MAX_L + 1
        return gaps[0] + 1
    
    def add_congruence(idx, r, cov):
        p = primes[idx]
        start = r if r > 0 else p
        if start <= MAX_L:
            cov[start::p] += 1
    
    def remove_congruence(idx, r, cov):
        p = primes[idx]
        start = r if r > 0 else p
        if start <= MAX_L:
            cov[start::p] -= 1
    
    # Phase 1: Greedy Construction
    # Process primes in increasing order.
    # For each prime, choose the residue that covers the current gap.
    
    # Sort by prime size (small primes first for more coverage)
    order = np.argsort(primes_arr)
    
    for idx in order:
        if time.time() > deadline - 60:
            break
            
        p = primes[idx]
        gap = find_first_gap(coverage)
        
        if gap > MAX_L:
            # Already covered beyond our range, pick 0
            residues[idx] = 0
            add_congruence(idx, 0, coverage)
            continue
        
        # The only residue that can extend L is one that covers the gap
        r = gap % p
        residues[idx] = r
        add_congruence(idx, r, coverage)
    
    # Phase 2: Structured Local Search
    # Try to improve L by changing residues to cover gaps.
    # Key insight: to cover a gap at position g, we need at least 2 primes 
    # whose residues satisfy g % p == r_p.
    
    current_L = find_first_gap(coverage) - 1
    if current_L > MAX_L:
        current_L = MAX_L
    
    improved = True
    max_rounds = 50
    
    while improved and time.time() < deadline - 20:
        improved = False
        
        gap = find_first_gap(coverage)
        if gap > MAX_L:
            break
        
        # Find how many primes currently cover the gap
        gap_coverage = coverage[gap]
        
        # If gap_coverage >= 2, the gap shouldn't exist. Recheck.
        if gap_coverage >= 2:
            # This shouldn't happen if find_first_gap is correct
            break
        
        # We need to increase coverage at gap by (2 - gap_coverage)
        # Try changing residues of primes that don't currently cover the gap
        
        # Prioritize small primes (they cover more positions)
        for idx in order:
            if time.time() > deadline - 20:
                break
            
            p = primes[idx]
            old_r = residues[idx]
            target_r = gap % p
            
            if old_r == target_r:
                # Already covers the gap
                continue
            
            # Try changing to target_r
            # First check: will this break any position < gap?
            # Remove old_r, add target_r, check if any position < gap drops below 2
            
            remove_congruence(idx, old_r, coverage)
            add_congruence(idx, target_r, coverage)
            
            # Check if any position in [1, gap-1] now has coverage < 2
            if gap > 1:
                prefix_min = np.min(coverage[1:gap])
            else:
                prefix_min = 2  # no prefix to check
            
            if prefix_min >= 2:
                # Safe change, L might improve
                new_L = find_first_gap(coverage) - 1
                if new_L > current_L:
                    current_L = new_L
                    residues[idx] = target_r
                    improved = True
                else:
                    # Revert
                    remove_congruence(idx, target_r, coverage)
                    add_congruence(idx, old_r, coverage)
            else:
                # Revert
                remove_congruence(idx, target_r, coverage)
                add_congruence(idx, old_r, coverage)
    
    # Phase 3: Simulated Annealing for local minima escape
    if time.time() < deadline - 5:
        import random
        random.seed(42)
        
        best_L = current_L
        best_residues = residues.copy()
        
        T = 100.0
        cooling = 0.97
        no_improve = 0
        max_no_improve = 500
        
        while time.time() < deadline - 2 and no_improve < max_no_improve:
            idx = random.randint(0, n_primes - 1)
            p = primes[idx]
            old_r = residues[idx]
            new_r = random.randint(0, p - 1)
            
            if new_r == old_r:
                continue
            
            remove_congruence(idx, old_r, coverage)
            add_congruence(idx, new_r, coverage)
            residues[idx] = new_r
            
            new_L = find_first_gap(coverage) - 1
            if new_L < 0:
                new_L = 0
            
            if new_L >= best_L:
                best_L = new_L
                best_residues = residues.copy()
                T = max(T * cooling, 1.0)
                no_improve = 0
            else:
                delta = best_L - new_L
                if delta > 0 and random.random() < np.exp(-delta / max(T, 0.1)):
                    T = max(T * cooling, 0.5)
                    no_improve += 1
                else:
                    remove_congruence(idx, new_r, coverage)
                    add_congruence(idx, old_r, coverage)
                    residues[idx] = old_r
                    T = max(T * cooling, 0.5)
                    no_improve += 1
    
    # Restore best solution
    if best_L > current_L:
        coverage = np.zeros(MAX_L + 1, dtype=np.int32)
        for idx in range(n_primes):
            p = primes[idx]
            r = best_residues[idx]
            add_congruence(idx, r, coverage)
        residues = best_residues
    
    # Final: return as list of ints
    return [int(r) for r in residues]
