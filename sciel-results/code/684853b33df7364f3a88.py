"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import random

def _get_primes():
    """Return list of primes <= 1000."""
    sieve = [True] * 1001
    sieve[0] = sieve[1] = False
    for i in range(2, int(1000**0.5) + 1):
        if sieve[i]:
            sieve[i*i::i] = [False] * len(sieve[i*i::i])
    return [i for i in range(2, 1001) if sieve[i]]

def construct():
    start_time = time.time()
    deadline = start_time + 145  # 5s safety margin before 150s kill
    
    primes = _get_primes()
    n = len(primes)
    
    # Use MAX_L = 10000 for speed; the target is L >= 1000
    MAX_L = 10000
    
    def build_coverage(residues):
        """Build the coverage array from scratch."""
        cov = np.zeros(MAX_L + 1, dtype=np.int32)
        for i, p in enumerate(primes):
            r = residues[i]
            if r == 0:
                start = p
            else:
                start = r
            if start <= MAX_L:
                cov[start:MAX_L+1:p] += 1
        return cov
    
    def find_first_gap(cov):
        """Find the largest L such that all 1..L have coverage >= 2."""
        bad_indices = np.where(cov[1:MAX_L+1] < 2)[0]
        if len(bad_indices) == 0:
            return MAX_L
        first_bad = bad_indices[0] + 1
        return first_bad - 1
    
    def get_deficient_positions(cov, L):
        """Get positions in [1, L] with coverage < 2."""
        if L >= MAX_L:
            return np.array([], dtype=np.int32)
        bad = np.where(cov[1:L+1] < 2)[0] + 1
        return bad
    
    def update_coverage(cov, idx, old_r, new_r):
        """Update coverage in-place when changing residue at idx."""
        p = primes[idx]
        if old_r == 0:
            old_start = p
        else:
            old_start = old_r
        if old_start <= MAX_L:
            cov[old_start:MAX_L+1:p] -= 1
        if new_r == 0:
            new_start = p
        else:
            new_start = new_r
        if new_start <= MAX_L:
            cov[new_start:MAX_L+1:p] += 1
    
    def count_deficiencies(cov, L):
        """Count positions in [1, L] with coverage < 2."""
        if L >= MAX_L:
            return 0
        return int(np.sum(cov[1:L+1] < 2))
    
    best_L_global = 0
    best_residues_global = [0] * n
    
    # Multiple initializations to explore different basins
    def init_seed():
        return [((i + 2) // 2) % p for i, p in enumerate(primes)]
    
    def init_random():
        return [random.randint(0, p - 1) for p in primes]
    
    def init_perturbed():
        r = init_seed()
        for _ in range(10):
            idx = random.randint(0, n - 1)
            r[idx] = random.randint(0, primes[idx] - 1)
        return r
    
    def init_targeted():
        """Initialize by targeting coverage of early positions."""
        residues = [0] * n
        # For each position x in [1, 200], try to cover it with 2 primes
        for x in range(1, min(201, MAX_L + 1)):
            # Find primes that can cover x
            covering_primes = []
            for i, p in enumerate(primes):
                r_needed = x % p
                if residues[i] == 0 and r_needed != 0:
                    covering_primes.append((i, r_needed))
            # Assign to 2 primes if possible
            if len(covering_primes) >= 2:
                random.shuffle(covering_primes)
                for i, r_needed in covering_primes[:2]:
                    residues[i] = r_needed
        # Fill remaining with random
        for i, p in enumerate(primes):
            if residues[i] == 0:
                residues[i] = random.randint(0, p - 1)
        return residues
    
    initializers = [init_seed, init_targeted, init_perturbed, init_random, init_seed, init_targeted]
    
    for init_func in initializers:
        if time.time() > deadline - 5:
            break
        
        residues = init_func()
        coverage = build_coverage(residues)
        current_L = find_first_gap(coverage)
        
        if current_L > best_L_global:
            best_L_global = current_L
            best_residues_global = residues[:]
        
        # Deficit-driven optimization: focus on fixing deficient positions
        improved = True
        while improved and time.time() < deadline - 5:
            improved = False
            
            # Get deficient positions in current range
            deficient = get_deficient_positions(coverage, current_L)
            
            if len(deficient) == 0:
                # No deficiencies in [1, current_L], try to extend
                break
            
            # For each deficient position, find primes that could cover it
            # and prioritize changes that fix multiple deficiencies
            prime_indices = list(range(n))
            random.shuffle(prime_indices)
            
            for idx in prime_indices:
                if time.time() > deadline - 5:
                    break
                
                p = primes[idx]
                old_r = residues[idx]
                
                # Generate candidates that cover deficient positions
                candidates = set()
                candidates.add(old_r)
                
                # Add residues that would cover deficient positions
                for pos in deficient[:50]:  # Limit to 50 for speed
                    r_needed = pos % p
                    candidates.add(r_needed)
                
                # Also try to cover current_L + 1 to extend
                target = current_L + 1
                if target <= MAX_L:
                    candidates.add(target % p)
                
                # Add neighbors
                candidates.add((old_r + 1) % p)
                candidates.add((old_r - 1) % p)
                
                # Add random
                for _ in range(min(10, p)):
                    candidates.add(random.randint(0, p - 1))
                
                best_r_for_idx = old_r
                best_L_for_idx = current_L
                best_deficit_for_idx = count_deficiencies(coverage, current_L)
                
                for r in candidates:
                    if r == old_r:
                        continue
                    if time.time() > deadline - 5:
                        break
                    
                    update_coverage(coverage, idx, old_r, r)
                    new_L = find_first_gap(coverage)
                    new_deficit = count_deficiencies(coverage, new_L)
                    update_coverage(coverage, idx, r, old_r)
                    
                    # Prefer higher L, then lower deficit
                    if new_L > best_L_for_idx or (new_L == best_L_for_idx and new_deficit < best_deficit_for_idx):
                        best_L_for_idx = new_L
                        best_deficit_for_idx = new_deficit
                        best_r_for_idx = r
                
                if best_r_for_idx != old_r:
                    update_coverage(coverage, idx, old_r, best_r_for_idx)
                    residues[idx] = best_r_for_idx
                    current_L = best_L_for_idx
                    improved = True
                    if current_L > best_L_global:
                        best_L_global = current_L
                        best_residues_global = residues[:]
        
        # End of optimization for this initialization
    
    return [int(r) for r in best_residues_global]
# EVOLVE-BLOCK-END
