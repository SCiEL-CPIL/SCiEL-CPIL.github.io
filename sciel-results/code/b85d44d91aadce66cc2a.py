"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import random
import math

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
    # 150s limit. Use 140s for safety.
    deadline = start_time + 140
    
    primes = _get_primes()
    n = len(primes)
    MAX_L = 100000
    
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
                cov[start::p] += 1
        return cov
    
    def find_first_gap(cov):
        """Find the first x >= 1 where cov[x] < 2."""
        # We only care up to MAX_L
        bad = np.where(cov[1:MAX_L+1] < 2)[0]
        if len(bad) == 0:
            return MAX_L
        return bad[0] + 1 # 0-indexed in bad, so +1 to get x
    
    def update_coverage(cov, idx, old_r, new_r):
        """
        Update coverage array in-place when changing residue at idx from old_r to new_r.
        """
        p = primes[idx]
        
        # Remove old
        if old_r == 0:
            old_start = p
        else:
            old_start = old_r
        if old_start <= MAX_L:
            cov[old_start::p] -= 1
            
        # Add new
        if new_r == 0:
            new_start = p
        else:
            new_start = new_r
        if new_start <= MAX_L:
            cov[new_start::p] += 1

    # 1. Greedy Initialization
    def greedy_init():
        residues = [0] * n
        assigned = [False] * n
        coverage = np.zeros(MAX_L + 1, dtype=np.int32)
        
        # We will try to cover x=1, 2, 3...
        # For each x, if coverage[x] < 2, we assign residues to unassigned primes.
        # We prefer small primes because they have higher frequency.
        
        for x in range(1, 200): # Greedy up to 200
            if time.time() > deadline - 5:
                break
            # Check current coverage of x
            if coverage[x] >= 2:
                continue
            
            # Need (2 - coverage[x]) more primes to cover x
            needed = 2 - coverage[x]
            
            # Find unassigned primes
            # Sort by prime value (smallest first) to maximize future coverage density?
            # Or just iterate through primes and pick unassigned ones.
            candidates = [i for i in range(n) if not assigned[i]]
            
            for idx in candidates:
                if needed <= 0:
                    break
                p = primes[idx]
                # Calculate required residue to cover x
                r_needed = x % p
                # Assign it
                residues[idx] = r_needed
                assigned[idx] = True
                # Update coverage
                update_coverage(coverage, idx, 0, r_needed)
                # Check if this actually covers x
                # It does by definition of r_needed
                coverage[x] += 1
                needed -= 1
            
            # If we still don't have enough coverage, we can't fix it with new primes
            # We might need to reassign existing primes, but greedy doesn't do that.
            
        # Assign remaining unassigned primes randomly or 0
        for i in range(n):
            if not assigned[i]:
                residues[i] = random.randint(0, primes[i]-1)
                # Update coverage for these too? 
                # Actually, build_coverage is cleaner for the final state.
        
        return residues

    # 2. Local Search
    def local_search(residues):
        coverage = build_coverage(residues)
        current_L = find_first_gap(coverage)
        best_L = current_L
        best_res = residues[:]
        
        # Simulated Annealing / ILS
        temperature = 10.0
        cooling = 0.99
        
        while time.time() < deadline - 2:
            if temperature < 0.1:
                temperature = 1.0
                # Perturb best solution to restart
                residues = best_res[:]
                for _ in range(5):
                    idx = random.randint(0, n-1)
                    residues[idx] = random.randint(0, primes[idx]-1)
                coverage = build_coverage(residues)
                current_L = find_first_gap(coverage)
                if current_L > best_L:
                    best_L = current_L
                    best_res = residues[:]
                    temperature = 10.0
                continue

            # Pick a prime to change
            # Bias towards primes that are "active" near the gap?
            # Or just random.
            idx = random.randint(0, n-1)
            old_r = residues[idx]
            
            # Generate new residue
            # Strategy: Try to cover the current gap (current_L + 1)
            target_x = current_L + 1
            if random.random() < 0.5:
                # Targeted move
                new_r = target_x % primes[idx]
            else:
                # Random move
                new_r = random.randint(0, primes[idx]-1)
                
            if new_r == old_r:
                continue
                
            # Apply
            update_coverage(coverage, idx, old_r, new_r)
            residues[idx] = new_r
            
            new_L = find_first_gap(coverage)
            
            # Acceptance
            if new_L > current_L:
                current_L = new_L
                if new_L > best_L:
                    best_L = new_L
                    best_res = residues[:]
            else:
                # Metropolis criterion
                delta = current_L - new_L # > 0
                prob = math.exp(-delta / temperature)
                if random.random() < prob:
                    current_L = new_L
                else:
                    # Revert
                    update_coverage(coverage, idx, new_r, old_r)
                    residues[idx] = old_r
            
            temperature *= cooling
        
        return best_res

    # Run
    init_res = greedy_init()
    final_res = local_search(init_res)
    
    # Final check
    return [int(r) for r in final_res]

# EVOLVE-BLOCK-END
