"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import math
import numpy as np

# EVOLVE-BLOCK-START

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # safety margin before 120s kill

    # Generate primes <= 1000
    def get_primes(limit):
        sieve = [True] * (limit + 1)
        sieve[0] = sieve[1] = False
        for i in range(2, int(math.isqrt(limit)) + 1):
            if sieve[i]:
                for j in range(i * i, limit + 1, i):
                    sieve[j] = False
        return [i for i, is_p in enumerate(sieve) if is_p]

    primes = get_primes(1000)
    n = len(primes)  # 168
    L_max = 5000  # Search space upper bound

    def evaluate(residues):
        """Return the largest L such that every integer in 1..L is covered by at least 2 congruences."""
        counts = np.zeros(L_max + 1, dtype=np.int32)
        
        for i in range(n):
            p = primes[i]
            r = residues[i]
            if r == 0:
                # Indices 1..L_max where x % p == 0 are p, 2p, ...
                # 0-indexed in numpy: p-1, 2p-1, ...
                if p <= L_max:
                    counts[p::p] += 1
            else:
                # Indices 1..L_max where x % p == r are r, r+p, ...
                # 0-indexed in numpy: r-1, r-1+p, ...
                if r <= L_max:
                    counts[r::p] += 1
        
        # Find the first index where counts < 2
        # We care about 1..L_max
        # counts[0] is unused (or represents 0, but problem starts at 1)
        mask = counts[1:] < 2
        if np.any(mask):
            first_bad_idx = np.argmax(mask)
            L = first_bad_idx # This is 0-indexed offset from 1, so L = index
            # If mask[0] is True, L=0. If mask[1] is True, L=1.
            # Wait, np.argmax returns index in mask.
            # If mask[0] is True, argmax is 0. L=0.
            # If mask[1] is True, argmax is 1. L=1.
            # The problem asks for largest L such that 1..L are good.
            # If 1 is bad, L=0.
            # If 1,2 good, 3 bad, L=2.
            # mask[0] corresponds to integer 1.
            # So L = first_bad_idx.
            return L
        else:
            return L_max

    # Initialization
    residues = [((i + 2) // 2) % primes[i] for i in range(n)]
    current_L = evaluate(residues)
    best_residues = residues[:]
    best_L = current_L

    if time.time() > deadline - 10:
        return best_residues

    # Precompute index slices for fast updates?
    # For each prime p and residue r, we need the slice or array of indices.
    # Since p varies, we can just use slicing in the update step.
    
    def get_indices(p, r, L_limit):
        """Return numpy array of indices in 1..L_limit where x % p == r."""
        if r == 0:
            first = p
        else:
            first = r
        if first > L_limit:
            return np.array([], dtype=np.int32)
        return np.arange(first, L_limit + 1, p, dtype=np.int32)

    # Greedy Directed Search
    # We maintain a coverage array to update incrementally.
    coverage = np.zeros(L_max + 1, dtype=np.int32)
    for i in range(n):
        p = primes[i]
        r = residues[i]
        idx = get_indices(p, r, L_max)
        if len(idx) > 0:
            coverage[idx] += 1

    def get_L_from_coverage(cov):
        mask = cov[1:] < 2
        if np.any(mask):
            return np.argmax(mask)
        return L_max

    current_L = get_L_from_coverage(coverage)
    
    # Random seed for tie-breaking or perturbations
    random.seed(42)
    
    while time.time() < deadline - 5:
        # Find current bottleneck B
        if current_L >= L_max:
            break
            
        # B is the first bad index (1-indexed)
        # coverage[1:] < 2
        mask = coverage[1:] < 2
        if not np.any(mask):
            break
        B_idx_0 = np.argmax(mask) # 0-indexed in mask, so B = B_idx_0 + 1
        B = B_idx_0 + 1
        
        improved = False
        
        # Try to fix B by changing residues of primes that don't currently cover B
        # Or even those that do? 
        # If a prime covers B, changing it might remove coverage from B.
        # We want to ADD coverage to B.
        # So we look for primes i where r_i != B % p_i.
        
        candidates = []
        for i in range(n):
            p = primes[i]
            r_target = B % p
            if residues[i] != r_target:
                candidates.append(i)
                
        # Shuffle candidates to avoid bias
        random.shuffle(candidates)
        
        for i in candidates:
            if time.time() > deadline - 5:
                break
                
            p = primes[i]
            old_r = residues[i]
            new_r = B % p
            
            # Get indices for old and new
            old_idx = get_indices(p, old_r, L_max)
            new_idx = get_indices(p, new_r, L_max)
            
            # Update coverage
            if len(old_idx) > 0:
                coverage[old_idx] -= 1
            if len(new_idx) > 0:
                coverage[new_idx] += 1
                
            # Check new L
            new_L = get_L_from_coverage(coverage)
            
            if new_L > current_L:
                # Accept change
                residues[i] = new_r
                current_L = new_L
                if current_L > best_L:
                    best_L = current_L
                    best_residues = residues[:]
                improved = True
                # If we improved, the bottleneck B might be fixed.
                # We can continue trying to fix the same B with other primes?
                # Or re-evaluate B?
                # If B is fixed, the new bottleneck is further out.
                # It's safe to continue with other candidates for the OLD B?
                # If we change another prime to also cover B, B gets more coverage.
                # That's fine.
                # But if we change a prime that covered B to NOT cover B, that would be bad.
                # But we only change primes that do NOT cover B (r_i != B % p).
                # So we are only ADDING coverage to B.
                # So B will remain fixed (or become more covered).
                # So we can continue.
            else:
                # Revert
                if len(new_idx) > 0:
                    coverage[new_idx] -= 1
                if len(old_idx) > 0:
                    coverage[old_idx] += 1
                # residues[i] remains old_r

        if not improved:
            # Stuck. Try a random perturbation to escape local optimum.
            # Change 1-3 random residues to random values.
            num_changes = random.randint(1, 3)
            for _ in range(num_changes):
                k = random.randrange(n)
                old_r_k = residues[k]
                p_k = primes[k]
                new_r_k = random.randrange(p_k)
                
                old_idx_k = get_indices(p_k, old_r_k, L_max)
                new_idx_k = get_indices(p_k, new_r_k, L_max)
                
                if len(old_idx_k) > 0:
                    coverage[old_idx_k] -= 1
                if len(new_idx_k) > 0:
                    coverage[new_idx_k] += 1
                    
                residues[k] = new_r_k
                
            new_L = get_L_from_coverage(coverage)
            
            if new_L > current_L:
                current_L = new_L
                if current_L > best_L:
                    best_L = current_L
                    best_residues = residues[:]
            else:
                # Revert perturbation
                # This is tricky because we changed multiple.
                # Easier: Just reset to best_residues and recompute coverage?
                # Or store the changes and revert them.
                # Let's just reset to best_residues for simplicity, as perturbation is rare.
                residues = best_residues[:]
                coverage = np.zeros(L_max + 1, dtype=np.int32)
                for i in range(n):
                    p = primes[i]
                    r = residues[i]
                    idx = get_indices(p, r, L_max)
                    if len(idx) > 0:
                        coverage[idx] += 1
                current_L = get_L_from_coverage(coverage)
                
                # If we are stuck at best_L, break?
                # No, we might have time for more perturbations.
                # But if we keep reverting to best, we are looping.
                # Let's break if we revert to best and no improvement in last few steps.
                # For now, just continue.
                
    return best_residues

# EVOLVE-BLOCK-END
