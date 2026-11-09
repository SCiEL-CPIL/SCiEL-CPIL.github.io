import time
import numpy as np
import random

def construct():
    start_time = time.time()
    deadline = start_time + 145.0
    
    def get_primes(limit):
        sieve = [True] * (limit + 1)
        sieve[0] = sieve[1] = False
        for i in range(2, int(limit ** 0.5) + 1):
            if sieve[i]:
                for j in range(i * i, limit + 1, i):
                    sieve[j] = False
        return [i for i in range(2, limit + 1) if sieve[i]]
    
    primes = get_primes(1000)
    n_primes = len(primes)
    L_max = 5000
    
    # Precompute index arrays for each prime
    prime_indices = []
    for p in primes:
        # indices[r] will be the positions in 1..L_max where x ≡ r (mod p)
        indices = []
        for r in range(p):
            if r == 0:
                first = p
            else:
                first = r
            if first <= L_max:
                indices.append(np.arange(first, L_max + 1, p, dtype=np.int32))
            else:
                indices.append(np.array([], dtype=np.int32))
        prime_indices.append(indices)
    
    def compute_score(coverage):
        """Compute the score from a coverage array."""
        below = np.where(coverage[1:L_max+1] < 2)[0]
        if len(below) == 0:
            return L_max
        return below[0] + 1  # +1 because coverage[1:] starts at index 1
    
    def build_coverage(residues):
        """Build coverage array from residues."""
        coverage = np.zeros(L_max + 1, dtype=np.int32)
        for i, p in enumerate(primes):
            r = residues[i]
            idx = prime_indices[i][r]
            if len(idx) > 0:
                coverage[idx] += 1
        return coverage
    
    # Initialize with random residues
    best_residues = np.array([random.randint(0, primes[i]-1) for i in range(n_primes)], dtype=np.int64)
    best_coverage = build_coverage(best_residues)
    best_score = compute_score(best_coverage)
    
    # Simulated annealing
    current_residues = best_residues.copy()
    current_coverage = best_coverage.copy()
    current_score = best_score
    
    T_start = 1000.0
    T_end = 0.1
    total_steps = 500000
    
    for step in range(total_steps):
        if time.time() > deadline:
            break
        
        # Temperature schedule
        progress = step / total_steps
        T = T_start * (T_end / T_start) ** progress
        
        # Randomly select a prime index to modify
        i = random.randint(0, n_primes - 1)
        p = primes[i]
        old_r = current_residues[i]
        
        # Try a new random residue
        new_r = random.randint(0, p - 1)
        if new_r == old_r:
            continue
        
        # Update coverage incrementally
        old_idx = prime_indices[i][old_r]
        new_idx = prime_indices[i][new_r]
        
        current_coverage[old_idx] -= 1
        current_coverage[new_idx] += 1
        
        new_score = compute_score(current_coverage)
        delta = new_score - current_score
        
        # Accept if better or with probability exp(-delta/T) for worse
        if delta >= 0 or random.random() < np.exp(-delta / max(T, 1e-10)):
            # Accept the move
            current_residues[i] = new_r
            current_score = new_score
            
            if new_score > best_score:
                best_score = new_score
                best_residues = current_residues.copy()
                best_coverage = current_coverage.copy()
        else:
            # Revert
            current_coverage[new_idx] -= 1
            current_coverage[old_idx] += 1
    
    # Final local improvement pass
    improved = True
    while improved and time.time() < deadline - 2.0:
        improved = False
        for i in range(n_primes):
            if time.time() > deadline - 2.0:
                break
            p = primes[i]
            old_r = best_residues[i]
            best_local_score = best_score
            best_local_r = old_r
            
            for r in range(p):
                if r == old_r:
                    continue
                old_idx = prime_indices[i][old_r]
                new_idx = prime_indices[i][r]
                
                best_coverage[old_idx] -= 1
                best_coverage[new_idx] += 1
                local_score = compute_score(best_coverage)
                
                if local_score > best_local_score:
                    best_local_score = local_score
                    best_local_r = r
                else:
                    best_coverage[new_idx] -= 1
                    best_coverage[old_idx] += 1
            
            if best_local_r != old_r:
                # Apply the best
                old_idx = prime_indices[i][old_r]
                new_idx = prime_indices[i][best_local_r]
                best_coverage[old_idx] -= 1
                best_coverage[new_idx] += 1
                best_residues[i] = best_local_r
                best_score = best_local_score
                improved = True
            # coverage is already correct after revert or apply
    
    return [int(r) for r in best_residues]
