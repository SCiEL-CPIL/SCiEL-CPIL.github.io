import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 130  # 130s limit, 20s margin
    
    # Generate primes <= 1000
    def get_primes(limit):
        sieve = [True] * (limit + 1)
        sieve[0] = sieve[1] = False
        for i in range(2, int(limit**0.5) + 1):
            if sieve[i]:
                sieve[i*i::i] = [False] * len(sieve[i*i::i])
        return [i for i, is_p in enumerate(sieve) if is_p]
    
    primes = get_primes(1000)
    n_primes = len(primes)
    assert n_primes == 168
    
    # Max L to consider
    L_max = 2000  # We'll search up to this
    
    def compute_L(residues):
        """Compute the largest L such that every x in [1,L] has count >= 2."""
        counts = np.zeros(L_max + 1, dtype=np.int32)
        for i, p in enumerate(primes):
            r = residues[i]
            if r == 0:
                # Covers p, 2p, 3p, ...
                for x in range(p, L_max + 1, p):
                    counts[x] += 1
            else:
                # Covers r, r+p, r+2p, ...
                for x in range(r, L_max + 1, p):
                    counts[x] += 1
        # Find largest L where min(counts[1:L+1]) >= 2
        L = 0
        for x in range(1, L_max + 1):
            if counts[x] >= 2:
                L = x
            else:
                break
        return L
    
    def compute_L_fast(residues):
        """Faster version using vectorized operations where possible."""
        counts = np.zeros(L_max + 1, dtype=np.int32)
        for i, p in enumerate(primes):
            r = residues[i]
            if r == 0:
                idx = np.arange(p, L_max + 1, p)
            else:
                idx = np.arange(r, L_max + 1, p)
            if len(idx) > 0:
                counts[idx] += 1
        L = 0
        for x in range(1, L_max + 1):
            if counts[x] >= 2:
                L = x
            else:
                break
        return L
    
    # Initial assignment: try to be smart
    # For each prime, initially set r_p to cover small x values
    residues = np.zeros(n_primes, dtype=np.int64)
    for i, p in enumerate(primes):
        # Choose r_p to maximize coverage of [1, some range]
        # Simple heuristic: r_p = 1 (covers 1, 1+p, 1+2p, ...)
        residues[i] = 1 % p
    
    best_L = compute_L_fast(residues)
    best_residues = residues.copy()
    
    # Local search
    import random
    rng = random.Random(42)
    
    max_iterations = 50000
    for it in range(max_iterations):
        if time.time() > deadline:
            break
        if it % 100 == 0:
            # Occasionally do a random restart or perturbation
            pass
        
        # Pick a random prime to change
        i = rng.randrange(n_primes)
        p = primes[i]
        old_r = residues[i]
        
        # Try a new residue
        new_r = rng.randrange(p)
        residues[i] = new_r
        
        new_L = compute_L_fast(residues)
        
        if new_L >= best_L:
            best_L = new_L
            best_residues = residues.copy()
            # Keep the change
        else:
            # Revert
            residues[i] = old_r
    
    # Final refinement: try all residues for each prime
    for i in range(n_primes):
        if time.time() > deadline:
            break
        p = primes[i]
        old_r = best_residues[i]
        best_local = best_L
        best_local_r = old_r
        for r in range(p):
            if r == old_r:
                continue
            best_residues[i] = r
            L = compute_L_fast(best_residues)
            if L > best_local:
                best_local = L
                best_local_r = r
        best_residues[i] = best_local_r
        best_L = best_local
    
    # Convert to list of ints
    return [int(r) for r in best_residues]
