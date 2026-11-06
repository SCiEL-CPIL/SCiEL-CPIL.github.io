"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import random

def construct():
    primes = [p for p in range(2, 1001) if all(p % q for q in range(2, int(p ** 0.5) + 1))]
    n_primes = len(primes)
    primes_arr = np.array(primes, dtype=np.int64)
    
    # Work with a window of size W
    W = 2000  # We'll try to cover [1, W]
    
    # Initial residues
    residues = np.array([((i + 2) // 2) % p for i, p in enumerate(primes)], dtype=np.int64)
    
    # Compute initial coverage
    coverage = np.zeros(W + 1, dtype=np.int32)
    for i in range(n_primes):
        p = primes_arr[i]
        r = residues[i]
        start = r if r >= 1 else p
        if start <= W:
            coverage[start:W+1:p] += 1
    
    # Find current max L
    def get_max_L(coverage):
        # Find largest L such that coverage[1:L+1] all >= 2
        bad = np.where(coverage[1:] < 2)[0]
        if len(bad) == 0:
            return W
        return bad[0]  # first bad x, so L = bad[0] - 1... wait
        # coverage[x] < 2 means x is bad. L is largest such that all 1..L good.
        # So L = (first bad x) - 1, or W if no bad.
    
    # Let me redefine
    def compute_L(coverage):
        bad_positions = np.where(coverage[1:W+1] < 2)[0]
        if len(bad_positions) == 0:
            return W
        first_bad = bad_positions[0] + 1  # 1-indexed
        return first_bad - 1
    
    current_L = compute_L(coverage)
    best_L = current_L
    best_residues = residues.copy()
    
    # Local search
    random.seed(123)
    T = 0.1
    steps = 20000
    
    for step in range(steps):
        i = random.randint(0, n_primes - 1)
        p = primes_arr[i]
        old_r = residues[i]
        new_r = random.randint(0, p - 1)
        if new_r == old_r:
            continue
        
        # Update coverage: remove old_r contributions, add new_r contributions
        # Old: x ≡ old_r (mod p), x in [1, W]
        old_start = old_r if old_r >= 1 else p
        if old_start <= W:
            old_xs = np.arange(old_start, W + 1, p)
            coverage[old_xs] -= 1
        
        new_start = new_r if new_r >= 1 else p
        if new_start <= W:
            new_xs = np.arange(new_start, W + 1, p)
            coverage[new_xs] += 1
        
        new_L = compute_L(coverage)
        
        if new_L >= best_L:
            best_L = new_L
            best_residues = residues.copy()
        else:
            if random.random() < T:
                residues[i] = new_r
            else:
                # Revert
                if new_start <= W:
                    coverage[new_xs] -= 1
                if old_start <= W:
                    coverage[old_xs] += 1
        
        T *= 0.9995
    
    # Try to extend beyond W if best_L == W
    if best_L >= W:
        # Extend window
        W2 = 5000
        coverage2 = np.zeros(W2 + 1, dtype=np.int32)
        for i in range(n_primes):
            p = primes_arr[i]
            r = best_residues[i]
            start = r if r >= 1 else p
            if start <= W2:
                coverage2[start:W2+1:p] += 1
        # Check if [1, W2] is covered
        bad = np.where(coverage2[1:W2+1] < 2)[0]
        if len(bad) == 0:
            # Even better, try to go further
            W3 = 10000
            coverage3 = np.zeros(W3 + 1, dtype=np.int32)
            for i in range(n_primes):
                p = primes_arr[i]
                r = best_residues[i]
                start = r if r >= 1 else p
                if start <= W3:
                    coverage3[start:W3+1:p] += 1
            bad3 = np.where(coverage3[1:W3+1] < 2)[0]
            if len(bad3) == 0:
                pass  # Can go even further, but for now return best_residues
    
    return [int(r) for r in best_residues]
# EVOLVE-BLOCK-END
