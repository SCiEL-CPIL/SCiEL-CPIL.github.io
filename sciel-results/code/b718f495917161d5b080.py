"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random

def construct():
    deadline = time.time() + 140
    
    primes = [p for p in range(2, 1001) if all(p % q for q in range(2, int(p ** 0.5) + 1))]
    n = len(primes)
    
    # Initial assignment: try to cover small integers densely
    residues = [(i + 1) % p for i, p in enumerate(primes)]
    
    cap = 5000
    coverage = [0] * (cap + 1)
    for r, p in zip(residues, primes):
        first = r if r > 0 else p
        x = first
        while x <= cap:
            coverage[x] += 1
            x += p
    
    def get_L(coverage):
        L = 0
        for i in range(1, cap + 1):
            if coverage[i] >= 2:
                L = i
            else:
                break
        return L
    
    current_L = get_L(coverage)
    
    rng = random.Random(42)
    
    for it in range(300000):
        if time.time() > deadline:
            break
        
        idx = rng.randint(0, n - 1)
        p = primes[idx]
        old_r = residues[idx]
        new_r = rng.randint(0, p - 1)
        if new_r == old_r:
            continue
        
        # Remove old coverage
        old_first = old_r if old_r > 0 else p
        x = old_first
        while x <= cap:
            coverage[x] -= 1
            x += p
        
        # Add new coverage
        new_first = new_r if new_r > 0 else p
        x = new_first
        while x <= cap:
            coverage[x] += 1
            x += p
        
        new_L = get_L(coverage)
        
        if new_L >= current_L:
            residues[idx] = new_r
            current_L = new_L
        else:
            # Revert
            x = new_first
            while x <= cap:
                coverage[x] -= 1
                x += p
            x = old_first
            while x <= cap:
                coverage[x] += 1
                x += p
    
    return residues
# EVOLVE-BLOCK-END
