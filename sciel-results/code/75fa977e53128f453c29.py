"""Optimized hill-climbing for doubly-covered prefix maximization."""
import time
import random
import numpy as np

# EVOLVE-BLOCK-START

def _primes_upto(n):
    sieve = bytearray(b'\x01') * (n + 1)
    sieve[0:2] = b'\x00\x00'
    for i in range(2, int(n**0.5) + 1):
        if sieve[i]:
            sieve[i*i::i] = bytearray(len(sieve[i*i::i]))
    return [i for i in range(2, n + 1) if sieve[i]]

def _compute_L(residues, primes, max_check=100000):
    n = len(primes)
    coverage = np.zeros(max_check + 1, dtype=np.int32)
    for i in range(n):
        p = primes[i]
        r = residues[i]
        if r == 0:
            start = p
        else:
            start = r
        if start <= max_check:
            coverage[start::p] += 1
    L = 0
    for x in range(1, max_check + 1):
        if coverage[x] >= 2:
            L = x
        else:
            break
    return L

def construct():
    start_time = time.time()
    deadline = start_time + 135
    
    primes = _primes_upto(1000)
    n = len(primes)
    
    # Initialize
    residues = [((i + 2) // 2) % p for i, p in enumerate(primes)]
    current_L = _compute_L(residues, primes)
    best_L = current_L
    best_residues = residues[:]
    
    # Precompute for each prime, the positions it covers in [1, max_check]
    # coverage[i][x] = 1 if prime i covers x, else 0
    # Too memory heavy. Instead, use incremental updates.
    
    # Strategy: maintain coverage array, and when we change residue for prime i,
    # subtract old coverage and add new coverage.
    
    max_check = 50000
    coverage = np.zeros(max_check + 1, dtype=np.int32)
    
    def update_coverage(prime_idx, old_r, new_r):
        p = primes[prime_idx]
        # Remove old
        if old_r == 0:
            old_start = p
        else:
            old_start = old_r
        if old_start <= max_check:
            coverage[old_start::p] -= 1
        # Add new
        if new_r == 0:
            new_start = p
        else:
            new_start = new_r
        if new_start <= max_check:
            coverage[new_start::p] += 1
    
    # Initialize coverage
    for i in range(n):
        p = primes[i]
        r = residues[i]
        if r == 0:
            start = p
        else:
            start = r
        if start <= max_check:
            coverage[start::p] += 1
    
    def get_L_from_coverage(coverage, from_L=0):
        L = from_L
        for x in range(from_L + 1, max_check + 1):
            if coverage[x] >= 2:
                L = x
            else:
                break
        return L
    
    # Verify initial L
    current_L = get_L_from_coverage(coverage, 0)
    best_L = current_L
    best_residues = residues[:]
    
    no_improve = 0
    max_no_improve = 200
    
    while time.time() < deadline - 5:
        # Try each prime once
        order = list(range(n))
        random.shuffle(order)
        
        round_improved = False
        for prime_idx in order:
            if time.time() > deadline - 5:
                break
            
            p = primes[prime_idx]
            old_r = residues[prime_idx]
            
            # Generate candidates
            candidates = [old_r]
            # Residues covering boundary region
            for dx in range(-5, 10):
                x = current_L + dx
                if x >= 1:
                    r = x % p
                    if r not in candidates:
                        candidates.append(r)
            # Random
            for _ in range(min(30, p)):
                r = random.randint(0, p - 1)
                if r not in candidates:
                    candidates.append(r)
            
            best_r = old_r
            best_L = current_L
            
            for r in candidates:
                if r == old_r:
                    continue
                # Apply change
                update_coverage(prime_idx, old_r, r)
                new_L = get_L_from_coverage(coverage, current_L)
                if new_L > best_L:
                    best_L = new_L
                    best_r = r
                else:
                    # Revert
                    update_coverage(prime_idx, r, old_r)
            
            if best_r != old_r:
                # Apply best
                update_coverage(prime_idx, old_r, best_r)
                residues[prime_idx] = best_r
                current_L = best_L
                if current_L > best_L:
                    best_L = current_L
                    best_residues = residues[:]
                    no_improve = 0
                    round_improved = True
                else:
                    no_improve += 1
            else:
                no_improve += 1
        
        if not round_improve:
            no_improve += 1
        
        if no_improve >= max_no_improve:
            if time.time() > deadline - 10:
                break
            # Perturb
            for _ in range(15):
                idx = random.randint(0, n - 1)
                old_r = residues[idx]
                new_r = random.randint(0, primes[idx] - 1)
                update_coverage(idx, old_r, new_r)
                residues[idx] = new_r
            current_L = get_L_from_coverage(coverage, 0)
            no_improve = 0
    
    return best_residues

# EVOLVE-BLOCK-END
