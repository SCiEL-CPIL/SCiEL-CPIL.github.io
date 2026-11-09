"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import math
import numpy as np

# EVOLVE-BLOCK-START

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # safety margin before 120s kill

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
    max_check = 100000

    def evaluate(residues):
        """Return the largest L such that every integer in 1..L is covered by at least 2 congruences."""
        counts = np.zeros(max_check + 1, dtype=np.int32)
        
        for i in range(n):
            p = primes[i]
            r = residues[i]
            if r == 0:
                counts[p::p] += 1
            else:
                counts[r::p] += 1
        
        # Find the first index where counts < 2
        mask = counts < 2
        if np.any(mask[1:]):
            first_bad_idx = np.argmax(mask[1:])
            L = first_bad_idx
        else:
            L = max_check
            
        return L

    # Initialization: Random
    random.seed(42)
    residues = [random.randrange(primes[i]) for i in range(n)]
    current_L = evaluate(residues)
    best_residues = residues[:]
    best_L = current_L
    
    if time.time() > deadline - 10:
        return best_residues

    # Local Search
    steps = 0
    stuck_count = 0
    
    while time.time() < deadline:
        # Pick a random index
        idx = random.randrange(n)
        old_r = residues[idx]
        p = primes[idx]
        
        # Try a random new residue
        new_r = random.randrange(p)
        if new_r == old_r:
            continue
            
        residues[idx] = new_r
        L = evaluate(residues)
        
        if L > best_L:
            best_L = L
            best_residues = residues[:]
            stuck_count = 0
        else:
            # Revert
            residues[idx] = old_r
            stuck_count += 1
        
        steps += 1
        
        # If stuck for a while, kick
        if stuck_count > 500:
            # Change 5 random residues
            for _ in range(5):
                k = random.randrange(n)
                residues[k] = random.randrange(primes[k])
            L = evaluate(residues)
            if L > best_L:
                best_L = L
                best_residues = residues[:]
            else:
                # Revert to best state if kick failed
                residues = best_residues[:]
            stuck_count = 0

        # If time is running low, stop
        if time.time() > deadline - 5:
            break

    return best_residues

# EVOLVE-BLOCK-END
