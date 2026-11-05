"""Optimized solver for the covering problem."""
import random
import time
import numpy as np

def _get_primes(n):
    """Sieve of Eratosthenes to get primes up to n."""
    sieve = [True] * (n + 1)
    sieve[0] = sieve[1] = False
    for i in range(2, int(n ** 0.5) + 1):
        if sieve[i]:
            sieve[i*i::i] = [False] * len(sieve[i*i::i])
    return [i for i, is_prime in enumerate(sieve) if is_prime]

def _count_covered(residues, primes, max_L):
    """
    Fast count of the largest L such that every integer 1..L is covered at least twice.
    """
    counts = np.zeros(max_L + 1, dtype=np.int32)
    for r, p in zip(residues, primes):
        if r == 0:
            start = p
        else:
            start = r
        if start <= max_L:
            counts[start::p] += 1
    
    # Find the first position in 1..max_L where count < 2
    # We check indices 1 to max_L
    below_two = np.where(counts[1:] < 2)[0]
    if len(below_two) == 0:
        return max_L
    else:
        return below_two[0]

def construct():
    start_time = time.time()
    deadline = start_time + 115  # 5s safety margin for 120s limit
    
    primes = _get_primes(1000)
    n_primes = len(primes)
    
    # Weights for selecting primes: smaller primes have more impact
    weights = np.array([1.0 / p for p in primes])
    weights /= weights.sum()
    
    # Initial random solution
    best_residues = [random.randint(0, p - 1) for p in primes]
    best_L = _count_covered(best_residues, primes, 100000)
    
    # Hill Climbing
    improved = True
    stuck_count = 0
    max_stuck = 200 # Lower threshold for kick
    
    while (time.time() - start_time) < deadline:
        if improved:
            stuck_count = 0
            improved = False
        else:
            stuck_count += 1
            if stuck_count > max_stuck:
                # Random kick: change a random subset of residues
                # Focus on small primes for the kick too?
                # Just random subset
                for _ in range(n_primes // 20):
                    idx = random.choices(range(n_primes), weights=weights, k=1)[0]
                    p = primes[idx]
                    best_residues[idx] = random.randint(0, p - 1)
                best_L = _count_covered(best_residues, primes, 100000)
                stuck_count = 0
                continue

        # Pick a prime to modify, weighted by 1/p
        i = random.choices(range(n_primes), weights=weights, k=1)[0]
        p = primes[i]
        old_r = best_residues[i]
        
        # Try a few random new residues for this prime
        # Increase trials for small primes?
        trials = 10 if p < 100 else 3
        
        for _ in range(trials):
            if (time.time() - start_time) > deadline:
                break
                
            new_r = random.randint(0, p - 1)
            if new_r == old_r:
                continue
                
            test_residues = best_residues[:]
            test_residues[i] = new_r
            
            new_L = _count_covered(test_residues, primes, 100000)
            
            if new_L > best_L:
                best_residues[i] = new_r
                best_L = new_L
                improved = True
                break

    return best_residues
