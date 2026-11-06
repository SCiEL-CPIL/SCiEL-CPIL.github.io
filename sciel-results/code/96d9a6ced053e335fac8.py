import numpy as np
import time
import random
import math

def construct():
    start_time = time.time()
    # Use 110s deadline to safely return before 120s kill
    deadline = start_time + 110.0
    
    # Generate primes up to 1000
    def sieve(n):
        is_prime = [True] * (n + 1)
        is_prime[0] = is_prime[1] = False
        for i in range(2, int(n ** 0.5) + 1):
            if is_prime[i]:
                for j in range(i * i, n + 1, i):
                    is_prime[j] = False
        return [i for i, prime in enumerate(is_prime) if prime]
        
    primes = sieve(1000)
    n_primes = len(primes)
    
    # Score is capped at 10^5. We don't need to track L beyond that for the score.
    # If L > 10^5, score is 10^5.
    # Using MAX_L = 100000 reduces memory and scan time.
    MAX_L = 100000
    coverage = np.zeros(MAX_L + 1, dtype=np.int32)
    
    # Initialize residues randomly
    residues = [random.randint(0, p-1) for p in primes]
    
    # Initial coverage
    for i, p in enumerate(primes):
        r = residues[i]
        coverage[r::p] += 1
        
    def get_L():
        # Find first x in 1..MAX_L where coverage[x] < 2
        invalid = np.where(coverage[1:] < 2)[0]
        if len(invalid) == 0:
            return MAX_L
        else:
            return int(invalid[0])
            
    current_L = get_L()
    best_L = current_L
    best_residues = residues.copy()
    
    # Simulated Annealing with Guided Moves
    T = 10.0
    cooling = 0.999
    min_T = 0.1
    
    exp = math.exp
    rand = random.random
    randint = random.randint
    steps = 0
    
    while time.time() < deadline:
        # Find the first gap
        if current_L < MAX_L:
            x_bad = current_L + 1
            # Find primes that do NOT cover x_bad
            # r_p != x_bad % p
            candidates = []
            for i in range(n_primes):
                p = primes[i]
                if residues[i] != x_bad % p:
                    candidates.append(i)
            
            if candidates:
                # Pick a random candidate
                i = candidates[randint(0, len(candidates) - 1)]
                p = primes[i]
                old_r = residues[i]
                new_r = x_bad % p
                
                # Update coverage
                coverage[old_r::p] -= 1
                coverage[new_r::p] += 1
                
                new_L = get_L()
                
                if new_L > current_L:
                    current_L = new_L
                    residues[i] = new_r
                    if current_L > best_L:
                        best_L = current_L
                        best_residues = residues.copy()
                else:
                    delta = new_L - current_L
                    if delta < 0:
                        prob = exp(delta / T)
                        if rand() < prob:
                            current_L = new_L
                            residues[i] = new_r
                        else:
                            # Revert
                            coverage[new_r::p] -= 1
                            coverage[old_r::p] += 1
                    else:
                        # delta == 0, accept
                        current_L = new_L
                        residues[i] = new_r
            else:
                # All primes already cover x_bad?
                # This implies coverage[x_bad] >= 1, but get_L said < 2.
                # If all cover, coverage >= 168. So this shouldn't happen for a gap.
                # Just do a random move to escape potential local minima or bugs.
                i = randint(0, n_primes - 1)
                p = primes[i]
                old_r = residues[i]
                new_r = randint(0, p - 1)
                if old_r == new_r:
                    continue
                coverage[old_r::p] -= 1
                coverage[new_r::p] += 1
                new_L = get_L()
                if new_L > current_L:
                    current_L = new_L
                    residues[i] = new_r
                    if current_L > best_L:
                        best_L = current_L
                        best_residues = residues.copy()
                else:
                    delta = new_L - current_L
                    if delta < 0:
                        prob = exp(delta / T)
                        if rand() < prob:
                            current_L = new_L
                            residues[i] = new_r
                        else:
                            coverage[new_r::p] -= 1
                            coverage[old_r::p] += 1
                    else:
                        current_L = new_L
                        residues[i] = new_r
        else:
            # No gap, we are at MAX_L or higher
            i = randint(0, n_primes - 1)
            p = primes[i]
            old_r = residues[i]
            new_r = randint(0, p - 1)
            if old_r == new_r:
                continue
            coverage[old_r::p] -= 1
            coverage[new_r::p] += 1
            new_L = get_L()
            if new_L > current_L:
                current_L = new_L
                residues[i] = new_r
                if current_L > best_L:
                    best_L = current_L
                    best_residues = residues.copy()
            else:
                delta = new_L - current_L
                if delta < 0:
                    prob = exp(delta / T)
                    if rand() < prob:
                        current_L = new_L
                        residues[i] = new_r
                    else:
                        coverage[new_r::p] -= 1
                        coverage[old_r::p] += 1
                else:
                    current_L = new_L
                    residues[i] = new_r

        T *= cooling
        if T < min_T:
            T = min_T
            
        steps += 1
        if steps % 100 == 0 and time.time() >= deadline:
            break
            
    return best_residues
