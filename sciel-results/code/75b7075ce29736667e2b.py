import time
import numpy as np
import math

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
    
    # Start with a dynamic L_max
    L_max = 1500

    def get_score(residues, L_max):
        coverage = np.zeros(L_max + 1, dtype=np.int32)
        for i, p in enumerate(primes):
            r = residues[i]
            if r == 0:
                first = p
            else:
                first = r
            if first <= L_max:
                coverage[first:L_max+1:p] += 1
        
        below = np.where(coverage[1:] < 2)[0]
        if len(below) == 0:
            return L_max
        return below[0]

    # Initial random solution
    residues = np.array([np.random.randint(0, p) for p in primes], dtype=np.int64)
    current_score = get_score(residues, L_max)
    best_residues = residues.copy()
    best_score = current_score
    
    # Simulated Annealing Parameters
    T = 50.0
    cooling = 0.99
    min_T = 0.1
    steps_per_temp = 500

    while T > min_T and time.time() < deadline - 10.0:
        for _ in range(steps_per_temp):
            if time.time() > deadline - 10.0:
                break
            
            # Pick a random prime to change
            idx = np.random.randint(0, n_primes)
            p = primes[idx]
            old_r = residues[idx]
            
            # Pick a new random residue
            new_r = np.random.randint(0, p)
            if new_r == old_r:
                continue
            
            # Apply change
            residues[idx] = new_r
            
            # Evaluate
            new_score = get_score(residues, L_max)
            
            # If we hit the limit, expand L_max for future checks
            if new_score == L_max:
                L_max += 100
                
            delta = new_score - current_score
            
            if delta > 0 or np.random.rand() < math.exp(delta / T):
                current_score = new_score
                if current_score > best_score:
                    best_score = current_score
                    best_residues = residues.copy()
            else:
                # Revert
                residues[idx] = old_r
        
        T *= cooling

    # Final polish: Greedy local search on the best solution found
    residues = best_residues.copy()
    current_score = get_score(residues, L_max)
    
    improved = True
    while improved and time.time() < deadline - 5.0:
        improved = False
        for i in range(n_primes):
            p = primes[i]
            old_r = residues[i]
            best_r = old_r
            best_local_score = current_score
            
            # Try all residues for this prime
            for r in range(p):
                if r == old_r:
                    continue
                residues[i] = r
                score = get_score(residues, L_max)
                if score > best_local_score:
                    best_local_score = score
                    best_r = r
                if time.time() > deadline - 5.0:
                    break
            
            residues[i] = best_r
            if best_r != old_r:
                current_score = best_local_score
                if current_score > best_score:
                    best_score = current_score
                    best_residues = residues.copy()
                improved = True

    return [int(r) for r in best_residues]
