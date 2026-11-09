import time
import numpy as np
import random
import math

N = 5000

def _primes(limit):
    sieve = [True] * (limit + 1)
    sieve[0] = sieve[1] = False
    for p in range(2, int(limit ** 0.5) + 1):
        if sieve[p]:
            for q in range(p * p, limit + 1, p):
                sieve[q] = False
    return [p for p, ok in enumerate(sieve) if ok]

PRIMES = _primes(N)

def construct():
    start_time = time.time()
    deadline = start_time + 290
    
    best_A = None
    best_size = 0
    
    # Precompute residues for each prime
    # prime_residues[p] = list of lists, where prime_residues[p][r] = np.array of x in [0, N] s.t. x % p == r
    prime_residues = {}
    for p in PRIMES:
        if p > N: break
        prime_residues[p] = [np.arange(r, N + 1, p) for r in range(p)]
    
    def run_lazy_greedy(seed=0):
        random.seed(seed)
        A = np.ones(N + 1, dtype=bool)
        current_size = N + 1
        
        # counts[p][r] = number of elements in A with x % p == r
        counts = {}
        for p in PRIMES:
            if p > N: break
            counts[p] = np.zeros(p, dtype=int)
            for r in range(p):
                counts[p][r] = len(prime_residues[p][r])
        
        # To speed up critical prime detection, we can just check all primes <= current_size
        # Since N is small, this is fine.
        
        while True:
            # Find critical primes
            critical_primes = []
            for p in PRIMES:
                if p > current_size:
                    break
                # Check if A covers all residues mod p
                # i.e., counts[p][r] > 0 for all r
                if np.all(counts[p] > 0):
                    critical_primes.append(p)
            
            if not critical_primes:
                break
            
            # Select a critical prime
            # Strategy: Pick the one with the smallest minimum count
            # This minimizes the number of elements removed
            min_counts = {}
            for p in critical_primes:
                min_counts[p] = np.min(counts[p])
            
            # Find primes with global min count
            global_min = min(min_counts.values())
            candidates = [p for p in critical_primes if min_counts[p] == global_min]
            
            # Random tie-breaking
            p_star = random.choice(candidates)
            
            # Find residue to remove
            # Pick r with min count
            min_r_count = np.min(counts[p_star])
            candidate_rs = [r for r in range(p_star) if counts[p_star][r] == min_r_count]
            r_star = random.choice(candidate_rs)
            
            # Remove elements
            to_remove = prime_residues[p_star][r_star]
            # Only remove those currently in A
            to_remove = to_remove[A[to_remove]]
            
            if len(to_remove) == 0:
                # Should not happen if counts are correct
                break
                
            # Update A
            A[to_remove] = False
            current_size -= len(to_remove)
            
            # Update counts for all primes
            for x in to_remove:
                for q in PRIMES:
                    if q > current_size + 100: # Optimization: don't update for large q if not needed? 
                        # Actually we need counts for q <= current_size in next iteration.
                        # But current_size shrinks.
                        # Let's just update all q <= N.
                        pass
                    # x % q
                    r_q = x % q
                    counts[q][r_q] -= 1
                    
        # Extract result
        A_list = np.where(A)[0].tolist()
        return A_list, len(A_list)

    # Run multiple trials
    for i in range(50):
        if time.time() > deadline - 5:
            break
        A_list, size = run_lazy_greedy(seed=i)
        if size > best_size:
            best_size = size
            best_A = A_list
            
    # Polish with Local Search / SA?
    # The lazy greedy result is a set of removed classes.
    # We can try to "un-remove" some classes if it doesn't break admissibility.
    # Or swap residues.
    
    # Given the time, the lazy greedy is likely strong enough.
    # Let's add a simple polish: 
    # For the best A found, identify the removed classes.
    # Actually, we don't track them explicitly in the return, but we can re-derive them or just try to add elements back?
    # Adding elements back is hard.
    
    # Let's just return the best greedy result.
    if best_A is not None:
        return best_A
    return [0]
