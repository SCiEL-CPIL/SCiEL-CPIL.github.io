import time
import numpy as np
import random

def _primes_up_to(n):
    sieve = [True] * (n + 1)
    sieve[0] = sieve[1] = False
    for i in range(2, int(n ** 0.5) + 1):
        if sieve[i]:
            for j in range(i * i, n + 1, i):
                sieve[j] = False
    return [i for i in range(2, n + 1) if sieve[i]]

def _compute_L(counts, max_L):
    below = np.where(counts[1:max_L + 1] < 2)[0]
    if len(below) == 0:
        return max_L
    return int(below[0])

def _build_counts(residues, primes, L_max):
    counts = np.zeros(L_max + 1, dtype=np.int32)
    for i, p in enumerate(primes):
        r = int(residues[i])
        if r == 0:
            if p <= L_max:
                counts[p::p] += 1
        else:
            if r <= L_max:
                counts[r::p] += 1
    return counts

def construct():
    start_time = time.time()
    deadline = start_time + 115.0
    
    primes = _primes_up_to(1000)
    n_primes = len(primes)
    N = 50000
    
    best_L = 0
    best_residues = None
    
    rng = np.random.default_rng(42)
    random.seed(42)
    
    def build_counts(residues):
        counts = np.zeros(N + 1, dtype=np.int32)
        for i, p in enumerate(primes):
            r = int(residues[i])
            if r == 0:
                if p <= N:
                    counts[p::p] += 1
            else:
                if r <= N:
                    counts[r::p] += 1
        return counts
    
    def get_score(counts):
        failures = np.nonzero(counts[1:N+1] < 2)[0]
        if len(failures) == 0:
            return N
        return int(failures[0]) + 1
    
    initial_strategies = [
        lambda i, p: random.randint(0, p - 1),
        lambda i, p: 1 % p,
        lambda i, p: (p // 2) % p,
        lambda i, p: (i * 13 + 5) % p,
        lambda i, p: (i * 7 + 3) % p,
        lambda i, p: ((i + 2) // 2) % p,
        lambda i, p: (i * 31 + 11) % p,
        lambda i, p: (i * 17 + 9) % p,
    ]
    
    for strat in initial_strategies:
        if time.time() > deadline - 5.0:
            break
            
        residues = np.array([strat(i, p) for i, p in enumerate(primes)], dtype=np.int64)
        counts = build_counts(residues)
        current_score = get_score(counts)
        
        if current_score > best_L:
            best_L = current_score
            best_residues = residues.copy()
            
        # Local search with plateau exploration
        stall_count = 0
        max_stall = 3000
        
        while time.time() < deadline - 2.0:
            i = int(rng.integers(n_primes))
            p = primes[i]
            r_old = int(residues[i])
            r_new = int(rng.integers(0, p))
            
            if r_old == r_new:
                continue
            
            # Update counts
            if r_old == 0:
                if p <= N:
                    counts[p::p] -= 1
            else:
                if r_old <= N:
                    counts[r_old::p] -= 1
                    
            if r_new == 0:
                if p <= N:
                    counts[p::p] += 1
            else:
                if r_new <= N:
                    counts[r_new::p] += 1
            
            new_score = get_score(counts)
            
            accepted = False
            if new_score > current_score:
                accepted = True
            elif new_score == current_score:
                if rng.random() < 0.5:
                    accepted = True
            
            if accepted:
                residues[i] = r_new
                current_score = new_score
                stall_count = 0
                
                if current_score > best_L:
                    best_L = current_score
                    best_residues = residues.copy()
            else:
                # Revert
                if r_new == 0:
                    if p <= N:
                        counts[p::p] -= 1
                else:
                    if r_new <= N:
                        counts[r_new::p] -= 1
                        
                if r_old == 0:
                    if p <= N:
                        counts[p::p] += 1
                else:
                    if r_old <= N:
                        counts[r_old::p] += 1
                        
                stall_count += 1
                if stall_count > max_stall:
                    break
    
    # Final verification
    if best_residues is not None and time.time() < deadline - 1.0:
        N_verify = 100000
        counts_v = np.zeros(N_verify + 1, dtype=np.int32)
        for i, p in enumerate(primes):
            r = int(best_residues[i])
            if r == 0:
                if p <= N_verify:
                    counts_v[p::p] += 1
            else:
                if r <= N_verify:
                    counts_v[r::p] += 1
        
        failures = np.nonzero(counts_v[1:N_verify+1] < 2)[0]
        if len(failures) == 0:
            final_L = N_verify
        else:
            final_L = int(failures[0]) + 1
        
        # Return best residues found
        return [int(r) for r in best_residues]
    elif best_residues is not None:
        return [int(r) for r in best_residues]
    else:
        return [0] * n_primes
