import numpy as np
import time
import random
import math

def _sieve_primes(limit):
    """Return list of primes <= limit using sieve of Eratosthenes."""
    if limit < 2:
        return []
    is_prime = bytearray(b'\x01') * (limit + 1)
    is_prime[0] = 0
    is_prime[1] = 0
    for i in range(2, int(limit ** 0.5) + 1):
        if is_prime[i]:
            step = i
            start = i * i
            is_prime[start:limit+1:step] = bytearray(len(range(start, limit+1, step)))
    return [i for i in range(2, limit + 1) if is_prime[i]]

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 120s limit
    
    primes = _sieve_primes(1000)
    n_primes = len(primes)
    
    # Search range: We aim for L up to 20,000 for faster iteration.
    N = 20000
    
    best_residues = None
    best_score = 0
    
    # Heuristic initializations
    initial_strategies = [
        lambda i, p: random.randint(0, p - 1),
        lambda i, p: 1 % p,
        lambda i, p: (p // 2) % p,
        lambda i, p: (i * 13 + 5) % p,
        lambda i, p: (i * 7 + 3) % p,
        lambda i, p: ((i + 2) // 2) % p,
        lambda i, p: (i * 31 + 11) % p,
        lambda i, p: (i * 17 + 2) % p,
    ]
    
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
        # We care about x=1..N
        # Find first index >= 1 where counts < 2
        failures = np.nonzero(counts[1:N+1] < 2)[0]
        if len(failures) == 0:
            return N
        return int(failures[0]) + 1

    def get_vulnerable_count(counts):
        # Count integers in 1..N with coverage exactly 1
        return int(np.sum(counts[1:N+1] == 1))

    # Run multiple restarts
    for strat in initial_strategies:
        if time.time() > deadline - 5.0:
            break
            
        residues = np.array([strat(i, p) for i, p in enumerate(primes)], dtype=np.int64)
        counts = build_counts(residues)
        current_score = get_score(counts)
        current_vulnerable = get_vulnerable_count(counts)
        
        if current_score > best_score:
            best_score = current_score
            best_residues = residues.copy()
            
        # Local Search with Simulated Annealing on vulnerable count
        T = 1.0  # Temperature
        T_min = 0.01
        cooling = 0.9995
        stall_count = 0
        max_stall = 2000
        
        while time.time() < deadline - 2.0:
            i = int(rng.integers(n_primes))
            p = primes[i]
            r_old = int(residues[i])
            
            # Choose move type: 70% directed towards first failure, 30% random
            if rng.random() < 0.7:
                target = current_score + 1
                if target > N:
                    target = N // 2
                r_new = target % p
            else:
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
            new_vulnerable = get_vulnerable_count(counts)
            
            # Acceptance Criterion:
            # 1. If score improves, accept.
            # 2. If score same, accept if vulnerable count decreases (or with prob exp(-delta_v/T) if increases).
            # 3. If score decreases, reject (we prioritize maximizing L).
            
            accepted = False
            if new_score > current_score:
                accepted = True
            elif new_score == current_score:
                delta_v = new_vulnerable - current_vulnerable
                if delta_v <= 0:
                    accepted = True
                else:
                    # Accept bad moves with probability exp(-delta_v / T)
                    if rng.random() < math.exp(-delta_v / T):
                        accepted = True
            else:
                # Score decreased, do not accept
                accepted = False
            
            if accepted:
                residues[i] = r_new
                current_score = new_score
                current_vulnerable = new_vulnerable
                stall_count = 0
                
                if current_score > best_score:
                    best_score = current_score
                    best_residues = residues.copy()
            else:
                # Revert counts
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
                    break # Restart with new initialization
            
            # Cool down
            T = max(T_min, T * cooling)
            
    # Final Verification: Check if we can go beyond N
    if best_residues is not None and time.time() < deadline - 1.0:
        # Check up to 100,000
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
            final_score = N_verify
        else:
            final_score = int(failures[0]) + 1
            
    return best_residues.tolist()
