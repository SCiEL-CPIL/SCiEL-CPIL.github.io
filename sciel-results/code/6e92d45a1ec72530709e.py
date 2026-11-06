import time
import numpy as np

_DEADLINE = time.time() + 140  # Safety margin before 150s kill

def _sieve_primes(limit):
    sieve = bytearray([1]) * (limit + 1)
    sieve[0] = sieve[1] = 0
    for i in range(2, int(limit**0.5) + 1):
        if sieve[i]:
            sieve[i*i::i] = bytearray(len(sieve[i*i::i]))
    return [i for i, v in enumerate(sieve) if v]

def construct():
    start_time = time.time()
    primes = _sieve_primes(1000)
    n_primes = len(primes)
    max_L = 100000
    
    # Heuristic Initialization
    # Try to cover 1..200 with at least 2 congruences
    residues = np.zeros(n_primes, dtype=np.int32)
    # Set all to 0 initially?
    # Or random?
    # Let's try to explicitly assign residues to cover small numbers.
    
    # For each x in 1..100, pick 2 primes and assign r_p = x % p
    # To avoid conflicts, pick primes that are not yet "used" for this x?
    # Actually, a prime can cover multiple x's if they are congruent mod p.
    # Simple heuristic:
    # For x in 1..100:
    #   Pick 2 random primes p1, p2
    #   Set r_p1 = x % p1, r_p2 = x % p2
    # This might overwrite previous assignments.
    
    # Better:
    # Initialize r_p = 0 for all p.
    # This covers multiples of p.
    # 1 is not covered by any r_p=0 (unless p=1, not prime).
    # So 1 is covered 0 times.
    # We need to fix 1.
    
    # Let's start with random and let the search fix it.
    # But a good start helps.
    
    # Try: r_p = 1 % p = 1 for all p?
    # Then 1 is covered by all 168 primes.
    # 2 is covered by p where 2 % p == 1 => p=1 (no), p>2 => 2%p=2 != 1.
    # So 2 is covered 0 times.
    
    # How about r_p = p-1?
    # x=1: 1 % p != p-1 (unless p=2, 1%2=1, p-1=1. Yes).
    # So only p=2 covers 1.
    
    # Let's use the local search with a random start. It's robust.
    residues = np.array([np.random.randint(0, p) for p in primes], dtype=np.int32)
    
    coverage = np.zeros(max_L + 1, dtype=np.int32)
    
    def update_coverage(idx, old_r, new_r):
        if old_r == new_r:
            return
        p = primes[idx]
        if old_r == 0:
            start = p
        else:
            start = old_r
        if start <= max_L:
            coverage[start::p] -= 1
            
        if new_r == 0:
            start = p
        else:
            start = new_r
        if start <= max_L:
            coverage[start::p] += 1

    def get_L():
        bad = np.where(coverage[1:] < 2)[0]
        if len(bad) == 0:
            return max_L
        return bad[0]

    # Initial coverage
    for i, p in enumerate(primes):
        r = residues[i]
        if r == 0:
            start = p
        else:
            start = r
        if start <= max_L:
            coverage[start::p] += 1
            
    current_L = get_L()
    best_L = current_L
    best_residues = residues.copy()
    
    no_improve_count = 0
    max_no_improve = 100
    
    while time.time() < _DEADLINE:
        if current_L >= max_L:
            break
            
        x_bad = current_L + 1
        
        candidates = []
        for i, p in enumerate(primes):
            target_r = x_bad % p
            if residues[i] != target_r:
                candidates.append((i, target_r))
                
        if not candidates:
            break

        if len(candidates) > 10:
            np.random.shuffle(candidates)
            candidates = candidates[:10]
            
        best_candidate = None
        best_candidate_L = current_L
        
        for idx, new_r in candidates:
            old_r = residues[idx]
            update_coverage(idx, old_r, new_r)
            new_L = get_L()
            
            if new_L > best_candidate_L:
                best_candidate_L = new_L
                best_candidate = (idx, new_r, old_r)
            
            update_coverage(idx, new_r, old_r)
            
        if best_candidate is not None:
            idx, new_r, old_r = best_candidate
            update_coverage(idx, old_r, new_r)
            residues[idx] = new_r
            current_L = best_candidate_L
            if current_L > best_L:
                best_L = current_L
                best_residues = residues.copy()
            no_improve_count = 0
        else:
            no_improve_count += 1
            if no_improve_count > max_no_improve:
                # Random perturbation
                idx = np.random.randint(0, n_primes)
                old_r = residues[idx]
                new_r = np.random.randint(0, primes[idx])
                if new_r != old_r:
                    update_coverage(idx, old_r, new_r)
                    residues[idx] = new_r
                    current_L = get_L()
                    if current_L > best_L:
                        best_L = current_L
                        best_residues = residues.copy()
                no_improve_count = 0

    return best_residues.tolist()
