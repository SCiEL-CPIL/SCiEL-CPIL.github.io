import numpy as np
import time
import random

# EVOLVE-BLOCK-START
def construct():
    def get_primes(limit):
        sieve = [True] * (limit + 1)
        sieve[0] = sieve[1] = False
        for i in range(2, int(limit ** 0.5) + 1):
            if sieve[i]:
                sieve[i*i:limit+1:i] = [False] * len(sieve[i*i:limit+1:i])
        return [i for i, is_prime in enumerate(sieve) if is_prime]

    primes = get_primes(1000)
    n_primes = len(primes)
    X_MAX = 100000

    # Precompute indices for fast updates
    # indices[i][r] is array of indices in [0, X_MAX) covered by residue r for prime i
    indices_cache = []
    for p in primes:
        cache_p = []
        for r in range(p):
            start = (r - 1) % p
            # Ensure start is in range [0, p)
            # np.arange(start, X_MAX, p)
            cache_p.append(np.arange(start, X_MAX, p))
        indices_cache.append(cache_p)

    def compute_initial_counts(residues):
        counts = np.zeros(X_MAX, dtype=np.int8)
        for i, (r, p) in enumerate(zip(residues, primes)):
            counts[indices_cache[i][r]] += 1
        return counts

    def get_L(counts):
        invalid = counts < 2
        if not np.any(invalid):
            return X_MAX
        return np.argmax(invalid)

    def update_counts(counts, i, r_old, r_new):
        # Subtract old
        counts[indices_cache[i][r_old]] -= 1
        # Add new
        counts[indices_cache[i][r_new]] += 1

    start_time = time.time()
    deadline = start_time + 115.0 # Safety margin

    # Initial state
    residues = [random.randint(0, p-1) for p in primes]
    counts = compute_initial_counts(residues)
    current_L = get_L(counts)
    
    best_L = current_L
    best_residues = residues.copy()
    best_counts = counts.copy()
    
    T = 5.0
    T_min = 0.1
    cooling_rate = 0.9995
    
    no_improve_counter = 0
    max_no_improve = 500
    
    while time.time() < deadline:
        if no_improve_counter > max_no_improve:
            if random.random() < 0.5:
                residues = best_residues.copy()
                counts = best_counts.copy()
                current_L = best_L
            else:
                residues = [random.randint(0, p-1) for p in primes]
                counts = compute_initial_counts(residues)
                current_L = get_L(counts)
            no_improve_counter = 0
            T = 5.0
            
        i = random.randint(0, n_primes - 1)
        p = primes[i]
        r_old = residues[i]
        
        # Bias towards fixing the current failure point
        # current_L is the first index with count < 2.
        # We want to increase count at index current_L.
        # Integer is current_L + 1.
        # r_new = (current_L + 1) % p
        if random.random() < 0.7:
            r_new = (current_L + 1) % p
        else:
            r_new = random.randint(0, p - 1)
            
        if r_new == r_old:
            continue
            
        update_counts(counts, i, r_old, r_new)
        new_L = get_L(counts)
        
        delta = new_L - current_L
        
        if delta > 0:
            residues[i] = r_new
            current_L = new_L
            no_improve_counter = 0
            if new_L > best_L:
                best_L = new_L
                best_residues = residues.copy()
                best_counts = counts.copy()
        else:
            if delta == 0:
                if random.random() < 0.5:
                    residues[i] = r_new
                    current_L = new_L
                    no_improve_counter += 1
                else:
                    update_counts(counts, i, r_new, r_old)
            else:
                prob = np.exp(delta / T)
                if random.random() < prob:
                    residues[i] = r_new
                    current_L = new_L
                    no_improve_counter += 1
                else:
                    update_counts(counts, i, r_new, r_old)
            
        no_improve_counter += 1
        T *= cooling_rate
        if T < T_min:
            T = T_min

    return best_residues
# EVOLVE-BLOCK-END
