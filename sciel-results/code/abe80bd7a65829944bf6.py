import time
import random
import math

def construct():
    start_time = time.time()
    deadline = start_time + 45.0  # Safety margin for 60s limit
    
    # 1. Generate pool of 5-smooth numbers (primes 2, 3, 5) up to 10000
    pool = set()
    for a in range(0, 15):
        v2 = 2 ** a
        if v2 > 10000:
            break
        for b in range(0, 15):
            v23 = v2 * (3 ** b)
            if v23 > 10000:
                break
            for c in range(0, 15):
                val = v23 * (5 ** c)
                if val > 10000:
                    break
                pool.add(val)
    pool = sorted(list(pool))
    pool_size = len(pool)
    
    # Precompute smallest prime factor (SPF) up to 20000
    MAX_SUM = 20000
    spf = list(range(MAX_SUM + 1))
    for i in range(2, int(MAX_SUM ** 0.5) + 1):
        if spf[i] == i:
            for j in range(i * i, MAX_SUM + 1, i):
                if spf[j] == j:
                    spf[j] = i
    
    # Precompute prime factors for all pairs in the pool
    # pair_factors[i][j] = set of prime factors of pool[i] + pool[j]
    # To save memory and time, we can store as a list of primes
    # pool size is ~200-300. Pairs ~ 50k.
    # Storing sets might be heavy, but 50k sets of small ints is okay in Python?
    # Let's check size. 300 * 300 / 2 = 45000.
    # Each set has ~2-5 primes.
    # It's fine.
    
    # Optimization: Map pool values to indices for fast lookup
    val_to_idx = {v: i for i, v in enumerate(pool)}
    
    # Precompute factors for all pairs i < j
    # We will store factors[i][j] as a frozenset or tuple of primes
    # Since we need to update counts, we need to iterate over primes.
    # Let's store as a list of primes.
    
    # Actually, we can just compute on the fly if the pool is small enough?
    # 66 pairs * 200000 iterations = 13.2 million factorizations.
    # With SPF, factorization is ~log(n) divisions. ~15 divisions.
    # 13.2M * 15 = 200M operations. Might be too slow in Python (10-20s).
    # Precomputation is safer.
    
    # Let's precompute factors for all pairs in the pool
    # We only need pairs where both elements are in the pool.
    # We will create a dictionary: (idx1, idx2) -> list of primes
    # Or a 2D list if pool is small enough.
    # Pool size is likely around 200-250.
    # 250x250 matrix of lists is fine.
    
    pair_primes = [[None for _ in range(pool_size)] for _ in range(pool_size)]
    
    for i in range(pool_size):
        vi = pool[i]
        for j in range(i + 1, pool_size):
            vj = pool[j]
            s = vi + vj
            factors = set()
            n = s
            while n > 1:
                p = spf[n]
                factors.add(p)
                while n % p == 0:
                    n //= p
            # Store as tuple for hashability and faster iteration
            f_tuple = tuple(factors)
            pair_primes[i][j] = f_tuple
            pair_primes[j][i] = f_tuple
            
    # If time is running out during precomputation, abort and use a simpler method
    if time.time() > deadline - 5.0:
         # Fallback: just return a random set
         return random.sample(pool, 12)
         
    # Initialize with a random subset of the pool
    # To get a good start, we can try to pick numbers that are close to each other
    # or just random. Random is fine with SA.
    current_set = random.sample(pool, 12)
    current_idx_set = [val_to_idx[v] for v in current_set]
    
    # Maintain prime counts
    # prime_count[p] = number of pairs (i,j) in current_set such that p divides a_i + a_j
    prime_counts = {}
    
    # Initialize counts
    for i in range(12):
        for j in range(i + 1, 12):
            idx_i = current_idx_set[i]
            idx_j = current_idx_set[j]
            primes = pair_primes[idx_i][idx_j]
            for p in primes:
                if p in prime_counts:
                    prime_counts[p] += 1
                else:
                    prime_counts[p] = 1
                    
    current_score = len(prime_counts)
    best_score = current_score
    best_set = current_set[:]
    best_idx_set = current_idx_set[:]
    
    # Simulated Annealing
    temp = 1.0
    cooling = 0.9999
    min_temp = 0.01
    
    # We can also try to restart if stuck
    restarts = 0
    max_restarts = 5
    
    while time.time() < deadline:
        if restarts > max_restarts:
            break
            
        # Pick random index to swap
        idx = random.randint(0, 11)
        old_val = current_set[idx]
        old_idx = current_idx_set[idx]
        
        # Pick new value from pool
        # To improve diversity, pick from a random part of the pool
        new_idx = random.randint(0, pool_size - 1)
        new_val = pool[new_idx]
        
        if new_idx == old_idx:
            continue
        if new_val in current_set:
            continue
            
        # Apply change: remove old_idx, add new_idx
        # 1. Remove contributions of old_idx with all other elements
        for k in range(12):
            if k == idx:
                continue
            other_idx = current_idx_set[k]
            primes = pair_primes[old_idx][other_idx]
            for p in primes:
                prime_counts[p] -= 1
                if prime_counts[p] == 0:
                    del prime_counts[p]
                    
        # 2. Add contributions of new_idx with all other elements
        for k in range(12):
            if k == idx:
                continue
            other_idx = current_idx_set[k]
            primes = pair_primes[new_idx][other_idx]
            for p in primes:
                if p in prime_counts:
                    prime_counts[p] += 1
                else:
                    prime_counts[p] = 1
                    
        # Update current state
        current_set[idx] = new_val
        current_idx_set[idx] = new_idx
        new_score = len(prime_counts)
        
        if new_score < current_score:
            current_score = new_score
            if new_score < best_score:
                best_score = new_score
                best_set = current_set[:]
                best_idx_set = current_idx_set[:]
                # Early exit if we hit theoretical minimum for 5-smooth (3)
                # Or even 2 if we are lucky
                if best_score <= 3:
                    # We can try to push for 2, but 3 is very good.
                    # Let's continue for a bit to see if 2 is possible.
                    # If we hit 3, we might be stuck. 
                    # 2 primes means all sums are of form 2^a 3^b.
                    # This is very restrictive.
                    pass
        else:
            # Accept worse solution with probability p
            delta = new_score - current_score
            if delta < 0:
                accept_prob = 1.0
            else:
                accept_prob = math.exp(-delta / temp)
            
            if random.random() < accept_prob:
                current_score = new_score
            else:
                # Revert
                # Remove new_idx contributions
                for k in range(12):
                    if k == idx:
                        continue
                    other_idx = current_idx_set[k]
                    primes = pair_primes[new_idx][other_idx]
                    for p in primes:
                        prime_counts[p] -= 1
                        if prime_counts[p] == 0:
                            del prime_counts[p]
                            
                # Add old_idx contributions back
                for k in range(12):
                    if k == idx:
                        continue
                    other_idx = current_idx_set[k]
                    primes = pair_primes[old_idx][other_idx]
                    for p in primes:
                        if p in prime_counts:
                            prime_counts[p] += 1
                        else:
                            prime_counts[p] = 1
                            
                current_set[idx] = old_val
                current_idx_set[idx] = old_idx
                # current_score remains unchanged
                
        temp *= cooling
        if temp < min_temp:
            temp = min_temp
            restarts += 1
            # Restart with a new random set
            current_set = random.sample(pool, 12)
            current_idx_set = [val_to_idx[v] for v in current_set]
            prime_counts = {}
            for i in range(12):
                for j in range(i + 1, 12):
                    idx_i = current_idx_set[i]
                    idx_j = current_idx_set[j]
                    primes = pair_primes[idx_i][idx_j]
                    for p in primes:
                        if p in prime_counts:
                            prime_counts[p] += 1
                        else:
                            prime_counts[p] = 1
            current_score = len(prime_counts)
            if current_score < best_score:
                best_score = current_score
                best_set = current_set[:]
                best_idx_set = current_idx_set[:]
            temp = 1.0

    return best_set

# EVOLVE-BLOCK-START
# The construct function is defined above. 
# EVOLVE-BLOCK-END
