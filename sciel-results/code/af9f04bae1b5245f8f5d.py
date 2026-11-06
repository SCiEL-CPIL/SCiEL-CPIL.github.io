import time
import random
from math import isqrt

def construct():
    start_time = time.time()
    deadline = start_time + 85.0
    
    # 1. Precomputation
    MAX_VAL = 10000
    MAX_SUM = 2 * MAX_VAL
    
    # Sieve of Eratosthenes
    is_prime = [True] * (MAX_SUM + 1)
    is_prime[0] = is_prime[1] = False
    for i in range(2, isqrt(MAX_SUM) + 1):
        if is_prime[i]:
            for j in range(i*i, MAX_SUM + 1, i):
                is_prime[j] = False
    
    primes = [i for i, p in enumerate(is_prime) if p]
    prime_index = {p: i for i, p in enumerate(primes)}
    
    # Bitmask for prime factors
    pf_mask = [0] * (MAX_SUM + 1)
    for p in primes:
        bit = 1 << prime_index[p]
        for j in range(p, MAX_SUM + 1, p):
            pf_mask[j] |= bit
            
    def get_score(s):
        mask = 0
        # s is a list of 12 integers
        for i in range(12):
            si = s[i]
            for j in range(i + 1, 12):
                mask |= pf_mask[si + s[j]]
        return mask.bit_count()
        
    def local_search(initial_set, deadline):
        current_set = list(initial_set)
        current_score = get_score(current_set)
        best_set = current_set[:]
        best_score = current_score
        
        # If we already found a very low score, we can stop early or tighten deadline
        # But let's just run until deadline
        
        while time.time() < deadline:
            # Random move: replace one element
            i = random.randint(0, 11)
            old_val = current_set[i]
            
            # Bias towards small numbers?
            # Sums of small numbers are smaller, potentially smoother?
            # Or just uniform.
            new_val = random.randint(1, MAX_VAL)
            
            if new_val == old_val or new_val in current_set:
                continue
                
            new_set = current_set[:]
            new_set[i] = new_val
            new_score = get_score(new_set)
            
            if new_score < current_score:
                current_set = new_set
                current_score = new_score
                if current_score < best_score:
                    best_score = current_score
                    best_set = current_set[:]
            elif new_score == current_score:
                # Accept to explore the plateau
                if random.random() < 0.3:
                    current_set = new_set
                    current_score = new_score
            # else reject
            
            # Occasionally try a swap
            if random.random() < 0.1:
                j = random.randint(0, 11)
                if i != j:
                    temp = current_set[i]
                    current_set[i] = current_set[j]
                    current_set[j] = temp
                    swap_score = get_score(current_set)
                    if swap_score > current_score:
                        # Revert
                        current_set[i], current_set[j] = current_set[j], current_set[i]
                    else:
                        current_score = swap_score
                        if current_score < best_score:
                            best_score = current_score
                            best_set = current_set[:]

        return best_set, best_score

    best_global_set = None
    best_global_score = float('inf')
    
    # Initial sets
    # 1. Small consecutive
    s1 = list(range(1, 13))
    # 2. Small odds
    s2 = [2*i+1 for i in range(12)]
    # 3. Powers of 2
    s3 = [2**i for i in range(12)]
    # 4. Random small
    s4 = random.sample(range(1, 100), 12)
    # 5. Random large
    s5 = random.sample(range(1, MAX_VAL + 1), 12)
    
    initial_sets = [s1, s2, s3, s4, s5]
    
    for init in initial_sets:
        if time.time() > deadline - 10:
            break
        res_set, res_score = local_search(init, deadline)
        if res_score < best_global_score:
            best_global_score = res_score
            best_global_set = res_set
            
    if best_global_set is None:
        return list(range(1, 13))
        
    return best_global_set
