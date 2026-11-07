import time
import math

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin for 120s limit
    
    max_m = 10**18
    max_k = 10**9  # since k^2 <= M <= 10^18
    
    # Use more primes to increase the search space and collision probability
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]
    
    solutions = {}  # M -> list of k values
    
    # Precompute for each prime the list of (p_pow, term) pairs
    # where term = p^e * sigma(p^e)
    prime_data = []
    for p in primes:
        data = []
        p_pow = 1
        sig = 1
        while True:
            term = p_pow * sig
            if term > max_m or p_pow > max_k:
                break
            data.append((p_pow, term))
            p_pow *= p
            sig += p_pow
        prime_data.append(data)
    
    def gen(idx, current_k, current_m):
        if time.time() > deadline:
            return
        
        if idx == len(primes):
            if 1 < current_m <= max_m:
                if current_m not in solutions:
                    solutions[current_m] = []
                solutions[current_m].append(current_k)
            return
        
        for (p_pow, term) in prime_data[idx]:
            next_k = current_k * p_pow
            if next_k > max_k:
                break
            next_m = current_m * term
            if next_m > max_m:
                break
            
            gen(idx + 1, next_k, next_m)
            
            if time.time() > deadline:
                return
    
    gen(0, 1, 1)
    
    # Find best M
    best_m = 0
    best_ks = []
    max_count = 0
    
    for m, ks in solutions.items():
        if len(ks) > max_count:
            max_count = len(ks)
            best_m = m
            best_ks = ks
    
    if max_count == 0 or best_m == 0:
        return {'M': 333312, 'k': [336, 372, 434]}
    
    # Ensure correct types
    best_m = int(best_m)
    best_ks = [int(k) for k in best_ks]
    
    return {'M': best_m, 'k': best_ks}
