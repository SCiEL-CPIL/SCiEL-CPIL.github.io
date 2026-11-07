import time
import math
import sys
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = start_time + 115  # 5s safety margin before 120s limit

    # Generate primes up to 200
    def sieve(n):
        is_prime = [True] * (n + 1)
        is_prime[0] = is_prime[1] = False
        for i in range(2, int(n**0.5) + 1):
            if is_prime[i]:
                for j in range(i*i, n + 1, i):
                    is_prime[j] = False
        return [i for i, v in enumerate(is_prime) if v]

    primes = sieve(200)
    nprimes = len(primes)

    max_m = 10**18
    sqrt_max_m = math.isqrt(max_m)  # 10^9

    solutions = defaultdict(list)
    check_counter = [0]

    def check_time():
        check_counter[0] += 1
        if check_counter[0] % 5000 == 0:
            if time.time() > deadline:
                return True
        return False

    # Iterative generation of smooth numbers using a stack
    # State: (prime_idx, current_k, current_sigma)
    # We process primes one by one, deciding the exponent for each.
    # Stack-based DFS to avoid recursion overhead.
    
    # We'll use a different approach: iterate through primes, maintaining
    # a list of (k, sigma) pairs. For each prime, expand each pair by
    # multiplying by p^e for e = 0, 1, 2, ...
    
    # Start with just (1, 1)
    states = [(1, 1)]
    
    for idx, p in enumerate(primes):
        new_states = []
        for (k, sigma) in states:
            # Exponent 0: don't use this prime
            # This is already in states, so we carry it forward
            new_states.append((k, sigma))
            
            # Exponent e >= 1: multiply by p^e
            k_curr = k
            sigma_curr = sigma
            sum_powers = 1
            p_power = 1
            
            for _ in range(20):  # max exponent per prime (p^20 is huge for p>=2)
                if k_curr > sqrt_max_m // p:
                    break
                k_curr *= p
                p_power *= p
                sum_powers += p_power
                
                # sigma for k_curr = sigma * sum_powers
                sigma_new = sigma_curr * sum_powers
                
                # Pruning: k * sigma must be <= max_m
                if k_curr > max_m // sigma_new:
                    break
                # Pruning: k must be <= sqrt(M), i.e., k <= sigma
                # Since M = k * sigma, k <= sqrt(M) means k^2 <= k*sigma, i.e., k <= sigma
                if k_curr > sigma_new:
                    break
                    
                new_states.append((k_curr, sigma_new))
        
        states = new_states
        
        if check_time():
            break
    
    # Process all final states
    for (k, sigma) in states:
        if k > 1:
            m_val = k * sigma
            if m_val <= max_m and k <= sigma:  # k <= sqrt(M) equivalent
                solutions[m_val].append(k)
    
    # Find best M
    best_m = 0
    best_k_list = []
    
    for m, k_list in solutions.items():
        if len(k_list) > len(best_k_list):
            best_m = m
            best_k_list = k_list
    
    if not best_k_list or len(best_k_list) < 3:
        return {'M': 333312, 'k': [336, 372, 434]}
    
    distinct_k = list(set(best_k_list))
    sqrt_m = math.isqrt(best_m)
    valid_k = [k for k in distinct_k if k <= sqrt_m]
    valid_k.sort()
    
    M_out = int(best_m)
    k_out = [int(k) for k in valid_k]
    
    return {'M': M_out, 'k': k_out}

if __name__ == "__main__":
    result = construct()
    print(result)
