import math
import time
import heapq

def construct():
    """
    Returns an integer n <= 10^18 maximizing the number of adjacent
    coprime pairs in its divisor sequence.
    """
    start_time = time.time()
    TIME_LIMIT = 140  # Safety margin below 150s
    
    max_val = 10**18
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97]
    
    candidates = []
    
    # DFS to find numbers with high divisor counts (tau)
    # We store (tau, n)
    def dfs(idx, max_exp, current_n, current_tau):
        if time.time() - start_time > TIME_LIMIT:
            return
        
        if current_n > 1:
            candidates.append((current_tau, current_n))
            
        if idx >= len(primes):
            return
            
        p = primes[idx]
        limit_n = max_val // current_n
        
        e_max_possible = 0
        temp = 1
        while temp * p <= limit_n and e_max_possible < max_exp:
            temp *= p
            e_max_possible += 1
            
        for e in range(1, e_max_possible + 1):
            next_n = current_n * (p ** e)
            next_tau = current_tau * (e + 1)
            dfs(idx + 1, e, next_n, next_tau)
            
            if time.time() - start_time > TIME_LIMIT:
                return

    # Start DFS
    dfs(0, 60, 1, 1)
    
    # Also add the square-free number with maximum primes
    sf_n = 1
    for p in primes:
        if sf_n * p <= max_val:
            sf_n *= p
        else:
            break
    candidates.append((sf_n.bit_count(), sf_n))  # tau = 2^k where k is number of distinct primes
    
    if not candidates:
        return 2 * 3 * 5 * 7 * 11 * 13 * 17 * 19
        
    # Sort by tau descending
    candidates.sort(key=lambda x: x[0], reverse=True)
    
    best_n = candidates[0][1]
    best_score = -1
    
    def tau_perp_fast(n):
        """
        Compute tau_perp(n) more efficiently by generating divisors in sorted order.
        For n = p1^a1 * p2^a2 * ... * pk^ak, we can generate divisors using
        a heap-based approach or by iterative multiplication.
        """
        # Factorize n
        factors = []
        temp = n
        for p in primes:
            if p * p > temp:
                break
            if temp % p == 0:
                count = 0
                while temp % p == 0:
                    temp //= p
                    count += 1
                factors.append((p, count))
        if temp > 1:
            factors.append((temp, 1))
        
        if not factors:
            return 0
        
        # Generate all divisors using iterative approach
        divs = [1]
        for p, a in factors:
            new_divs = []
            for d in divs:
                val = d
                for e in range(a + 1):
                    new_divs.append(val)
                    val *= p
            divs = new_divs
        
        divs.sort()
        
        score = 0
        for i in range(len(divs) - 1):
            if math.gcd(divs[i], divs[i+1]) == 1:
                score += 1
        return score

    # Evaluate top candidates
    eval_count = min(len(candidates), 500)
    
    for tau, n in candidates[:eval_count]:
        if time.time() - start_time > TIME_LIMIT - 10:
            break
        score = tau_perp_fast(n)
        if score > best_score:
            best_score = score
            best_n = n
            
    return best_n
