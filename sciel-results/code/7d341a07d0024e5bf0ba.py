import time
import math
import heapq

def construct():
    start_time = time.time()
    deadline = start_time + 110.0
    
    # Primes to consider
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31]
    K_MAX = 10**9
    
    # We want to find M with many k.
    # M = k * sigma(k)
    # We generate k, compute sigma(k), then M.
    
    # To avoid duplicates and ensure we cover all combinations,
    # we can use a recursive generation or a heap.
    # A heap of (k, next_prime_index, sigma_k) is tricky because sigma is not multiplicative in a simple "step" way without knowing factors.
    # However, if we generate k in increasing order, we can factorize it if we store its "last prime" and "power".
    
    # Simpler: Recursive DFS.
    # state: (current_k, current_sigma, current_prime_index, current_power_of_last_prime, last_prime)
    # Actually, we just need to iterate through primes.
    
    # Let's use a list to store (M, k) pairs? No, M can be huge.
    # We store M -> count and M -> list of k (only for the best one? No, we don't know the best one yet).
    # Storing lists for all M is memory heavy if there are many M.
    # But number of k is ~1.5 million. Number of unique M is likely similar.
    # 1.5 million entries in a dict is fine.
    
    # To save memory, we can first count, then do a second pass to collect k for the best M.
    # Or, we can store the list of k for each M, but limit the size? No, we need all.
    # Let's store M -> list of k.
    
    solutions = {}
    
    # Optimization: Precompute sigma for prime powers?
    # sigma(p^a) = 1 + p + ... + p^a.
    
    def generate(k, sigma_k, idx):
        if time.time() > deadline - 5.0:
            return
        
        if idx == len(primes):
            # Process k
            if k > 1: # k=1 gives M=1, usually not interesting for max count, but valid.
                M = k * sigma_k
                if M <= 10**18:
                    if M in solutions:
                        solutions[M].append(k)
                    else:
                        solutions[M] = [k]
            return
        
        p = primes[idx]
        # Try powers of p
        current_k = k
        current_sigma = sigma_k
        # sigma(p^0) = 1
        # sigma(p^1) = 1+p
        # sigma(p^a) = sigma(p^(a-1)) * p + 1? No.
        # sigma(p^a) = (p^(a+1)-1)/(p-1).
        # Let's just multiply.
        # Start with power 0 (factor 1)
        generate(current_k, current_sigma, idx + 1)
        
        power_val = p
        sigma_power_val = 1 + p
        current_k *= power_val
        current_sigma = sigma_k * sigma_power_val
        
        while current_k <= K_MAX:
            # Check if M exceeds limit early?
            # M = current_k * current_sigma
            # We can't easily prune here without knowing future primes.
            # But if current_k * current_sigma > 10^18, and future primes > 1, it will only grow.
            # So if current_k * current_sigma > 10^18, we can stop increasing power of p?
            # No, we can still add 1s for other primes (power 0).
            # But if we stop, we miss solutions where this prime has this power.
            # However, if current_k * current_sigma > 10^18, then any extension (multiplying by >1) will exceed 10^18.
            # So we can break the loop if current_k * current_sigma > 10^18?
            # Wait, we are in the loop for prime p.
            # If we break, we don't recurse for higher powers of p.
            # But we also don't recurse for the current power with next primes if it's already too big?
            # The recursive call `generate` handles the next primes.
            # So if `current_k * current_sigma > 10^18`, we should still call `generate`?
            # No, if the current partial product exceeds 10^18, any completion will exceed it.
            # So we can skip the `generate` call and break the loop?
            # Yes.
            
            if current_k * current_sigma > 10**18:
                break
            
            generate(current_k, current_sigma, idx + 1)
            
            # Next power
            if current_k > K_MAX // p:
                break
            current_k *= p
            # Update sigma
            # sigma(p^(a+1)) = sigma(p^a) * p + 1
            # Let's verify:
            # a=1: 1+p.
            # a=2: 1+p+p^2.
            # (1+p)*p + 1 = p + p^2 + 1. Correct.
            sigma_power_val = sigma_power_val * p + 1
            current_sigma = sigma_k * sigma_power_val

    # Start with k=1, sigma=1, idx=0
    generate(1, 1, 0)
    
    if not solutions:
        return {'M': 333312, 'k': [336, 372, 434]}
    
    # Find M with max count
    best_M = max(solutions, key=lambda m: len(solutions[m]))
    best_k_list = solutions[best_M]
    
    # Check constraints
    # k <= isqrt(M)
    # k * sigma(k) == M (already ensured)
    # distinct k (ensured by generation)
    
    # Filter k <= isqrt(M) just in case (though M = k sigma(k) > k^2 implies k < sqrt(M))
    limit = math.isqrt(best_M)
    best_k_list = [k for k in best_k_list if k <= limit]
    
    if not best_k_list:
        return {'M': 333312, 'k': [336, 372, 434]}
        
    return {'M': best_M, 'k': best_k_list}
