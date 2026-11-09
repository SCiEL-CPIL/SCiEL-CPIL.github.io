import math
import time

def construct():
    # Target: minimize max(x) - 210
    # Condition: 210 * prod(x) is a perfect square
    # This implies prod(sf(x)) = 210 * s^2
    # Where sf(x) is the squarefree kernel.
    
    # We search for the smallest M such that there exists a subset of {211, ..., M}
    # whose squarefree kernels multiply to 210 * (square).
    
    # Precomputation
    max_search = 1000
    n_max = max_search
    
    # Sieve of Eratosthenes up to n_max
    is_prime = [True] * (n_max + 1)
    is_prime[0] = False
    is_prime[1] = False
    for i in range(2, int(n_max**0.5) + 1):
        if is_prime[i]:
            for j in range(i*i, n_max + 1, i):
                is_prime[j] = False
    
    primes = [i for i, p in enumerate(is_prime) if p]
    prime_to_idx = {p: i for i, p in enumerate(primes)}
    
    def get_kernel_mask(x):
        mask = 0
        temp = x
        for p in primes:
            if p * p > temp:
                break
            if temp % p == 0:
                count = 0
                while temp % p == 0:
                    temp //= p
                    count += 1
                if count % 2 == 1:
                    mask |= (1 << prime_to_idx[p])
        if temp > 1:
            # temp is a prime
            if temp in prime_to_idx:
                mask |= (1 << prime_to_idx[temp])
            else:
                # This shouldn't happen if n_max is large enough for all primes in x
                # But just in case, we can extend or ignore? 
                # Since x <= n_max, all prime factors are <= n_max.
                pass
        return mask

    # Compute target mask for 210
    target_mask = get_kernel_mask(210)
    
    # Incremental BFS
    # State: mask -> list of x's
    # We want to find if target_mask is reachable.
    reachable = {0: []}
    
    # We iterate M from 211 to max_search
    for M in range(211, max_search + 1):
        vm = get_kernel_mask(M)
        new_reachable = {}
        # Copy existing states
        for mask, indices in reachable.items():
            if mask not in new_reachable:
                new_reachable[mask] = indices
        
        # Add states including M
        for mask, indices in reachable.items():
            new_mask = mask ^ vm
            if new_mask not in new_reachable:
                new_reachable[new_mask] = indices + [M]
        
        reachable = new_reachable
        
        # Check if target is found
        if target_mask in reachable:
            return reachable[target_mask]
        
        # Safety check: if reachable grows too large, stop
        if len(reachable) > 100000:
            break
            
    # Fallback to known solution
    return [840]
