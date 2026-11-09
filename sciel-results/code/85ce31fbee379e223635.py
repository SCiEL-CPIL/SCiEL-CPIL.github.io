"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import math
import time
import bisect
from collections import defaultdict

def construct():
    start_time = time.time()
    DEADLINE = start_time + 120.0
    
    def time_exceeded():
        return time.time() >= DEADLINE

    def get_primes(limit):
        if limit < 2: return []
        sieve = bytearray(b'\x01') * (limit + 1)
        sieve[0] = sieve[1] = 0
        for i in range(2, int(limit**0.5) + 1):
            if sieve[i]:
                sieve[i*i::i] = bytearray(len(sieve[i*i::i]))
        return [i for i in range(2, limit + 1) if sieve[i]]

    # Generate powerful numbers up to L
    L = 10**7
    primes = get_primes(int(L**0.5) + 1)
    
    powerful = [1]
    powerful_set = set([1])
    
    # Iterative generation using a queue-like approach
    # Start with 1, and for each powerful number, multiply by p^2 for primes p
    # where p is the next prime after the largest prime factor used
    # This is tricky iteratively. Let's use a different approach:
    # Generate all numbers of the form prod p_i^e_i with e_i >= 2
    # by iterating over primes and building up.
    
    # Use a list to store powerful numbers and build incrementally
    # For each prime p, for each existing powerful number x, add x * p^2, x * p^3, ...
    # But we need to ensure we don't miss combinations.
    
    # Better: use the standard method - for each powerful number, try multiplying by p^2
    # for each prime p >= last_prime_factor. But tracking last_prime_factor is complex.
    
    # Alternative: generate all powerful numbers by checking each number? Too slow.
    
    # Let's use a recursive approach but with explicit stack to avoid recursion limit
    def generate_powerful_iterative():
        pn = [1]
        pn_set = {1}
        # For each powerful number, try to extend by multiplying by p^k
        # We need to ensure unique generation. Use the property that each powerful
        # number has a unique prime factorization with exponents >= 2.
        # Generate by: for each prime p, for each existing pn that doesn't have p as factor,
        # add pn * p^2, pn * p^3, ...
        # But "doesn't have p as factor" is hard to check quickly.
        
        # Simpler: use the DFS but with a stack
        stack = [(0, 1)]
        while stack:
            idx, val = stack.pop()
            if val > L:
                continue
            if val > 1:
                pn_set.add(val)
            
            for i in range(idx, len(primes)):
                p = primes[i]
                nv = val * p * p
                if nv > L:
                    break
                stack.append((i + 1, nv))
                while nv * p <= L:
                    nv *= p
                    pn_set.add(nv)
                    stack.append((i + 1, nv))
                    if time_exceeded():
                        return pn_set
        
        return pn_set
    
    powerful_set = generate_powerful_iterative()
    
    if time_exceeded():
        return {'N': 2, 'pairs': [[1, 1]]}
    
    pn_list = sorted(powerful_set)
    
    # Use sum-counting approach: for each pair (i,j) with i<=j, count[pn[i]+pn[j]] += 1
    # Then find the N with max count, and reconstruct pairs.
    # Limit to first K powerful numbers to keep O(K^2) manageable.
    
    # Estimate K: if L=10^7, P ~ 10^(7*2/3) = 10^(14/3) ~ 10^4.67 ~ 46000
    # K^2 = 46000^2 = 2.1e9 which is too slow.
    # So limit K to ~3000: K^2 = 9e6, fast.
    
    K = min(len(pn_list), 3000)
    small_pn = pn_list[:K]
    
    sum_count = defaultdict(int)
    sum_pairs = defaultdict(list)
    
    for i in range(K):
        a = small_pn[i]
        for j in range(i, K):
            b = small_pn[j]
            s = a + b
            if s <= 2 * L:  # Both a and b <= L, so s <= 2L
                sum_count[s] += 1
                sum_pairs[s].append([a, b])
    
    if not sum_count:
        return {'N': 2, 'pairs': [[1, 1]]}
    
    best_N = max(sum_count, key=sum_count.get)
    best_count = sum_count[best_N]
    best_pairs = sum_pairs[best_N]
    
    # Verify: all pairs should have a <= b and a+b = best_N, and a,b powerful
    # They are by construction.
    
    # Safety: ensure N >= 2
    if best_N < 2:
        return {'N': 2, 'pairs': [[1, 1]]}
    
    return {'N': best_N, 'pairs': best_pairs}

# EVOLVE-BLOCK-END
