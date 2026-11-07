import math
import time

def construct():
    start_time = time.time()
    deadline = start_time + 85.0
    
    N_MIN = 211
    N_MAX = 10210
    
    # Sieve primes up to N_MAX
    is_prime = [True] * (N_MAX + 1)
    is_prime[0] = is_prime[1] = False
    for i in range(2, int(math.isqrt(N_MAX)) + 1):
        if is_prime[i]:
            for j in range(i*i, N_MAX + 1, i):
                is_prime[j] = False
    
    primes = [i for i, prime in enumerate(is_prime) if prime]
    prime_to_idx = {p: i for i, p in enumerate(primes)}
    
    # Pre-compute parity bitsets for all numbers up to N_MAX
    parities = [0] * (N_MAX + 1)
    for x in range(2, N_MAX + 1):
        temp = x
        b = 0
        for p in primes:
            if p * p > temp:
                break
            if temp % p == 0:
                count = 0
                while temp % p == 0:
                    temp //= p
                    count += 1
                if count % 2 == 1:
                    b |= (1 << prime_to_idx[p])
        if temp > 1:
            if temp in prime_to_idx:
                b |= (1 << prime_to_idx[temp])
        parities[x] = b
    
    target = parities[210]
    
    # Strategy 1: Find minimal pair (a, b) with a < b, a >= N_MIN
    # such that parity(a) XOR parity(b) == target
    seen = {}  # parity -> smallest number with that parity
    for M in range(N_MIN, N_MAX + 1):
        if time.time() > deadline - 10:
            break
        v = parities[M]
        complement = v ^ target
        if complement in seen:
            a = seen[complement]
            prod = 210 * a * M
            s = math.isqrt(prod)
            if s * s == prod:
                return [a, M]
        if v not in seen:
            seen[v] = M
    
    # Strategy 2: Gaussian elimination for larger sets
    basis = {}  # pivot_bit -> (vec, mask)
    
    for M in range(N_MIN, N_MAX + 1):
        if time.time() > deadline - 3:
            break
        
        v = parities[M]
        pos = M - N_MIN
        m = 1 << pos
        
        current_v = v
        current_m = m
        
        while current_v != 0:
            pivot_bit = 1 << (current_v.bit_length() - 1)
            if pivot_bit in basis:
                bv, bm = basis[pivot_bit]
                current_v ^= bv
                current_m ^= bm
            else:
                break
        
        if current_v != 0:
            pivot_bit = 1 << (current_v.bit_length() - 1)
            basis[pivot_bit] = (current_v, current_m)
        
        w = target
        result_mask = 0
        for pivot_bit in sorted(basis.keys(), reverse=True):
            bv, bm = basis[pivot_bit]
            if w & pivot_bit:
                w ^= bv
                result_mask ^= bm
        
        if w == 0:
            selected = []
            for i in range(M - N_MIN + 1):
                if (result_mask >> i) & 1:
                    selected.append(N_MIN + i)
            
            if not selected:
                continue
            
            prod = 210
            for x in selected:
                prod *= x
            s = math.isqrt(prod)
            if s * s == prod:
                return list(selected)
    
    # Fallback
    return [840]
