import math
import time

def construct():
    start_time = time.time()
    deadline = start_time + 80.0
    
    MAX_VAL = 10210
    
    # Sieve for primes up to MAX_VAL
    is_prime = [True] * (MAX_VAL + 1)
    is_prime[0] = is_prime[1] = False
    for i in range(2, int(MAX_VAL ** 0.5) + 1):
        if is_prime[i]:
            for j in range(i * i, MAX_VAL + 1, i):
                is_prime[j] = False
                
    primes = [i for i, val in enumerate(is_prime) if val]
    prime_to_idx = {p: i for i, p in enumerate(primes)}
    
    def compute_vec(n):
        vec = 0
        x = n
        for p in primes:
            if p * p > x:
                break
            if x % p == 0:
                cnt = 0
                while x % p == 0:
                    x //= p
                    cnt += 1
                if cnt % 2 == 1:
                    vec |= (1 << prime_to_idx[p])
        if x > 1:
            vec |= (1 << prime_to_idx[x])
        return vec
    
    # Precompute vectors for 210 and 211..MAX_VAL
    vec_210 = compute_vec(210)
    vecs = {}
    for i in range(211, MAX_VAL + 1):
        vecs[i] = compute_vec(i)
        
    # basis_map: pivot_idx -> (vector, subset)
    # subset is a set of integers
    basis_map = {}
    
    def add_to_basis(vec, num):
        u = vec
        S = {num}
        while u != 0:
            h = u.bit_length() - 1
            if h in basis_map:
                b, S_b = basis_map[h]
                u ^= b
                S = S.symmetric_difference(S_b)
            else:
                basis_map[h] = (u, S)
                break
                
    def check_span(target_vec):
        t = target_vec
        S_t = set()
        while t != 0:
            h = t.bit_length() - 1
            if h in basis_map:
                b, S_b = basis_map[h]
                t ^= b
                S_t = S_t.symmetric_difference(S_b)
            else:
                return False, None
        return True, S_t
        
    for M in range(211, MAX_VAL + 1):
        if time.time() > deadline:
            break
        add_to_basis(vecs[M], M)
        found, subset = check_span(vec_210)
        if found:
            # Verify the solution
            prod = 210
            for x in subset:
                prod *= x
            k = math.isqrt(prod)
            if k * k == prod:
                # Ensure distinct and in range
                if len(subset) == len(set(subset)) and all(210 < x <= MAX_VAL for x in subset):
                    return [int(x) for x in subset]
                    
    # Fallback (should not happen)
    return [840]
