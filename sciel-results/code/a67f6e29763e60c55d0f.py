import math

def construct():
    # Step 1: Generate primes p == 1 mod 4
    # We need enough primes to maximize the product <= 10^18
    # The product of the first 11 such primes is ~10^17
    # The product of the first 12 is ~10^19
    # So we need at least 12 primes to be safe, maybe 20.
    
    primes_1mod4 = []
    n = 5
    while len(primes_1mod4) < 30:
        if n % 4 == 1:
            # Check primality
            is_prime = True
            if n > 1:
                limit = int(math.isqrt(n))
                for i in range(3, limit + 1, 2):
                    if n % i == 0:
                        is_prime = False
                        break
            if is_prime:
                primes_1mod4.append(n)
        n += 2

    # Step 2: Select smallest primes to maximize count k
    # Greedy selection of smallest primes maximizes the number of factors for a fixed product limit
    N = 1
    selected_primes = []
    for p in primes_1mod4:
        if N * p <= 10**18:
            N *= p
            selected_primes.append(p)
        else:
            break
            
    if not selected_primes:
        # Fallback, though 5 <= 10^18
        return {'N': 2, 'pairs': [[1, 1]]} # 1 is powerful, 1+1=2. Pair (1,1).
        
    # Step 3: Find representation p = a^2 + b^2 for each selected prime
    prime_reps = []
    for p in selected_primes:
        found = False
        # a^2 + b^2 = p. a < b since p is not 2 mod 4 (p is 1 mod 4, odd)
        # a^2 < p/2
        limit_a = int(math.isqrt(p // 2))
        for a in range(1, limit_a + 1):
            b2 = p - a * a
            b = int(math.isqrt(b2))
            if b * b == b2:
                prime_reps.append((a, b))
                found = True
                break
        if not found:
            # This should not happen for p == 1 mod 4
            # But if it does, we cannot form N as sum of squares properly
            # However, Fermat's theorem guarantees it.
            pass

    # Step 4: Combine representations using Gaussian Integers
    # Start with 1 + 0i
    reps = [(1, 0)]
    
    for a, b in prime_reps:
        new_reps = []
        for u, v in reps:
            # (u + iv)(a + ib) = (ua - vb) + i(ub + va)
            u1 = u * a - v * b
            v1 = u * b + v * a
            new_reps.append((u1, v1))
            
            # (u + iv)(a - ib) = (ua + vb) + i(va - ub)
            u2 = u * a + v * b
            v2 = v * a - u * b
            new_reps.append((u2, v2))
        reps = new_reps
        
    # Step 5: Extract valid pairs (a_val, b_val) with a_val <= b_val
    pairs = []
    seen = set()
    for u, v in reps:
        # We need u, v > 0 for a, b to be positive powerful numbers
        # Note: 0 is not considered powerful in some contexts? 
        # "1 counts as powerful". 0 is not a positive integer.
        # The task says "Integers ... sums of two powerful numbers".
        # Usually implies positive integers.
        # If u=0, a=0. Is 0 powerful? 
        # "every prime factor appears squared". 0 has no prime factors. 
        # But usually 0 is excluded or treated specially.
        # The seed program used `if u <= 0 or v <= 0: continue`.
        # I will follow this convention.
        if u <= 0 or v <= 0:
            continue
            
        # Ensure u^2 <= v^2 => u <= v (since u,v > 0)
        if u > v:
            u, v = v, u
            
        a_val = u * u
        b_val = v * v
        
        # Check sum
        if a_val + b_val != N:
            continue
            
        pair = (a_val, b_val)
        if pair not in seen:
            seen.add(pair)
            pairs.append([a_val, b_val])
            
    return {'N': N, 'pairs': pairs}
