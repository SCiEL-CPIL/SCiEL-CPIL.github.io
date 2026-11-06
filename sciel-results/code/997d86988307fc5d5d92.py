import math

def construct():
    # Step 1: Find N <= 10^18 composed of primes 1 mod 4 that maximizes number of divisors
    # Primes 1 mod 4: 5, 13, 17, 29, 37, 41, 53, 61, 73, 89, 97, 101, 109, 113, 137, ...
    primes_1mod4 = [5, 13, 17, 29, 37, 41, 53, 61, 73, 89, 97, 101, 109, 113, 137, 149, 157, 173, 181, 193]
    
    best_N = 1
    best_divs = 1
    best_exponents = []
    
    # Recursive search for exponents
    # We want to maximize prod(e_i + 1) subject to prod(p_i^e_i) <= 10^18
    # Exponents should be non-increasing for optimal distribution (usually)
    # But since primes are small, we can just search.
    
    def search(idx, current_N, current_divs, exponents, max_exp):
        nonlocal best_N, best_divs, best_exponents
        if current_divs > best_divs:
            best_divs = current_divs
            best_N = current_N
            best_exponents = exponents[:]
        
        if idx >= len(primes_1mod4):
            return
            
        p = primes_1mod4[idx]
        
        # Try exponents from max_exp down to 0
        # We can limit max_exp based on current_N
        # p^e <= 10^18 / current_N
        max_e_possible = int(math.log(10**18 / current_N, p)) if current_N > 0 else 100
        if max_e_possible > max_exp:
            max_e_possible = max_exp
            
        for e in range(max_e_possible, -1, -1):
            next_N = current_N * (p ** e)
            if next_N > 10**18:
                continue
            next_divs = current_divs * (e + 1)
            exponents.append(e)
            search(idx + 1, next_N, next_divs, exponents, e)
            exponents.pop()

    # Initial call
    # Start with max exponent for first prime
    # 5^25 ~ 2.9e17, 5^26 > 1e18
    search(0, 1, 1, [], 60) # 60 is a safe upper bound for exponents
    
    # Step 2: Construct N from best_exponents
    N = 1
    for i, e in enumerate(best_exponents):
        if e > 0:
            N *= (primes_1mod4[i] ** e)
            
    # If N is 1, it has no representations with a,b >= 1.
    # But our search should find a good N.
    # Minimum best_N found will be 5^something.
    
    # Step 3: Find representations x^2 + y^2 = N
    # We need to factor N into primes p_i and find u_i, v_i such that p_i = u_i^2 + v_i^2
    # Then generate all sums of two squares.
    
    def find_sum_two_squares(p):
        # p is prime 1 mod 4
        u = 1
        while u * u < p:
            v2 = p - u * u
            v = int(math.isqrt(v2))
            if v * v == v2:
                return (u, v)
            u += 1
        return (1, 1) # Should not happen for 1 mod 4 primes
    
    # Get prime factorization of N (we know it)
    prime_factors = []
    for i, e in enumerate(best_exponents):
        if e > 0:
            p = primes_1mod4[i]
            u, v = find_sum_two_squares(p)
            prime_factors.append((p, e, u, v))
            
    # Generate all representations using Gaussian integers (as tuples (re, im))
    # Start with [ (1, 0) ]
    reps = [(1, 0)]
    
    for p, e, u, v in prime_factors:
        new_reps = []
        # Precompute powers of (u, v) and (u, -v)
        # z = (u+iv)^k * (u-iv)^(e-k)
        # We can iterate k from 0 to e
        
        # Calculate (u+iv)^j and (u-iv)^j
        pow_pos = [(1, 0)]
        pow_neg = [(1, 0)]
        for j in range(1, e + 1):
            # (a,b) * (u,v) = (au-bv, av+bu)
            a, b = pow_pos[-1]
            pow_pos.append((a*u - b*v, a*v + b*u))
            
            a, b = pow_neg[-1]
            # (u, -v)
            pow_neg.append((a*u - b*(-v), a*(-v) + b*u))
            
        for k in range(e + 1):
            # Term is pow_pos[k] * pow_neg[e-k]
            a1, b1 = pow_pos[k]
            a2, b2 = pow_neg[e - k]
            # Multiply (a1, b1) and (a2, b2)
            ca, cb = (a1*a2 - b1*b2, a1*b2 + b1*a2)
            
            # Multiply all existing reps by (ca, cb)
            for rx, ry in reps:
                # (rx, ry) * (ca, cb)
                nx = rx*ca - ry*cb
                ny = rx*cb + ry*ca
                new_reps.append((nx, ny))
        
        reps = new_reps
        
    # Step 4: Extract pairs (a, b) from reps
    # Each rep (x, y) gives a solution x^2 + y^2 = N
    # a = x^2, b = y^2
    # We need a, b >= 1, a <= b, distinct pairs
    
    pairs_set = set()
    for x, y in reps:
        if x == 0 or y == 0:
            continue
        a = x * x
        b = y * y
        if a > b:
            a, b = b, a
        # a and b are powerful (squares)
        # a + b = N
        pairs_set.add((a, b))
        
    # Convert to list of lists
    pairs = [[a, b] for a, b in pairs_set]
    
    return {'N': N, 'pairs': pairs}

# EVOLVE-BLOCK-END
