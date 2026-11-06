import time
import math

def construct():
    start_time = time.time()
    deadline = start_time + 110

    # Primes 1 mod 4
    primes = [5, 13, 17, 29, 37, 41, 53, 61, 73, 89]
    
    # Exponents: 5^5, others ^1
    exponents = [5, 1, 1, 1, 1, 1, 1, 1, 1, 1]
    
    # Calculate M
    M = 1
    for p, e in zip(primes, exponents):
        M *= p ** e
        
    if M > 10**18:
        # Fallback to smaller M
        return {'N': 2, 'pairs': [[1, 1]]}
        
    N = M
    
    # Find sum of two squares for each prime
    # p = a^2 + b^2
    prime_factors_gaussian = []
    for p in primes:
        # Find a, b such that a^2 + b^2 = p
        found = False
        for a in range(1, int(math.isqrt(p)) + 1):
            b2 = p - a*a
            b = int(math.isqrt(b2))
            if b*b == b2 and b > 0:
                prime_factors_gaussian.append((a, b))
                found = True
                break
        if not found:
            # Should not happen for primes 1 mod 4
            return {'N': 2, 'pairs': [[1, 1]]}
            
    # Generate all combinations of Gaussian factors
    # For p^e, the factors are (a+bi)^k (a-bi)^(e-k) for k=0..e
    # We represent Gaussian integer as (real, imag)
    
    def multiply_gauss(z1, z2):
        r1, i1 = z1
        r2, i2 = z2
        return (r1*r2 - i1*i2, r1*i2 + i1*r2)
        
    def power_gauss(z, n):
        result = (1, 0)
        for _ in range(n):
            result = multiply_gauss(result, z)
        return result
        
    factor_lists = []
    for i, p in enumerate(primes):
        e = exponents[i]
        a, b = prime_factors_gaussian[i]
        pi = (a, b)
        pi_bar = (a, -b)
        
        factors = []
        for k in range(e + 1):
            # z = pi^k * pi_bar^(e-k)
            z1 = power_gauss(pi, k)
            z2 = power_gauss(pi_bar, e - k)
            z = multiply_gauss(z1, z2)
            factors.append(z)
            
        factor_lists.append(factors)
        
    # Generate all products
    pairs_set = set()
    
    # Recursive generation or iterative product
    current_products = [(1, 0)]
    for factors in factor_lists:
        next_products = []
        for cp in current_products:
            for f in factors:
                next_products.append(multiply_gauss(cp, f))
        current_products = next_products
        
    for z in current_products:
        x, y = z
        x = abs(x)
        y = abs(y)
        if x == 0 or y == 0:
            continue
        if x > y:
            x, y = y, x
        # Check if x^2 + y^2 == M
        if x*x + y*y == M:
            pairs_set.add((x, y))
            
    pairs = [[x, y] for x, y in pairs_set]
    pairs.sort(key=lambda p: p[0])
    
    return {'N': N, 'pairs': pairs}
