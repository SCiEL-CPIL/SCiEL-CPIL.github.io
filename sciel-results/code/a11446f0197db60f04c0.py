import time
import math

DENOM_BOUND = 10**12

def construct():
    deadline = time.time() + 85.0
    answers = [None] * 254
    
    # Precompute primes up to 10000 for factorization
    def get_primes(n):
        sieve = [True] * (n + 1)
        sieve[0] = sieve[1] = False
        for i in range(2, int(math.isqrt(n)) + 1):
            if sieve[i]:
                for j in range(i*i, n + 1, i):
                    sieve[j] = False
        return [i for i, is_prime in enumerate(sieve) if is_prime]
    
    PRIMES = get_primes(10000)
    
    def factorize(num, primes):
        factors = {}
        for p in primes:
            if p * p > num:
                break
            if num % p == 0:
                count = 0
                while num % p == 0:
                    num //= p
                    count += 1
                factors[p] = count
        if num > 1:
            factors[num] = 1
        return factors

    def generate_divisors(factors):
        divisors = [1]
        for p, e in factors.items():
            new_divs = []
            p_power = 1
            for _ in range(e + 1):
                for d in divisors:
                    new_divs.append(d * p_power)
                p_power *= p
            divisors = new_divs
        return divisors

    # Cache for factorizations of B values
    factor_cache = {}
    
    def get_factorized_B(B):
        if B in factor_cache:
            return factor_cache[B]
        factors_B = factorize(B, PRIMES)
        factors_B2 = {p: 2*e for p, e in factors_B.items()}
        factor_cache[B] = factors_B2
        return factors_B2

    for i, n in enumerate(range(3, 257)):
        if time.time() > deadline:
            break
            
        found = None
        nx = n
        
        # x must be > n/4 and <= 3n/4
        x_min = n // 4 + 1
        x_max = (3 * n) // 4
        
        # Try x values in order of increasing A = 4x - n
        # Smaller A tends to find solutions faster
        x_values = list(range(x_min, x_max + 1))
        x_values.sort(key=lambda x: 4*x - n)
        
        for x in x_values:
            if time.time() > deadline:
                break
                
            # A = 4x - n, B = nx
            A = 4 * x - n
            if A <= 0:
                continue
                
            B = nx * x
            
            # We need 1/y + 1/z = A/B
            # (Ay - B)(Az - B) = B^2
            # U = Ay - B, V = Az - B, UV = B^2
            # y = (U+B)/A, z = (V+B)/A
            
            # Simplify: let g = gcd(A, B)
            # A = g*A', B = g*B'
            # Then (g*A'*y - g*B')(g*A'*z - g*B') = g^2*B'^2
            # g^2*(A'*y - B')(A'*z - B') = g^2*B'^2
            # (A'*y - B')(A'*z - B') = B'^2
            # Let U' = A'*y - B', V' = A'*z - B'
            # U'V' = B'^2
            # y = (U' + B')/A', z = (V' + B')/A'
            
            g = math.gcd(A, B)
            A_prime = A // g
            B_prime = B // g
            
            # Factorize B_prime^2
            factors_B2 = get_factorized_B(B_prime)
            
            # Generate divisors of B_prime^2
            divs = generate_divisors(factors_B2)
            
            # U' must satisfy:
            # y >= x => (U' + B')/A' >= x => U' >= A'*x - B'
            # y <= z => U' <= V' => U' <= B' (since U'V' = B'^2)
            # U' > 0
            
            lower_U = A_prime * x - B_prime
            if lower_U < 1:
                lower_U = 1
            
            upper_U = B_prime
            
            # Filter and sort divisors in range
            divs_sorted = sorted([d for d in divs if lower_U <= d <= upper_U])
            
            # Iterate divisors U'
            for U in divs_sorted:
                # Check integrality of y
                if (U + B_prime) % A_prime != 0:
                    continue
                
                y = (U + B_prime) // A_prime
                
                # Check distinctness with x
                if y == x:
                    continue
                
                # Calculate V'
                V = (B_prime * B_prime) // U
                
                # Check integrality of z
                if (V + B_prime) % A_prime != 0:
                    continue
                
                z = (V + B_prime) // A_prime
                
                # Check order y <= z
                if y > z:
                    continue
                
                # Check distinctness y != z
                if y == z:
                    continue
                
                # Check bounds
                if y > DENOM_BOUND or z > DENOM_BOUND:
                    continue
                
                # Check positive
                if y < 1 or z < 1:
                    continue
                
                found = [x, y, z]
                break
            
            if found:
                break
                
        answers[i] = found
        
        # Clear cache periodically to avoid memory issues
        if len(factor_cache) > 5000:
            factor_cache.clear()
    
    return answers
