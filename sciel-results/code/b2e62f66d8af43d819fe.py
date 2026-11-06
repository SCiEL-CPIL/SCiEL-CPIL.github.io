import time
import math

DENOM_BOUND = 10**12

def construct():
    deadline = time.time() + 85.0
    answers = [None] * 254  # index 0 -> n=3, ..., index 253 -> n=256
    
    # Precompute primes up to a reasonable limit for fast factorization
    # Max x might be larger, but let's start with a sieve up to 5000
    # If x is larger, we can fall back to trial division or extend sieve
    LIMIT = 5000
    spf = list(range(LIMIT + 1))
    for i in range(2, int(math.isqrt(LIMIT)) + 1):
        if spf[i] == i:  # i is prime
            for j in range(i * i, LIMIT + 1, i):
                if spf[j] == j:
                    spf[j] = i
    
    def factorize(num):
        """Return list of (prime, exponent) pairs for num."""
        if num <= 1:
            return []
        factors = []
        temp = num
        while temp > 1:
            if temp <= LIMIT:
                p = spf[temp]
            else:
                # Find smallest prime factor by trial division
                p = 2
                while p * p <= temp and temp % p != 0:
                    p += 1
                if p * p > temp:
                    p = temp
            count = 0
            while temp % p == 0:
                temp //= p
                count += 1
            factors.append((p, count))
        return factors

    def get_divisors_from_factors(factors):
        """Generate all divisors from a list of (prime, exponent) pairs."""
        divisors = [1]
        for p, e in factors:
            new_divs = []
            power = 1
            for i in range(e + 1):
                for d in divisors:
                    new_divs.append(d * power)
                power *= p
            divisors = new_divs
        return divisors

    for i, n in enumerate(range(3, 257)):
        if time.time() > deadline:
            break
        
        found = None
        x_min = n // 4 + 1
        
        # We iterate x upwards. In most cases, a solution is found with small x.
        # To be safe, we can iterate up to some limit or until time runs out.
        # However, if a solution exists, it's likely with x not too large.
        # Let's iterate x from x_min to x_min + 10000, or until found.
        # Actually, the algebraic method is exact for a given x.
        # If no solution for small x, we might need larger x.
        # But typically, the smallest x yields a solution.
        
        for x in range(x_min, x_min + 20000):
            if time.time() > deadline:
                break
                
            A = 4 * x - n
            if A <= 0:
                continue
            B = n * x
            
            # Factor B
            factors_B = factorize(B)
            
            # Factors of B^2 are (p, 2*e) for (p, e) in factors_B
            factors_B2 = [(p, 2 * e) for p, e in factors_B]
            
            # Get divisors of B^2
            # This can be expensive if B has many factors.
            # But B = n*x, and n, x are smallish.
            divisors_B2 = get_divisors_from_factors(factors_B2)
            
            # Sort divisors to check smaller d first? 
            # d < B. y = (B+d)/A. Smaller d -> smaller y.
            # We want y > x.
            # Let's filter divisors d < B.
            valid_ds = [d for d in divisors_B2 if d < B]
            
            for d in valid_ds:
                # Check divisibility
                if (B + d) % A != 0:
                    continue
                y = (B + d) // A
                if y <= x:
                    continue
                
                # Check z
                # z = (B + B^2//d) / A
                # B^2 // d is an integer since d | B^2
                B2_over_d = (B * B) // d
                if (B + B2_over_d) % A != 0:
                    continue
                z = (B + B2_over_d) // A
                
                if z <= y:
                    continue
                if z > DENOM_BOUND:
                    continue
                
                # Verify distinctness and equation
                if x != y and y != z and x != z:
                    # Cross multiplication check
                    # 1/x + 1/y + 1/z = 4/n
                    # n(yz + xz + xy) = 4xyz
                    if n * (y * z + x * z + x * y) == 4 * x * y * z:
                        found = [x, y, z]
                        break
            
            if found:
                break
        
        answers[i] = found
    
    return answers
