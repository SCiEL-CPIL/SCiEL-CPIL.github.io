import math
import time

def construct():
    start_time = time.time()
    deadline = start_time + 85.0
    answers = []
    
    # Precompute primes up to 1000 for factorization of V'
    # V' max is around 256 * 256 / gcd ~ 65000
    # sqrt(65000) ~ 255
    limit = 500
    is_prime = [True] * (limit + 1)
    is_prime[0] = is_prime[1] = False
    for i in range(2, int(limit**0.5) + 1):
        if is_prime[i]:
            for j in range(i*i, limit + 1, i):
                is_prime[j] = False
    primes = [i for i in range(2, limit + 1) if is_prime[i]]
    
    def factorize(num):
        factors = {}
        d = 2
        while d * d <= num:
            if num % d == 0:
                count = 0
                while num % d == 0:
                    num //= d
                    count += 1
                factors[d] = count
            d += 1 if d == 2 else 2
        if num > 1:
            factors[num] = 1
        return factors

    def get_divisors(factors):
        divs = [1]
        for p, e in factors.items():
            current_divs = []
            power = 1
            for _ in range(e + 1):
                for d in divs:
                    current_divs.append(d * power)
                power *= p
            divs = current_divs
        return divs

    for n in range(3, 257):
        if time.time() > deadline:
            answers.append(None)
            continue
        
        found = None
        # x range: n/4 < x < 3n/4
        x_min = n // 4 + 1
        x_max = (3 * n) // 4
        
        for x in range(x_min, x_max + 1):
            U = 4 * x - n
            V = n * x
            g = math.gcd(U, V)
            U_prime = U // g
            V_prime = V // g
            
            D = V_prime * V_prime
            sqrt_D = int(math.isqrt(D))
            
            # Factorize V_prime to get divisors of D = V_prime^2
            # If V_prime is large, factorization is fast.
            # Divisors of D are squares of divisors of V_prime? No.
            # If V_prime = prod p^e, D = prod p^(2e).
            # We need divisors X of D.
            
            # Optimization: We can just iterate potential X?
            # No, divisor generation is better.
            
            # However, for very small U_prime, the modulo condition is weak.
            # For large U_prime, it's strong.
            
            # Let's generate divisors of V_prime, then square them?
            # No, divisors of V_prime^2 are not just squares of divisors of V_prime.
            # Example: V'=6=2*3. V'^2=36=2^2*3^2. Divisors: 1,2,4,3,6,12,9,18,36.
            # Divisors of V': 1,2,3,6. Squares: 1,4,9,36. Missing 2,3,6,12,18.
            
            # So we must factorize V_prime and generate divisors of V_prime^2.
            
            factors_vp = factorize(V_prime)
            # Exponents in D are 2*e
            factors_D = {p: 2*e for p, e in factors_vp.items()}
            
            divs = get_divisors(factors_D)
            
            # Sort divisors to find smallest y (smallest X)
            # We need X < sqrt_D
            # And X % U_prime == (-V_prime) % U_prime
            
            target_mod = (-V_prime) % U_prime
            
            # To find the smallest y, we want the smallest X.
            # So we iterate divisors in increasing order.
            
            # Optimization: If U_prime is 1, target_mod is 0.
            # Any divisor X works (mod 1 is 0).
            
            # We can prune divisors > sqrt_D.
            
            # Note: get_divisors returns unsorted list.
            # Sorting O(K log K) where K is number of divisors.
            # K is small.
            
            divs.sort()
            
            for X in divs:
                if X >= sqrt_D:
                    break
                if X % U_prime != target_mod:
                    continue
                
                # Check if y is integer
                # y = (X + V_prime) / U_prime
                # Since X % U_prime == -V_prime % U_prime, X+V_prime is divisible by U_prime.
                y = (X + V_prime) // U_prime
                
                if y <= x:
                    continue
                
                # Calculate z
                Y = D // X
                z = (Y + V_prime) // U_prime
                
                # Check z > y (guaranteed by X < sqrt_D)
                # Check z <= 10^12
                if z <= 10**12:
                    found = [x, y, z]
                    break
            
            if found:
                break
        
        answers.append(found)
        
    return answers
