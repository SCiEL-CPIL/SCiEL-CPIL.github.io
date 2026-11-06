import math

def construct():
    # Precompute primes up to sqrt(256 * 10000) ~ 1600
    # Actually max z might be larger?
    # If we iterate z up to 10000, B ~ 2.5e6.
    # If we need larger z, we need larger primes.
    # Let's assume z < 10000 is enough.
    # If not, we can extend.
    
    limit = 100000
    is_prime = [True] * (limit + 1)
    is_prime[0] = is_prime[1] = False
    for i in range(2, int(math.isqrt(limit)) + 1):
        if is_prime[i]:
            for j in range(i*i, limit + 1, i):
                is_prime[j] = False
    primes = [i for i, v in enumerate(is_prime) if v]

    def factorize(num):
        factors = {}
        temp = num
        for p in primes:
            if p * p > temp:
                break
            while temp % p == 0:
                factors[p] = factors.get(p, 0) + 1
                temp //= p
        if temp > 1:
            factors[temp] = factors.get(temp, 0) + 1
        return factors

    def get_divisors_sq(factors):
        # Returns list of divisors of num^2
        # If num = p1^e1 ... pk^ek, num^2 = p1^(2e1) ...
        divs = [1]
        for p, e in factors.items():
            new_divs = []
            for d in divs:
                for i in range(2 * e + 1):
                    new_divs.append(d * (p ** i))
            divs = new_divs
        return divs

    answers = []
    for n in range(3, 257):
        found = None
        # Iterate z
        # Start z from 1
        # If not found, maybe try larger z?
        # But likely small z works.
        
        for z in range(1, 10000):
            A = 4 * z - n
            if A <= 0:
                continue
            B = n * z
            
            # Factor B
            fB = factorize(B)
            # Get divisors of B^2
            divs = get_divisors_sq(fB)
            
            for U in divs:
                if (B * B) % U != 0: # Should always be true
                    continue
                V = (B * B) // U
                if (U + B) % A != 0:
                    continue
                if (V + B) % A != 0:
                    continue
                x = (U + B) // A
                y = (V + B) // A
                if x > 0 and y > 0 and z > 0:
                    if len(set([x, y, z])) == 3:
                        if x <= 10**12 and y <= 10**12 and z <= 10**12:
                            found = [x, y, z]
                            break
            if found:
                break
        
        answers.append(found)
    return answers
