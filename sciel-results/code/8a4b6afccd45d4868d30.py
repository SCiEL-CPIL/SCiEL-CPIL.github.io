import time
import math

def construct():
    start_time = time.time()
    deadline = start_time + 85.0
    answers = []
    
    # Precompute primes up to 100000 for fast factorization
    # B = n*x where n<=256, x<=~192, so B <= 256*192 = 49152
    # But let's be safe and go up to 100000
    max_B = 256 * 200  # conservative upper bound for B
    sieve = bytearray(b'\x01') * (max_B + 1)
    sieve[0] = sieve[1] = 0
    for i in range(2, int(math.isqrt(max_B)) + 1):
        if sieve[i]:
            sieve[i*i::i] = bytearray(len(sieve[i*i::i]))
    primes = [i for i in range(2, max_B + 1) if sieve[i]]
    
    # Cache for factorizations
    factor_cache = {}
    
    def factorize(n):
        if n in factor_cache:
            return factor_cache[n]
        factors = {}
        temp = n
        for p in primes:
            if p * p > temp:
                break
            while temp % p == 0:
                factors[p] = factors.get(p, 0) + 1
                temp //= p
        if temp > 1:
            factors[temp] = factors.get(temp, 0) + 1
        factor_cache[n] = factors
        return factors

    def get_divisors_sorted(factors):
        """Generate divisors sorted for efficient range checking."""
        divs = [1]
        for p, e in factors.items():
            new_divs = []
            for d in divs:
                val = 1
                for _ in range(e + 1):
                    new_divs.append(d * val)
                    val *= p
            divs = new_divs
        divs.sort()
        return divs

    for n in range(3, 257):
        if time.time() > deadline:
            answers.append(None)
            continue
            
        found = None
        x_min = n // 4 + 1
        x_max = (3 * n) // 4
        
        for x in range(x_min, x_max + 1):
            A = 4 * x - n
            B = n * x
            
            # U must divide B^2
            # U < B
            # U > 4x^2 - 2nx
            # (U+B) % A == 0
            
            factors = factorize(B)
            factors_sq = {p: 2*e for p, e in factors.items()}
            divisors = get_divisors_sorted(factors_sq)
            
            lower_bound = 4*x*x - 2*n*x
            upper_bound = B - 1
            
            # Use binary search to narrow down the divisor range
            # Find first divisor >= lower_bound
            import bisect
            idx = bisect.bisect_left(divisors, lower_bound)
            
            for i in range(idx, len(divisors)):
                U = divisors[i]
                if U > upper_bound:
                    break
                if (U + B) % A != 0:
                    continue
                
                y = (U + B) // A
                V = (B * B) // U
                z = (V + B) // A
                
                # Verify distinctness and order
                if x < y < z:
                    if z <= 10**12:
                        found = [x, y, z]
                        break
                        
            if found:
                break
                
        answers.append(found)
        
    return answers
