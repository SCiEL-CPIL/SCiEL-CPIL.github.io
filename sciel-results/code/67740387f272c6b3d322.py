"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import math

def _get_prime_factors(n):
    factors = {}
    d = 2
    while d * d <= n:
        while (n % d) == 0:
            factors[d] = factors.get(d, 0) + 1
            n //= d
        d += 1
    if n > 1:
        factors[n] = factors.get(n, 0) + 1
    return factors

def _get_divisors_from_factors(factors):
    divisors = [1]
    for p, e in factors.items():
        current_divisors = []
        power = 1
        for _ in range(e + 1):
            for d in divisors:
                current_divisors.append(d * power)
            power *= p
        divisors = current_divisors
    return divisors

def construct():
    answers = []
    for n in range(3, 257):
        found = None
        
        # Case 1: n is divisible by 4
        if n % 4 == 0:
            x = n // 4
            y = n // 2
            z = n
            if x != y and y != z and x != z and x > 0 and y > 0 and z > 0:
                if x <= 10**12 and y <= 10**12 and z <= 10**12:
                    found = [x, y, z]
        
        # Case 2: General search for other n
        if found is None:
            start_x = n // 4 + 1
            # Limit search to avoid infinite loops, though solutions are dense
            for i in range(100):
                x = start_x + i
                A = 4 * x - n
                if A <= 0:
                    continue
                B = n * x
                # We need divisors of B^2
                b_factors = _get_prime_factors(B)
                # Factors of B^2 have exponents doubled
                b2_factors = {p: 2 * e for p, e in b_factors.items()}
                divs = _get_divisors_from_factors(b2_factors)
                
                for u in divs:
                    # Check divisibility condition for y
                    if (u + B) % A != 0:
                        continue
                    y = (u + B) // A
                    v = (B * B) // u
                    # Check divisibility condition for z
                    if (v + B) % A != 0:
                        continue
                    z = (v + B) // A
                    
                    # Check constraints: positive, distinct, within bound
                    if y > 0 and z > 0 and x > 0:
                        if y != z and y != x and z != x:
                            if y <= 10**12 and z <= 10**12 and x <= 10**12:
                                found = [x, y, z]
                                break
                if found:
                    break
        
        answers.append(found)
        
    return answers
# EVOLVE-BLOCK-END
