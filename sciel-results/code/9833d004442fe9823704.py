"""Optimized search for 1/x+1/y+1/z=4/n using divisor enumeration."""
import time

def _factorize(n):
    factors = {}
    d = 2
    while d * d <= n:
        while n % d == 0:
            factors[d] = factors.get(d, 0) + 1
            n //= d
        d += 1
    if n > 1:
        factors[n] = factors.get(n, 0) + 1
    return factors

def _get_divisors_of_square(factors):
    divs = [1]
    for p, e in factors.items():
        max_e = 2 * e
        current_divs = []
        for d in divs:
            p_power = 1
            for _ in range(max_e + 1):
                current_divs.append(d * p_power)
                p_power *= p
        divs = current_divs
    return divs

def construct():
    answers = []
    deadline = time.time() + 55  # safety margin
    for n in range(3, 257):
        if time.time() > deadline:
            answers.append(None)
            continue
        found = None
        start_x = n // 4 + 1
        end_x = (3 * n) // 4
        if end_x < start_x:
            end_x = start_x
        
        for x in range(start_x, end_x + 1):
            if time.time() > deadline:
                break
            
            A = 4 * x - n
            if A <= 0:
                continue
            B = n * x
            B_sq = B * B
            
            # Factorize B
            b_factors = _factorize(B)
            
            # Get divisors of B^2
            divs = _get_divisors_of_square(b_factors)
            
            for u in divs:
                if (u + B) % A != 0:
                    continue
                y = (u + B) // A
                if y <= 0:
                    continue
                
                v = B_sq // u
                if (v + B) % A != 0:
                    continue
                z = (v + B) // A
                if z <= 0:
                    continue
                
                # Check distinctness
                if x == y or x == z or y == z:
                    continue
                
                # Check bounds
                if y > 10**12 or z > 10**12:
                    continue
                
                found = [x, y, z]
                break
            
            if found:
                break
        
        answers.append(found)
    return answers
