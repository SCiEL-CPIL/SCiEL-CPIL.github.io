import math
import time

def _factorize(n):
    """Return dict of prime factors and exponents for n."""
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

def _divisors_from_factors(factors):
    """Generate all divisors from a prime factorization dict."""
    divisors = [1]
    for p, e in factors.items():
        new_divs = []
        p_powers = [1]
        for _ in range(e):
            p_powers.append(p_powers[-1] * p)
        for d in divisors:
            for pp in p_powers:
                new_divs.append(d * pp)
        divisors = new_divs
    return divisors

def construct():
    deadline = time.time() + 85
    answers = []
    
    for n in range(3, 257):
        if time.time() > deadline:
            answers.append(None)
            continue
        
        found = None
        min_x = n // 4 + 1
        
        # Strategy: try several x values, using factorization-based divisor search
        # First, try a small set of promising x values
        x_candidates = []
        # Always include min_x
        x_candidates.append(min_x)
        # Try a few more values
        for offset in range(1, 100):
            x_candidates.append(min_x + offset)
        
        for x in x_candidates:
            if time.time() > deadline:
                break
            if found:
                break
            
            a = 4 * x - n
            if a <= 0:
                continue
            b = n * x
            
            g = math.gcd(a, b)
            a_red = a // g
            b_red = b // g
            
            # Solve 1/y + 1/z = a_red / b_red
            # (a_red*y - b_red)(a_red*z - b_red) = b_red^2
            # Let u = a_red*y - b_red, v = a_red*z - b_red
            # u*v = b_red^2
            # y = (u + b_red) / a_red, z = (v + b_red) / a_red
            # Need u, v positive divisors of b_red^2 with (u+b_red) % a_red == 0
            
            b_sq = b_red * b_red
            
            # Factor b_red, then generate divisors of b_red^2
            # b_red^2 has factors: for each p^e in b_red, p^(2e) in b_red^2
            factors_b = _factorize(b_red)
            factors_b_sq = {p: 2*e for p, e in factors_b.items()}
            
            # Generate divisors of b_red^2
            divs = _divisors_from_factors(factors_b_sq)
            
            for u in divs:
                if time.time() > deadline:
                    break
                if u > b_sq:
                    break
                
                v = b_sq // u
                if u > v:
                    continue
                
                if (u + b_red) % a_red != 0:
                    continue
                y = (u + b_red) // a_red
                
                if (v + b_red) % a_red != 0:
                    continue
                z = (v + b_red) // a_red
                
                if y <= 0 or z <= 0:
                    continue
                
                if x == y or x == z or y == z:
                    continue
                if y > 10**12 or z > 10**12:
                    continue
                
                # Verify
                lhs = n * (y * z + x * z + x * y)
                rhs = 4 * x * y * z
                if lhs == rhs:
                    if x > 0 and y > 0 and z > 0 and x <= 10**12 and y <= 10**12 and z <= 10**12:
                        found = [x, y, z]
                        break
            
            if found:
                break
        
        # If factorization approach didn't work, fall back to k-scan for a few x values
        if not found:
            for x in x_candidates[:50]:
                if time.time() > deadline:
                    break
                if found:
                    break
                
                a = 4 * x - n
                if a <= 0:
                    continue
                b = n * x
                
                g = math.gcd(a, b)
                a_red = a // g
                b_red = b // g
                
                b_sq = b_red * b_red
                
                k_min = b_red // a_red + 1
                k_max = min((b_sq + b_red) // a_red + 1, k_min + 2000)
                
                for k in range(k_min, k_max + 1):
                    if time.time() > deadline:
                        break
                    if found:
                        break
                    
                    u = k * a_red - b_red
                    if u <= 0:
                        continue
                    if u > b_sq:
                        break
                    
                    if b_sq % u != 0:
                        continue
                    
                    v = b_sq // u
                    if u > v:
                        continue
                    
                    if (u + b_red) % a_red != 0:
                        continue
                    y = (u + b_red) // a_red
                    
                    if (v + b_red) % a_red != 0:
                        continue
                    z = (v + b_red) // a_red
                    
                    if y <= 0 or z <= 0:
                        continue
                    
                    if x == y or x == z or y == z:
                        continue
                    if y > 10**12 or z > 10**12:
                        continue
                    
                    lhs = n * (y * z + x * z + x * y)
                    rhs = 4 * x * y * z
                    if lhs == rhs:
                        if x > 0 and y > 0 and z > 0 and x <= 10**12 and y <= 10**12 and z <= 10**12:
                            found = [x, y, z]
                            break
        
        answers.append(found)
    
    return answers
