import time
from math import gcd
from collections import Counter

def _factorize(n):
    """Simple trial division factorization."""
    factors = []
    d = 2
    while d * d <= n:
        while n % d == 0:
            factors.append(d)
            n //= d
        d += 1
    if n > 1:
        factors.append(n)
    return factors

def construct():
    deadline = time.time() + 55.0  # Safety margin
    answers = []
    
    for n in range(3, 257):
        if time.time() > deadline:
            answers.append(None)
            continue
        
        found = None
        x_min = n // 4 + 1
        x_max = (4 * n) // 3  # Upper bound for x
        
        # We will iterate x. Try smaller x first as they're more likely to work
        for x in range(x_min, min(x_max + 1, x_min + 500)):
            if time.time() > deadline:
                break
            
            a = 4 * x - n
            if a <= 0:
                continue
            b = n * x
            
            # Estimate cost: number of divisors of b^2
            b_fac = _factorize(b)
            cnt_b = Counter(b_fac)
            
            # Calculate number of divisors of b^2
            div_count = 1
            for p, exp in cnt_b.items():
                div_count *= (2 * exp + 1)
                
            # Heuristic: If div_count is too high, skip this x to save time
            if div_count > 10000:
                continue
                
            # Generate divisors of b^2
            divisors_b2 = [1]
            for p, exp in cnt_b.items():
                new_divs = []
                p_pow = 1
                for e in range(2 * exp + 1):
                    for d in divisors_b2:
                        new_divs.append(d * p_pow)
                    p_pow *= p
                divisors_b2 = new_divs
                
            b_sq = b * b
            
            # Check divisors - look for d such that (d+b) % a == 0
            for d in divisors_b2:
                if time.time() > deadline:
                    break
                if (d + b) % a != 0:
                    continue
                y_val = (d + b) // a
                if y_val <= x:
                    continue
                if y_val > 10**12:
                    continue
                rem = b_sq // d
                if (rem + b) % a != 0:
                    continue
                z_val = (rem + b) // a
                if z_val <= y_val:
                    continue
                if z_val > 10**12:
                    continue
                if x != y_val and y_val != z_val and x != z_val:
                    found = [x, y_val, z_val]
                    break
            
            if found:
                break
        
        answers.append(found)
    
    return answers
