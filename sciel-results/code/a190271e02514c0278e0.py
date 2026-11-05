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
    deadline = time.time() + 55.0 # Safety margin
    answers = []
    
    for n in range(3, 257):
        if time.time() > deadline:
            answers.append(None)
            continue
        
        found = None
        x_min = n // 4 + 1
        
        # We will iterate x. If x is too expensive, skip.
        for x in range(x_min, x_min + 200): # Limit x search range to avoid infinite loops on hard cases
            if time.time() > deadline:
                break
            
            a = 4 * x - n
            if a <= 0:
                continue
            b = n * x
            
            # Estimate cost: number of divisors of b^2
            # Factor b
            b_fac = _factorize(b)
            cnt_b = Counter(b_fac)
            
            # Calculate number of divisors of b^2
            div_count = 1
            for p, exp in cnt_b.items():
                div_count *= (2 * exp + 1)
                
            # Heuristic: If div_count is too high, skip this x to save time for others
            # 10000 divisors is roughly 1-5ms to generate and check.
            # If we have 0.2s per n, we can afford ~20-50 such x's.
            if div_count > 5000:
                continue
                
            # Generate divisors of b^2
            # We only need divisors d such that (d+b) % a == 0
            # and (b^2//d + b) % a == 0
            
            # Generate divisors of b first, then square? No, generate divisors of b^2 directly.
            # b^2 prime factors: p^(2*exp)
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
            
            # Check divisors
            # Optimization: We can check (d+b)%a==0 early
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
                # Check if d divides b_sq (it does by construction)
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
