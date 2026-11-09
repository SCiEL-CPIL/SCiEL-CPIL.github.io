import time
import math

def _factorize(n):
    """Return a list of (prime, exponent) for n using trial division."""
    factors = []
    if n <= 1:
        return factors
    # Check small primes
    for p in [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97]:
        if p * p > n:
            break
        if n % p == 0:
            exp = 0
            while n % p == 0:
                n //= p
                exp += 1
            factors.append((p, exp))
    if n > 1:
        factors.append((n, 1))
    return factors

def _divisors_from_factors(factors):
    """Generate all divisors from a list of (prime, exponent)."""
    divs = [1]
    for p, e in factors:
        current_divs = []
        pe = 1
        for _ in range(e + 1):
            for d in divs:
                current_divs.append(d * pe)
            pe *= p
        divs = current_divs
    return divs

def _find_solution_for_n(n):
    """Find distinct x,y,z with 1/x+1/y+1/z=4/n, x<y<z, all <= 10^12."""
    # Try x values starting from n+1 (small offset from n/4)
    # x must satisfy: 4x > n (i.e., x > n/4) and 3/x >= 4/n (i.e., x <= 3n/4)
    x_min = n // 4 + 1
    x_max = (3 * n) // 4
    
    if x_max < x_min:
        return None
    
    # Build a priority list of x values to try:
    # Start with x close to n+1 (small a = 4x-n), then expand
    # Actually, let's try x = n+1, n+2, ... up to x_max, but also try x_min, x_min+1, ...
    # The "early success" insight suggests x near n works well
    
    # Generate candidate x values in order of likelihood
    # x = n+1 gives a = 3n+4, which is moderate
    # Let's try a few strategic x values first
    
    candidates = []
    
    # Strategy 1: x = n+1, n+2, n+3, ... (small positive offset from n)
    for k in range(1, min(20, x_max - n + 1)):
        x_val = n + k
        if x_min <= x_val <= x_max:
            candidates.append(x_val)
    
    # Strategy 2: x = x_min, x_min+1, ... (from the lower bound)
    for x_val in range(x_min, min(x_min + 20, x_max + 1)):
        if x_val not in candidates:
            candidates.append(x_val)
    
    # Strategy 3: remaining x values
    for x_val in range(x_min, x_max + 1):
        if x_val not in candidates:
            candidates.append(x_val)
    
    for x in candidates:
        a = 4 * x - n
        b = n * x
        if a <= 0 or b <= 0:
            continue
        
        g = math.gcd(a, b)
        a_red = a // g
        b_red = b // g
        
        b_red_sq = b_red * b_red
        
        # Factorize b_red to get factors of b_red_sq
        factors_b = _factorize(b_red)
        factors_sq = [(p, 2 * e) for p, e in factors_b]
        
        divs = _divisors_from_factors(factors_sq)
        
        for d in divs:
            num_y = b_red + d
            if num_y % a_red != 0:
                continue
            y = num_y // a_red
            
            if y <= x:
                continue
            
            num_z = b_red + b_red_sq // d
            if num_z % a_red != 0:
                continue
            z = num_z // a_red
            
            if z <= y:
                continue
            
            if y > 10**12 or z > 10**12:
                continue
            
            # Verify via cross multiplication
            lhs_num = n * y * z + n * x * z + n * x * y
            rhs_num = 4 * x * y * z
            if lhs_num == rhs_num:
                return [x, y, z]
    
    return None

def construct():
    deadline = time.time() + 85
    answers = []
    
    for n in range(3, 257):
        if time.time() > deadline:
            answers.append(None)
            continue
        
        found = _find_solution_for_n(n)
        answers.append(found)
    
    return answers
