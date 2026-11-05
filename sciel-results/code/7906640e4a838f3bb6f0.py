import time
import math

def factorize(n):
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

def get_divisors(factors):
    divisors = [1]
    for p, e in factors.items():
        new_divs = []
        for d in divisors:
            for k in range(e + 1):
                new_divs.append(d * (p ** k))
        divisors = new_divs
    return divisors

def solve_for_x(n, x):
    """Given fixed x, find y,z such that 1/x+1/y+1/z = 4/n with all distinct."""
    a = 4 * x - n
    if a <= 0:
        return None
    b = n * x
    
    # We need: 1/y + 1/z = a/b
    # => b(y+z) = a*y*z
    # => a*y*z - b*y - b*z = 0
    # => (a*y - b)(a*z - b) = b^2 + a*b... wait let me redo
    # Actually: a*y*z - b*y - b*z = 0
    # Multiply by a: a^2*y*z - a*b*y - a*b*z = 0
    # Add b^2: (a*y - b)(a*z - b) = b^2
    # So u = a*y - b, v = a*z - b, u*v = b^2
    # y = (u + b)/a, z = (v + b)/a
    # Need u|b^2, u ≡ -b (mod a), and both y,z positive integers
    
    # Alternative: use the quadratic approach
    # From 1/y + 1/z = a/b:
    # b(y+z) = a*y*z
    # a*y*z - b*y - b*z = 0
    # (2y - b/a)(2z - b/a) = (b/a)^2 + 4b/a... this gets messy with fractions
    
    # Let's use: y*z - (b/a)(y+z) = 0
    # Multiply through: a*y*z - b*y - b*z = 0
    # Add b^2/a: (a*y - b)(a*z - b) = b^2... wait that's not right either
    
    # Let me be careful:
    # 1/y + 1/z = a/b
    # (y+z)/(yz) = a/b
    # b(y+z) = a*yz
    # a*yz - b*y - b*z = 0
    # To complete the product: multiply by a:
    # a^2*yz - a*b*y - a*b*z = 0
    # Add b^2: (a*y - b)(a*z - b) = b^2
    # Yes! So u = a*y - b, v = a*z - b, and u*v = b^2
    # y = (u+b)/a, z = (v+b)/a
    # Conditions: u | b^2, (u+b) % a == 0, u > 0 (so y > 0), v > 0 (so z > 0)
    # Also need y != x, z != x, y != z, all <= 10^12
    
    # Factor b^2
    factors_b = factorize(b)
    factors_b2 = {p: 2*e for p, e in factors_b.items()}
    divs = get_divisors(factors_b2)
    
    target_mod = (-b) % a
    
    for u in divs:
        if u % a != target_mod:
            continue
        y = (u + b) // a
        v = b * b // u
        z = (v + b) // a
        
        if y < 1 or z < 1:
            continue
        if x == y or x == z or y == z:
            continue
        if y > 10**12 or z > 10**12:
            continue
        
        return [x, y, z]
    
    return None

def construct():
    deadline = time.time() + 50.0
    answers = []
    
    for n in range(3, 257):
        found = None
        x_min = n // 4 + 1
        
        # Try a range of x values, starting from x_min
        for x in range(x_min, x_min + 50):
            if time.time() > deadline:
                break
            
            result = solve_for_x(n, x)
            if result is not None:
                found = result
                break
        
        if time.time() > deadline:
            while len(answers) < 254:
                answers.append(None)
            return answers
        
        answers.append(found)
        
    return answers
