import time
from fractions import Fraction
from math import gcd
from sympy import factorint

def construct():
    target = Fraction(5, 31)
    start_time = time.time()
    time_limit = 55.0
    
    best_solution = [8, 28, 1736]
    best_max_denom = 1736
    
    def verify(solution):
        if not solution:
            return False
        if len(solution) < 1 or len(solution) > 256:
            return False
        if len(set(solution)) != len(solution):
            return False
        if any(d < 1 or d > 10000000 for d in solution):
            return False
        s = sum(Fraction(1, d) for d in solution)
        return s == target
    
    # Search 3-term solutions: 1/a + 1/b + 1/c = 5/31, a < b < c
    # For each a >= 7, solve 1/b + 1/c = (5a-31)/(31a)
    # Use (p*b - q)*(p*c - q) = q^2 where p=5a-31, q=31a
    
    for a in range(7, 100):
        if time.time() - start_time > time_limit:
            break
        
        p = 5 * a - 31
        q = 31 * a
        
        if p <= 0:
            continue
        
        # Reduce p/q
        g = gcd(p, q)
        p_r = p // g
        q_r = q // g
        
        # (p_r * b - q_r) * (p_r * c - q_r) = q_r^2
        # Let d1 = p_r * b - q_r, d2 = p_r * c - q_r
        # d1 * d2 = q_r^2
        # d1 > 0, d2 > 0, d1 < d2 (since b < c)
        # b = (d1 + q_r) / p_r must be integer
        # c = (d2 + q_r) / p_r must be integer
        # b > a, c > b
        
        q_r_sq = q_r * q_r
        
        # Find all divisors of q_r^2
        # For efficiency, iterate d1 up to sqrt(q_r^2) = q_r
        # d1 must divide q_r^2
        # d1 < d2 => d1 < q_r (since d1*d2 = q_r^2 and d1 < d2 means d1 < q_r)
        
        # d1 must satisfy: d1 | q_r^2 and (d1 + q_r) % p_r == 0
        # Also d1 > 0, and b = (d1 + q_r)/p_r > a
        
        # Iterate through possible d1 values
        # d1 must be a divisor of q_r^2
        # Let's iterate d1 from 1 to q_r (since d1 < d2 and d1*d2 = q_r^2)
        
        for d1 in range(1, q_r + 1):
            if time.time() - start_time > time_limit:
                break
            
            if q_r_sq % d1 != 0:
                continue
            
            d2 = q_r_sq // d1
            
            # Check b and c are integers
            if (d1 + q_r) % p_r != 0:
                continue
            if (d2 + q_r) % p_r != 0:
                continue
            
            b = (d1 + q_r) // p_r
            c = (d2 + q_r) // p_r
            
            if b <= a or c <= b:
                continue
            
            if c < best_max_denom:
                best_solution = [a, b, c]
                best_max_denom = c
    
    if verify(best_solution):
        return best_solution
    
    return [8, 28, 1736]
