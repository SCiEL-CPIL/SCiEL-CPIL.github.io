# EVOLVE-BLOCK-START
def construct():
    from fractions import Fraction
    
    # Target
    target = Fraction(5, 31)
    
    # 1. Check for 3-term solutions (current best known: 248)
    best_solution = None
    best_max = 10000001
    
    for a in range(7, 500):
        p = 5 * a - 31
        q = 31 * a
        if p <= 0:
            continue
        
        # 1/b + 1/c = p/q
        # (p*b - q)(p*c - q) = q*q
        q2 = q * q
        
        # Iterate divisors of q2
        # To be fast, iterate b
        for b in range(a + 1, 5000):
            denom_val = p * b - q
            if denom_val <= 0:
                continue
            
            numerator_val = q * b
            
            if numerator_val % denom_val != 0:
                continue
            
            c = numerator_val // denom_val
            
            if c <= b:
                continue
            
            if c < best_max:
                best_max = c
                best_solution = [a, b, c]
                
    # 2. Check for 4-term solutions with small denominators (max < best_max)
    # Search denominators from 10 to best_max
    # 1/d1 + 1/d2 + 1/d3 + 1/d4 = 5/31
    # d1 < d2 < d3 < d4
    
    # Pruning: d1 >= 10 (since 1/9 + 1/10 + 1/11 + 1/12 > 5/31? 
    # 1/9=0.11, 1/10=0.1, 1/11=0.09, 1/12=0.08. Sum=0.38. Too big.
    # We need sum 0.16.
    # 1/15+1/16+1/17+1/18 approx 0.066+0.0625+0.058+0.055 = 0.24.
    # 1/20+1/21+1/22+1/23 approx 0.18.
    # 1/25+1/26+1/27+1/28 approx 0.15.
    # So d1 should be around 20-30.
    
    # We'll search d1 from 10 to 100, d2 from d1+1 to 100, etc.
    # Limit to 4 terms to keep search space small.
    
    limit = min(best_max, 100)
    
    for d1 in range(10, limit):
        r1 = target - Fraction(1, d1)
        if r1 <= 0:
            continue
        for d2 in range(d1 + 1, limit):
            r2 = r1 - Fraction(1, d2)
            if r2 <= 0:
                continue
            for d3 in range(d2 + 1, limit):
                r3 = r2 - Fraction(1, d3)
                if r3 <= 0:
                    continue
                # 1/d4 = r3
                # d4 = 1/r3
                if r3.denominator % r3.numerator == 0:
                    d4 = r3.denominator // r3.numerator
                    if d4 > d3 and d4 < best_max:
                        best_max = d4
                        best_solution = [d1, d2, d3, d4]
                        
    if best_solution is not None:
        return best_solution
    return [8, 31, 248]
# EVOLVE-BLOCK-END
