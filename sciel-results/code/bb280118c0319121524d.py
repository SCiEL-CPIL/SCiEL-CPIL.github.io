"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def construct():
    import time
    from fractions import Fraction
    from math import gcd
    
    deadline = time.time() + 80.0  # safety margin before 90s kill
    target = Fraction(5, 31)
    
    best_solution = None
    best_max_denom = 10000001  # upper bound
    
    # Strategy: search for solutions with increasing largest denominator c
    # For each c, we need 1/a + 1/b = target - 1/c
    # Let r = target - 1/c = p/q (in lowest terms)
    # Then 1/a + 1/b = p/q => q(a+b) = p*a*b
    # => p*a*b - q*a - q*b = 0
    # => (p*a - q)(p*b - q) = q^2
    # So for each divisor d of q^2, we get a = (d+q)/p, b = (q^2/d + q)/p
    # We need both a and b to be positive integers, distinct from each other and from c,
    # and all in [1, 10000000].
    
    # First, try to find a 3-term solution by iterating c from small values upward.
    # We want to minimize the largest denominator, so we iterate c and check if
    # a valid (a,b) pair exists with a,b < c (to make c the largest).
    
    # But we also need to consider that the largest denominator might not be c.
    # Actually, if we fix c as the largest, then a < c and b < c.
    # Let's iterate c from 1 to some limit, and for each c, check if there exist
    # distinct a, b < c such that 1/a + 1/b + 1/c = 5/31.
    
    # Actually, let's be smarter. Let's just iterate c and find all valid (a,b) pairs.
    # We'll track the minimum largest denominator across all found solutions.
    
    # Let's start with c from small values. The largest denominator in the solution
    # will be max(a,b,c). We want to minimize this.
    
    # For efficiency, let's iterate c from 1 upward, and for each c, solve for (a,b).
    # We'll stop when c exceeds our current best_max_denom.
    
    # But first, let's try a few small c values to find an initial solution.
    
    def find_pairs_for_c(c):
        """Find all distinct pairs (a,b) with a < b < c such that 1/a + 1/b = target - 1/c."""
        r = target - Fraction(1, c)
        if r <= 0:
            return []
        
        p = r.numerator
        q = r.denominator
        
        # (p*a - q)(p*b - q) = q^2
        # Let d be a divisor of q^2. Then:
        # p*a - q = d  => a = (d + q) / p
        # p*b - q = q^2 / d  => b = (q^2 / d + q) / p
        
        # We need a and b to be positive integers.
        # Also a < b < c, and a, b distinct.
        
        q2 = q * q
        
        # Find all divisors of q^2
        # q can be large, so this might be slow. Let's factorize q first.
        # Actually, let's just iterate d from 1 to sqrt(q^2) = q, and check if d divides q^2.
        
        pairs = []
        d = 1
        while d * d <= q2:
            if q2 % d == 0:
                # d is a divisor
                # Case 1: p*a - q = d, p*b - q = q2/d
                a_num = d + q
                b_num = q2 // d + q
                
                if a_num % p == 0 and b_num % p == 0:
                    a = a_num // p
                    b = b_num // p
                    if 1 <= a < b < c and gcd(a, b) == 1:  # gcd check not needed, just distinct
                        if a != b:
                            pairs.append((a, b))
                
                # Case 2: p*a - q = q2/d, p*b - q = d (if d != q2/d)
                if d * d != q2:
                    a_num = q2 // d + q
                    b_num = d + q
                    if a_num % p == 0 and b_num % p == 0:
                        a = a_num // p
                        b = b_num // p
                        if 1 <= a < b < c and a != b:
                            if (a, b) not in pairs:
                                pairs.append((a, b))
            d += 1
        
        return pairs
    
    # Iterate c from 1 upward
    c = 1
    while c < best_max_denom and time.time() < deadline:
        pairs = find_pairs_for_c(c)
        for (a, b) in pairs:
            # Check all denominators are in [1, 10000000]
            if 1 <= a <= 10000000 and 1 <= b <= 10000000 and 1 <= c <= 10000000:
                max_denom = max(a, b, c)
                if max_denom < best_max_denom:
                    best_max_denom = max_denom
                    best_solution = [a, b, c]
        
        c += 1
        
        # Periodic check: if we've found a solution, we can stop when c >= best_max_denom
        if best_solution and c >= best_max_denom:
            break
    
    if best_solution:
        return best_solution
    
    # Fallback: use the greedy method from the seed
    r, values = target - Fraction(1, 8), [8]
    while r and time.time() < deadline:
        d = (r.denominator + r.numerator - 1) // r.numerator
        values.append(d)
        r -= Fraction(1, d)
    
    return values
# EVOLVE-BLOCK-END
