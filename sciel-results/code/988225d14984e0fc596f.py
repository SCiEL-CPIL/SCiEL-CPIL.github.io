def construct():
    from fractions import Fraction
    target = Fraction(5, 31)
    best = None
    best_max = float('inf')
    
    # Search for 3-term solutions: 1/a + 1/b + 1/c = 5/31
    # Using the substitution: let x = 5a - 31, y = 5b - 31
    # Then we need x*y ≡ 31^2 (mod 31^2) for c to be an integer
    # More precisely, c = 31ab / (5ab - 31a - 31b)
    # Let's use the direct formula but with a smarter search
    
    # For 3 terms: 1/a + 1/b + 1/c = 5/31
    # We can rewrite as: (5a-31)(5b-31) must divide 31^2 * (5a-31) * (5b-31) / something
    # Actually, let's just search efficiently
    
    # Key insight: for fixed a, c = 31ab / (5ab - 31a - 31b)
    # Rearranging: c(5ab - 31a - 31b) = 31ab
    # 5abc - 31ac - 31bc = 31ab
    # Add 31^2/5 * c to both sides... let's use a different approach
    
    # From 1/a + 1/b + 1/c = 5/31:
    # (bc + ac + ab) * 31 = 5abc
    # 31bc + 31ac + 31ab = 5abc
    # Rearrange to solve for c:
    # c(5ab - 31a - 31b) = 31ab
    # c = 31ab / (5ab - 31a - 31b)
    
    # For c to be a positive integer, we need:
    # 1) 5ab - 31a - 31b > 0
    # 2) (5ab - 31a - 31b) divides 31ab
    
    # Let's search with a and b where a <= b <= c to minimize max
    # Since we want to minimize the largest denominator, and a <= b <= c,
    # we want to minimize c, which means we want a and b as large as possible
    # (but still keeping c >= b)
    
    # Try all a from 1 upward, and for each a, find valid b values
    # where c = 31ab / (5ab - 31a - 31b) is an integer >= b
    
    upper_bound = 2000  # We'll search up to this for a and b
    
    for a in range(1, upper_bound + 1):
        # For this a, find b such that c is integer and c >= b
        # c = 31ab / (5ab - 31a - 31b)
        # Let D = 5ab - 31a - 31b
        # We need D > 0 and D | 31ab
        
        # Start b from a (since we assume a <= b)
        for b in range(a, upper_bound + 1):
            D = 5 * a * b - 31 * a - 31 * b
            if D <= 0:
                continue
            num = 31 * a * b
            if num % D != 0:
                continue
            c = num // D
            if c < b:
                continue
            if c > 10000000:
                continue
            # Check distinctness
            if a == b or b == c or a == c:
                continue
            m = c  # since a <= b <= c
            if m < best_max:
                best_max = m
                best = [a, b, c]
    
    if best is not None:
        return best
    
    # Fallback: use the original greedy approach
    r, values = Fraction(5, 31) - Fraction(1, 8), [8]
    while r:
        d = (r.denominator + r.numerator - 1) // r.numerator
        values.append(d)
        r -= Fraction(1, d)
    return values
