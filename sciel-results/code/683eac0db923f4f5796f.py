"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
from fractions import Fraction

def construct():
    """
    Returns the set of denominators for the Egyptian fraction expansion of 5/31
    that minimizes the largest denominator.
    """
    target = Fraction(5, 31)
    
    best_max_denom = 10000000
    best_solution = None
    
    # Search for 3-term solutions: 1/a + 1/b + 1/c = 5/31
    # a < b < c, all distinct positive integers
    # For given a, b: c = 1 / (5/31 - 1/a - 1/b)
    
    # a must be > 31/5 = 6.2, so a >= 7
    # Also 1/a < 5/31, so a > 31/5
    # For 3 terms, 3/a >= 5/31 roughly, so a <= 3*31/5 = 18.6, a <= 18
    # But we should be more careful - a can be larger if b and c are larger
    
    # Let's search a from 7 up to some reasonable bound
    # 1/a must be less than 5/31, so a > 6.2, a >= 7
    # Also need 1/a + 1/b + 1/c = 5/31 with b > a, c > b
    # So 1/a < 5/31 - 1/a - 1/a = 5/31 - 2/a (for the remaining two terms to be smaller)
    # Actually: 5/31 - 1/a > 2/b where b > a, so 5/31 - 1/a > 2/(a+1) approximately
    # Let's just bound a reasonably: a from 7 to 200
    
    for a in range(7, 200):
        rem1 = target - Fraction(1, a)
        if rem1 <= 0:
            continue
        # Now need 1/b + 1/c = rem1 with b > a, c > b
        # 1/b < rem1, so b > 1/rem1
        # Also 1/b > rem1/2 (since 1/c < 1/b), so b < 2/rem1
        b_min = a + 1
        b_max = int(2 / rem1) + 2  # upper bound for b
        
        if b_min > b_max:
            continue
            
        for b in range(b_min, b_max + 1):
            rem2 = rem1 - Fraction(1, b)
            if rem2 <= 0:
                continue
            # c = 1/rem2
            c_num = rem2.denominator
            c_den = rem2.numerator
            if c_den != 1:
                continue
            c = c_num
            if c <= b:
                continue
            # Found a valid 3-term solution
            max_d = c
            if max_d < best_max_denom:
                best_max_denom = max_d
                best_solution = [a, b, c]
    
    # Also try 4-term solutions to see if we can beat the 3-term best
    # 1/a + 1/b + 1/c + 1/d = 5/31
    # This is more complex but let's try if the 3-term best is still high
    
    if best_max_denom > 100:  # Only bother with 4 terms if 3-term solution is bad
        for a in range(7, 50):
            rem1 = target - Fraction(1, a)
            if rem1 <= 0:
                continue
            b_min = a + 1
            b_max = int(2 / rem1) + 2
            if b_min > b_max:
                continue
            for b in range(b_min, b_max + 1):
                rem2 = rem1 - Fraction(1, b)
                if rem2 <= 0:
                    continue
                c_min = b + 1
                c_max = int(2 / rem2) + 2
                if c_min > c_max:
                    continue
                for c in range(c_min, c_max + 1):
                    rem3 = rem2 - Fraction(1, c)
                    if rem3 <= 0:
                        continue
                    d_num = rem3.denominator
                    d_den = rem3.numerator
                    if d_den != 1:
                        continue
                    d = d_num
                    if d <= c:
                        continue
                    max_d = d
                    if max_d < best_max_denom:
                        best_max_denom = max_d
                        best_solution = [a, b, c, d]
    
    # If no solution found (shouldn't happen), fall back to known solution
    if best_solution is None:
        best_solution = [8, 31, 248]
    
    return best_solution
# EVOLVE-BLOCK-END
