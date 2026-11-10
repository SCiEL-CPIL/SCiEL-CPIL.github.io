"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def construct():
    from fractions import Fraction
    
    # The known optimal solution for 5/31 with minimum largest denominator
    # is {8, 31, 248}.
    # 1/8 + 1/31 + 1/248 = 31/248 + 8/248 + 1/248 = 40/248 = 5/31
    solution = [8, 31, 248]
    
    # Verify the solution
    total = sum(Fraction(1, d) for d in solution)
    if total != Fraction(5, 31):
        # Fallback to a valid, though likely suboptimal, solution
        # Greedy expansion starting with 8
        r = Fraction(5, 31) - Fraction(1, 8)
        values = [8]
        while r:
            d = (r.denominator + r.numerator - 1) // r.numerator
            values.append(d)
            r -= Fraction(1, d)
        return values
        
    return solution
# EVOLVE-BLOCK-END
