"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def construct():
    from fractions import Fraction
    
    # The known optimal 3-term Egyptian fraction for 5/31
    # 1/8 + 1/31 + 1/248 = 31/248 + 8/248 + 1/248 = 40/248 = 5/31
    candidate = [8, 31, 248]
    
    # Verification
    total = Fraction(0)
    for d in candidate:
        total += Fraction(1, d)
        
    if total == Fraction(5, 31):
        return candidate
    else:
        # Fallback to greedy if verification fails (should not happen)
        r, values = Fraction(5, 31), []
        while r:
            d = (r.denominator + r.numerator - 1) // r.numerator
            values.append(d)
            r -= Fraction(1, d)
        return values
# EVOLVE-BLOCK-END
