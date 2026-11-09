"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
from fractions import Fraction

def construct():
    """
    Greedy Egyptian fraction algorithm with forced denominator lower bounds.
    Minimizes term count by always selecting the largest possible unit fraction
    (smallest valid denominator) that does not exceed the remaining sum.
    """
    remaining = Fraction(1, 1)
    lower_bound = 5
    denominators = []
    
    while remaining > 0:
        # Find the smallest denominator d >= lower_bound such that 1/d <= remaining
        # i.e., d >= 1/remaining
        # d >= ceil(1/remaining) and d >= lower_bound
        inv_rem = Fraction(1, 1) / remaining
        min_d = max(lower_bound, int(inv_rem.numerator // inv_rem.denominator))
        if min_d * inv_rem.denominator < inv_rem.numerator:
            min_d += 1
        
        # Check if min_d works (1/min_d <= remaining)
        if Fraction(1, min_d) > remaining:
            min_d += 1
        
        # Ensure distinctness: min_d is always > previous since lower_bound increases
        denominators.append(min_d)
        remaining -= Fraction(1, min_d)
        lower_bound = min_d + 1
        
        # Safety cap: if we exceed 256 terms, we have a problem
        if len(denominators) > 256:
            raise RuntimeError("too many terms")
    
    # Verify all constraints
    assert len(denominators) >= 1 and len(denominators) <= 256
    assert all(5 <= d <= 10000000 for d in denominators)
    assert len(set(denominators)) == len(denominators)
    
    # Verify sum is exactly 1
    total = sum(Fraction(1, d) for d in denominators)
    assert total == 1, f"Sum is {total}, not 1"
    
    return denominators
# EVOLVE-BLOCK-END
