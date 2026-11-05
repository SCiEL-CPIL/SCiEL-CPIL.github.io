def construct():
    # Greedy Egyptian fraction algorithm:
    # At each step, choose the smallest integer denominator > 4 such that
    # 1/d <= remaining fraction, and the remaining fraction after subtracting 1/d
    # is still representable with the remaining terms.
    # We use the standard greedy (Fibonacci/Sylvester) variant but enforce
    # denominator > 4.
    
    from fractions import Fraction
    
    remaining = Fraction(1, 1)
    result = []
    
    # We'll use at most 256 terms, but greedy typically gives 5-7 terms.
    max_terms = 256
    
    for _ in range(max_terms):
        if remaining == 0:
            break
        if remaining < 0:
            raise RuntimeError("negative remaining")
        
        # The greedy choice: take the smallest d > 4 such that 1/d <= remaining
        # i.e., d >= 1/remaining, and d >= 5
        # d_min = ceil(1/remaining)
        
        num, den = remaining.numerator, remaining.denominator
        # 1/remaining = den/num
        # d_min = ceil(den/num)
        d_min = (den + num - 1) // num  # ceil(den/num)
        
        # Ensure d_min >= 5
        if d_min < 5:
            d_min = 5
        
        # Check that d_min doesn't exceed the upper bound
        if d_min > 10000000:
            raise RuntimeError("denominator exceeds bound")
        
        d = d_min
        
        # Verify that 1/d <= remaining (should be true by construction)
        if Fraction(1, d) > remaining:
            # Shouldn't happen, but just in case
            d += 1
        
        # Check that d is distinct from already chosen
        if d in result:
            # This shouldn't happen with greedy, but handle it
            raise RuntimeError("duplicate denominator")
        
        result.append(d)
        remaining -= Fraction(1, d)
        
        # If remaining is zero, we're done
        if remaining == 0:
            break
    
    if remaining != 0:
        raise RuntimeError("could not represent 1 with greedy")
    
    # Verify the result
    total = Fraction(0, 1)
    for d in result:
        total += Fraction(1, d)
    if total != 1:
        raise RuntimeError("sum not equal to 1")
    
    # Verify all denominators are in [5, 10000000] and distinct
    if len(result) != len(set(result)):
        raise RuntimeError("duplicates found")
    for d in result:
        if d < 5 or d > 10000000:
            raise RuntimeError("denominator out of range")
    
    return result
