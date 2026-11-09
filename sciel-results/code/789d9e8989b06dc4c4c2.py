"""Rewritten program using a greedy Egyptian fraction approach with forced lower bounds."""
# EVOLVE-BLOCK-START
from fractions import Fraction

def construct():
    """
    Return a list of 1..256 distinct integer denominators in [5, 10000000]
    whose reciprocal sum is exactly 1, minimizing the count.
    
    Uses a greedy algorithm with forced denominator lower bounds to iteratively
    select the largest possible unit fraction (smallest valid denominator) that
    does not exceed the remaining sum, ensuring distinct denominators.
    """
    remaining = Fraction(1, 1)
    denominators = []
    min_denom = 5  # Denominators must be >= 5
    
    while remaining > 0:
        if remaining == 0:
            break
            
        # Find the smallest denominator d >= min_denom such that 1/d <= remaining
        # This means d >= 1/remaining
        # We want the smallest such d, which is ceil(1/remaining)
        
        # But we also need to ensure d > all previously chosen denominators (distinct)
        # Since we're picking greedily with increasing denominators, min_denom handles this
        
        # The greedy choice: pick d = max(min_denom, ceil(1/remaining))
        # But we need to be careful: 1/d must be <= remaining, so d >= 1/remaining
        
        # Compute the minimal valid denominator
        # d must satisfy: d >= min_denom AND 1/d <= remaining AND d is integer
        # 1/d <= remaining  =>  d >= 1/remaining
        
        # Let's compute the lower bound for d
        # If remaining is a Fraction p/q, then 1/remaining = q/p
        # So d >= q/p, meaning d >= ceil(q/p)
        
        p = remaining.numerator
        q = remaining.denominator
        
        # d must be at least ceil(q/p)
        # ceil(q/p) = (q + p - 1) // p
        lower_bound = (q + p - 1) // p
        
        d = max(min_denom, lower_bound)
        
        # Ensure d is within the valid range [5, 10000000]
        if d > 10000000:
            # This shouldn't happen if we have a valid solution, but handle it
            raise RuntimeError(f"Denominator {d} exceeds maximum allowed value")
        
        # Check that 1/d <= remaining
        if Fraction(1, d) > remaining:
            # This can happen due to ceiling issues, increment d
            d += 1
            while Fraction(1, d) > remaining:
                d += 1
        
        # Add this denominator
        denominators.append(d)
        remaining -= Fraction(1, d)
        
        # Update the minimum denominator for the next iteration
        # Since we need distinct denominators and we're going in increasing order,
        # the next denominator must be > d
        min_denom = d + 1
        
        # Safety check: if we have too many terms
        if len(denominators) > 256:
            raise RuntimeError("Too many terms in Egyptian fraction")
        
        # If remaining becomes 0, we're done
        if remaining == 0:
            break
    
    if remaining != 0:
        raise RuntimeError(f"Failed to reach exactly 1, remaining: {remaining}")
    
    # Verify the result
    total = sum(Fraction(1, d) for d in denominators)
    if total != 1:
        raise RuntimeError(f"Verification failed: sum is {total}, expected 1")
    
    # Verify all constraints
    for d in denominators:
        if d < 5 or d > 10000000:
            raise RuntimeError(f"Denominator {d} out of range [5, 10000000]")
    
    if len(set(denominators)) != len(denominators):
        raise RuntimeError("Denominators are not distinct")
    
    return denominators
# EVOLVE-BLOCK-END
