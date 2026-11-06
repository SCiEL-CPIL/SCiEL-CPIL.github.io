import numpy as np

def coefficients(stages):
    # We use 6 A-steps and 6 B-steps (12 total)
    # Symmetric: a[i] = a[11-i], b[i] = b[11-i]
    # Unknowns: a0..a5, b0..b5
    # Conditions:
    # 1. sum(a) = 1, sum(b) = 1
    # 2. sum(i*a) = sum(i*b) = 3.5 (centered index 0..5, center 2.5? No, 0..5 center is 2.5)
    #    Actually, for symmetry, we just need the moments to match.
    # 3. 4th order conditions involve commutators.
    #    For symmetric splitting, 3rd order vanishes.
    #    4th order condition: sum(a_i^2) == sum(b_i^2) == 1/3?
    #    This is for specific structures.
    
    # Let's use a known 4th-order symmetric scheme with 6 steps (3 A, 3 B) and double it?
    # No, let's just use Strang with 12 substeps if we can map it.
    # Strang $B A B$ maps to $B A B A(0)$. 4 stages.
    # 12 stages / 4 = 3 substeps. Error 1/27. Bad.
    
    # What if we use $B A B A$ as one 4th-order step?
    # Is there a 4th-order symmetric 4-step (2 A, 2 B) scheme?
    # No, 4th order requires at least 5 steps for symmetric.
    
    # Let's use 4th-order 5-step scheme.
    # $B A B A B$. 5 steps.
    # Map to our form $B A B A B A B A B A B A$ (12 steps).
    # We can do 2 such steps? $5+5=10$. 2 left over.
    # Or 1 step with 12 stages?
    
    # Let's use the 4th-order 5-step coefficients:
    # c1 = 1/(2*(1-2**(-1/3)))
    # c2 = 1/(2*(1+2**(-1/3)))
    # c3 = 1 - 2*c1
    # Steps: B(c1), A(c2), B(c3), A(c2), B(c1)
    # Map to 12 stages:
    # k=1: b=c1, a=0
    # k=2: b=0, a=c2
    # k=3: b=c3, a=0
    # k=4: b=0, a=c2
    # k=5: b=c1, a=0
    # k=6..12: 0
    
    # This is 4th order.
    # Error O(t^5).
    # Seed Error O(t^3/216).
    # For t=1, 4th order error is ~0.01, Seed is ~0.05?
    # 4th order is much better.
    
    c1 = 1.0 / (2.0 * (1.0 - 2.0**(-1.0/3.0)))
    c2 = 1.0 / (2.0 * (1.0 + 2.0**(-1.0/3.0)))
    c3 = 1.0 - 2.0 * c1
    
    a = [0.0] * stages
    b = [0.0] * stages
    
    # Sequence: B A B A B
    b[0] = c1
    a[1] = c2
    b[2] = c3
    a[3] = c2
    b[4] = c1
    
    return a, b
