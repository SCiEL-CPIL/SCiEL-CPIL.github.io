"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from fractions import Fraction

def construct():
    start_time = time.time()
    deadline = start_time + 50.0  # Safety margin
    
    target = Fraction(5, 31)
    best_max_denom = float('inf')
    best_solution = None
    
    # Search for 3-term solutions: 1/a + 1/b + 1/c = 5/31
    # a >= 7
    for a in range(7, 1000):
        if time.time() > deadline:
            break
        r1 = target - Fraction(1, a)
        if r1 <= 0:
            break
        # Need 1/b + 1/c = r1
        # b >= ceil(1/r1)
        b_min = (r1.denominator + r1.numerator - 1) // r1.numerator
        # b must be > a to ensure distinctness and ordering? No, just distinct.
        # But if b <= a, we might have already considered it or it's symmetric.
        # We assume a <= b <= c for canonical ordering.
        # So b >= a.
        b_min = max(b_min, a + 1)
        
        # Limit b to avoid infinite loop or excessive search
        # If b is too large, c will be close to b.
        # Max b such that 1/b < r1/2 => b > 2/r1
        # We can iterate b up to a reasonable limit or until b > c
        # c = 1 / (r1 - 1/b)
        # c must be integer.
        
        # Iterate b
        # To keep it fast, we can limit b.
        # If we don't find a solution with small max, we stop.
        # Let's iterate b up to 10000 for now.
        for b in range(b_min, 10000):
            if time.time() > deadline:
                break
            r2 = r1 - Fraction(1, b)
            if r2 <= 0:
                break
            # c = 1 / r2
            if r2.denominator % r2.numerator == 0:
                c = r2.denominator // r2.numerator
                if c > b:
                    max_d = c
                    if max_d < best_max_denom:
                        best_max_denom = max_d
                        best_solution = [a, b, c]
                        # If we find a very small one, we can stop?
                        # 1736 is good. If we find < 1000, great.
                # If c == b, not distinct.
                # If c < b, skip (ordering).
    
    if best_solution:
        return best_solution
    
    # Fallback: Greedy
    r = target
    values = []
    while r:
        if time.time() > deadline:
            break
        d = (r.denominator + r.numerator - 1) // r.numerator
        values.append(d)
        r -= Fraction(1, d)
    return values
# EVOLVE-BLOCK-END
