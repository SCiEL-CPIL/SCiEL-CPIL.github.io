import time
from fractions import Fraction

def construct():
    start_time = time.time()
    deadline = start_time + 50  # Safety margin
    
    R = Fraction(1, 1)
    last_d = 4
    denoms = []
    
    while R > 0:
        if time.time() > deadline:
            break
            
        # We need d such that 1/d <= R  => d >= 1/R
        # Also d > last_d and d >= 5
        # 1/R is D/N if R=N/D
        D = R.denominator
        N = R.numerator
        min_d_from_R = (D + N - 1) // N  # ceil(D/N)
        
        d = max(min_d_from_R, last_d + 1, 5)
        
        # Safety check: ensure 1/d <= R
        # If d is too small, increment
        while Fraction(1, d) > R:
            d += 1
            
        # Check upper bound
        if d > 10000000:
             # If we exceed the bound, we might need to backtrack or fail
             # But for sum=1, greedy stays small.
             # If this happens, return what we have? No, sum must be 1.
             # Just break to avoid infinite loop, but this shouldn't happen for sum=1.
             break
             
        denoms.append(d)
        R = R - Fraction(1, d)
        last_d = d
        
    if R == 0:
        return denoms
    else:
        # Fallback if something went wrong (should not happen)
        # Try a known valid small set if possible, or raise
        # A known 5-term solution: 1/5 + 1/6 + 1/7 + 1/8 + 1/9 ... wait, sum is not 1.
        # Let's just return the list if it's close? No, must be exact.
        # The greedy algorithm is guaranteed to terminate for any rational number.
        # So R should become 0.
        raise RuntimeError("Greedy algorithm failed to converge")
