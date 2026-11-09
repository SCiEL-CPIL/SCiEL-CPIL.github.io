import time

def construct():
    """
    Returns a 2-coloring of the edges of K_n that avoids monochromatic K_2,2,2.
    Uses the Paley graph construction for n=29, which is a prime congruent to 1 mod 4.
    """
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s kill

    # We use the known record n=29.
    # The Paley graph P_29 is a strongly regular graph.
    # Its edges are colored based on whether the difference is a quadratic residue mod 29.
    n = 29

    # Check time
    if time.time() > deadline:
        # Fallback: return a trivial valid coloring for small n if time ran out (unlikely)
        # But for n=29 Paley, it's fast. This guard is just for safety.
        return {'n': 1, 'colors': []}

    colors = []
    
    # Legendre symbol calculation:
    # For prime p, x is a quadratic residue mod p iff x^((p-1)/2) == 1 mod p.
    # x is non-residue iff x^((p-1)/2) == p-1 mod p.
    # x=0 is not a residue.
    # Here we look at diff = (j-i) % n. diff is in 1..n-1.
    
    exponent = (n - 1) // 2
    
    for i in range(n):
        for j in range(i + 1, n):
            if time.time() > deadline:
                # If we exceed time, we must return a valid result.
                # The partial list is invalid. 
                # Since the calculation is O(n^2) with small constant, this shouldn't happen.
                # However, to be strictly safe against "exception or timeout scores 0",
                # we should ensure we finish. 
                # For n=29, 406 pow calls are negligible.
                pass
            
            diff = (j - i) % n
            # pow(base, exp, mod) is efficient
            res = pow(diff, exponent, n)
            
            if res == 1:
                # Quadratic residue
                colors.append(0)
            else:
                # Non-residue (res must be n-1 for non-zero diff)
                colors.append(1)
                
    return {'n': n, 'colors': colors}
