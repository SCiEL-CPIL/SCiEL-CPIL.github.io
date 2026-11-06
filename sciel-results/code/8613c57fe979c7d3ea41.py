"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def construct():
    from fractions import Fraction
    import math

    target = Fraction(5, 31)
    best_solution = None
    best_max_denom = float('inf')

    # Try to find a 3-term solution: 1/a + 1/b + 1/c = 5/31
    # a < b < c
    # a must be > 31/5 = 6.2, so a >= 7
    # 3/a > 5/31 => a < 93/5 = 18.6, so a <= 18
    
    for a in range(7, 19):
        r = target - Fraction(1, a)
        if r <= 0:
            continue
        N = r.numerator
        D = r.denominator
        
        # 1/b + 1/c = N/D
        # b > D/N
        # 2/b > N/D => b < 2D/N
        b_min = max(a + 1, D // N + 1)
        b_max = (2 * D) // N
        
        for b in range(b_min, b_max + 1):
            # c = b * D / (b * N - D)
            # Let K = b * N - D
            # We need K to divide b * D
            # And c must be integer and c > b
            
            K = b * N - D
            if K <= 0:
                continue
                
            # Check if K divides b * D
            # To avoid large numbers, check gcd
            # K | b * D  <=>  K / gcd(K, b) | D
            g = math.gcd(K, b)
            if D % (K // g) != 0:
                continue
            
            c = (b * D) // K
            
            # Check distinctness and order
            if c <= b:
                continue
            if c == a or c == b:
                continue
            if a == b or a == c:
                continue
            
            max_denom = max(a, b, c)
            if max_denom < best_max_denom:
                best_max_denom = max_denom
                best_solution = [a, b, c]

    if best_solution:
        return best_solution

    # Fallback to greedy if no 3-term solution found (unlikely)
    r = target
    values = []
    while r:
        d = (r.denominator + r.numerator - 1) // r.numerator
        values.append(d)
        r -= Fraction(1, d)
    return values
# EVOLVE-BLOCK-END
