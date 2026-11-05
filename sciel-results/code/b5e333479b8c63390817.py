"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from fractions import Fraction
import math

def construct():
    start_time = time.time()
    deadline = 85.0  # Safety margin before 90s kill
    
    target = Fraction(5, 31)
    
    best_solution = None
    best_max_den = float('inf')
    
    # Helper to solve 1/c + 1/d = N/D
    # Assumes gcd(N, D) = 1
    def solve_two_terms(N, D, min_den, max_den_limit):
        # We need X | D^2 and X = N*b - D for some integer b
        # So X + D is divisible by N
        # X = N*b - D => b = (X+D)/N
        # c = D/X
        # We want distinct c, d > min_den
        # D^2 can be large, so we iterate divisors of D^2
        # However, iterating all divisors of D^2 is slow if D is large.
        # Alternative: iterate b.
        # b ranges from ceil(D/N) to D (if c >= b)
        # c = D*b / (N*b - D)
        # N*b - D must divide D*b
        # Since gcd(N, D) = 1, gcd(N*b, D) = gcd(b, D).
        # Let g = gcd(b, D). Then N*b - D must divide D*b/g?
        # Simpler: just iterate b and check divisibility.
        # Range of b:
        # 1/b + 1/c = N/D => 1/b < N/D => b > D/N
        # Also 1/b + 1/b >= N/D => 2/b >= N/D => b <= 2D/N
        # So b is in (D/N, 2D/N]
        
        b_min = D // N + 1
        b_max = (2 * D) // N + 1
        
        # Optimization: If N is large, range is small.
        # If N is small, range is large.
        
        solutions = []
        for b in range(b_min, b_max + 1):
            # Check if N*b - D divides D*b
            num = N * b - D
            if num <= 0:
                continue
            if (D * b) % num == 0:
                c = (D * b) // num
                if c < b: # Ensure b <= c to avoid duplicates
                    continue
                if c < min_den or b < min_den:
                    continue
                if c > max_den_limit or b > max_den_limit:
                    continue
                if b == c:
                    continue
                solutions.append((b, c))
        return solutions

    # Search for 3 terms: 1/a + 1/b + 1/c = 5/31
    # Iterate a
    # 1/a < 5/31 => a > 6.2 => a >= 7
    # To keep max denom small, a shouldn't be too big.
    # If a is big, 1/a is small, remaining sum is close to 5/31.
    # Min denom for sum 5/31 is ~7. So b,c will be ~7-10.
    # So a can be larger, but b,c will be small.
    # We want to minimize max(a,b,c).
    # So we should search a up to a reasonable limit, say 1000.
    
    for a in range(7, 1000):
        if time.time() > deadline:
            break
        rem = target - Fraction(1, a)
        if rem <= 0:
            continue
        N = rem.numerator
        D = rem.denominator
        # Reduce fraction
        g = math.gcd(N, D)
        N //= g
        D //= g
        
        # Solve 1/b + 1/c = N/D
        # b must be > a? Not necessarily, but denominators must be distinct.
        # We can assume a < b < c for ordering, then check distinctness.
        # But a is fixed.
        # b_min = max(a+1, D//N + 1)
        # b_max = (2*D)//N + 1
        
        sols = solve_two_terms(N, D, a + 1, best_max_den - 1)
        
        for b, c in sols:
            # Check distinctness
            if a == b or a == c or b == c:
                continue
            current_max = max(a, b, c)
            if current_max < best_max_den:
                best_max_den = current_max
                best_solution = [a, b, c]
    
    # Search for 4 terms: 1/a + 1/b + 1/c + 1/d = 5/31
    # Iterate a, b
    # 1/a + 1/b < 5/31
    # a >= 7
    # b > a
    # 1/b < 5/31 - 1/a
    # b > 1 / (5/31 - 1/a)
    
    for a in range(7, 100):
        if time.time() > deadline:
            break
        rem1 = target - Fraction(1, a)
        if rem1 <= 0:
            continue
        
        # b must be > a
        # 1/b < rem1 => b > 1/rem1
        b_min = max(a + 1, int(1 / rem1) + 1)
        # 1/b + 1/c >= rem1 => 2/b >= rem1 => b <= 2/rem1
        b_max = int(2 / rem1) + 1
        
        # Limit b_max to avoid infinite loop if rem1 is very small
        if b_max > 10000:
            b_max = 10000
            
        for b in range(b_min, b_max + 1):
            if time.time() > deadline:
                break
            rem2 = rem1 - Fraction(1, b)
            if rem2 <= 0:
                continue
            
            N = rem2.numerator
            D = rem2.denominator
            g = math.gcd(N, D)
            N //= g
            D //= g
            
            # Solve 1/c + 1/d = N/D
            # c > b
            sols = solve_two_terms(N, D, b + 1, best_max_den - 1)
            
            for c, d in sols:
                if a == c or a == d or b == c or b == d or c == d:
                    continue
                current_max = max(a, b, c, d)
                if current_max < best_max_den:
                    best_max_den = current_max
                    best_solution = [a, b, c, d]

    if best_solution:
        return best_solution
    else:
        # Fallback to greedy if no solution found (should not happen)
        r, values = Fraction(5, 31), []
        while r:
            d = (r.denominator + r.numerator - 1) // r.numerator
            values.append(d)
            r -= Fraction(1, d)
        return values

# EVOLVE-BLOCK-END
