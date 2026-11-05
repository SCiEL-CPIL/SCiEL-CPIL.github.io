import time
from fractions import Fraction
from math import isqrt

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin for 60s limit
    
    target = Fraction(5, 31)
    max_terms_allowed = 256
    max_denom_limit = 10000000
    
    best_max = 1736
    best_solution = [8, 28, 1736]
    
    # Function to get divisors of n
    def get_divisors(n):
        divs = []
        for i in range(1, isqrt(n) + 1):
            if n % i == 0:
                divs.append(i)
                if i != n // i:
                    divs.append(n // i)
        return divs

    # Search for 3-term solutions: 1/a + 1/b + 1/c = 5/31
    # 1/b + 1/c = (5a - 31) / (31a)
    # Let N = 5a - 31, D = 31a
    # c = D*b / (N*b - D)
    # Let y = N*b - D => b = (D+y)/N
    # c = D*(D+y)/N / y = (D^2 + D*y) / (N*y)
    # Conditions:
    # 1. y must be a divisor of D^2 / gcd(D^2, N) ? 
    #    Actually, we just need (D^2 + D*y) to be divisible by N*y.
    #    This implies N*y | D^2 + D*y.
    #    Since N*y | D*y, we must have N*y | D^2.
    #    So y must be a divisor of D^2 / gcd(D^2, N) is sufficient?
    #    Let G = gcd(D^2, N). Then D^2 = G * K.
    #    N*y | G*K.
    #    This doesn't mean y | K.
    #    However, iterating all divisors of D^2 is safe and fast enough for small a.
    
    # We limit 'a' to a reasonable range. 
    # If a is large, 1/a is small. 
    # Max denominator is minimized by small a, b, c generally.
    # The known solution has a=8.
    # Let's search a up to 100.
    
    for a in range(1, 200):
        if time.time() > deadline:
            break
            
        N = 5 * a - 31
        if N <= 0:
            continue
            
        D = 31 * a
        D2 = D * D
        
        # Get divisors of D2
        # Optimization: We only care about y such that b = (D+y)/N is integer and b > a.
        # b > a => (D+y)/N > a => D+y > a*N => y > a*N - D = a(5a-31) - 31a = 5a^2 - 31a - 31a = 5a^2 - 62a.
        # Also b < c.
        
        divs = get_divisors(D2)
        
        for y in divs:
            if time.time() > deadline:
                break
            
            # Check if N*y divides D^2 + D*y
            # Which is equivalent to N*y dividing D^2
            if D2 % (N * y) != 0:
                continue
                
            # Calculate b
            num_b = D + y
            if num_b % N != 0:
                continue
            b = num_b // N
            
            # Check distinctness and order
            if b <= a:
                continue
                
            # Calculate c
            c = (D2 + D * y) // (N * y)
            
            if c <= b:
                continue
                
            # Check if c is integer (it is by construction if divisibility holds)
            # Check max denom
            current_max = c
            if current_max < best_max:
                # Verify sum
                if Fraction(1, a) + Fraction(1, b) + Fraction(1, c) == target:
                    best_max = current_max
                    best_solution = [a, b, c]
                    
    # If we found a better solution, return it.
    # If not, return the fallback.
    # We can also try a quick 4-term search if time permits, but 3-term is very likely optimal.
    # Given the stagnation, the 3-term search is a structural improvement over the DFS which might miss small 3-term solutions due to pruning or ordering.
    
    return best_solution
