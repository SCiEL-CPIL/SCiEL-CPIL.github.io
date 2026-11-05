"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

def construct():
    """
    Find 1..256 distinct denominators in [1,10000000] summing reciprocally to 5/31,
    minimizing the largest denominator.
    
    Strategy: Systematic search for 3-term Egyptian fractions 1/a + 1/b + 1/c = 5/31
    with a < b < c, finding the minimum possible value of c (the largest denominator).
    """
    start_time = time.time()
    deadline = start_time + 80  # 80s safety margin
    
    # We need 1/a + 1/b + 1/c = 5/31 with distinct positive integers
    # WLOG: a < b < c
    # 1/a < 5/31 => a > 31/5 = 6.2 => a >= 7
    # Also 1/a + 1/b + 1/c = 5/31, and since b > a, c > b:
    # 1/a > 1/3 * (5/31) roughly, but more precisely:
    # 1/a + 1/(a+1) + 1/(a+2) >= 5/31 for feasibility
    # And 3/a > 5/31 => a < 93/5 = 18.6 => a <= 18 (since a < b < c, 1/a is the largest term)
    # Actually: 1/a + 1/b + 1/c < 3/a, so 3/a > 5/31 => a < 93/5 = 18.6, so a <= 18
    
    best_max = float('inf')
    best_solution = None
    
    # a ranges from 7 to 18
    for a in range(7, 19):
        if time.time() > deadline:
            break
        # 1/b + 1/c = 5/31 - 1/a
        # Let r = 5/31 - 1/a = (5a - 31) / (31a)
        # We need 1/b + 1/c = r where b > a, c > b
        # 1/b < r => b > 1/r = 31a / (5a - 31)
        # Also 1/b + 1/c < 2/b => 2/b > r => b < 2/r = 62a / (5a - 31)
        # And c = 1/(r - 1/b) must be a positive integer > b
        
        num = 5 * a - 31  # numerator of r
        den = 31 * a      # denominator of r
        # r = num/den, must be positive
        if num <= 0:
            continue
        
        # b must satisfy: b > a and 1/b < r and 1/b + 1/c = r
        # 1/b < r => b > den/num
        # Also 1/b > r/2 (since 1/c < 1/b, so 1/b > r/2) => b < 2*den/num
        # More precisely: 1/b + 1/c = r, c > b => 1/c < 1/b => r = 1/b + 1/c < 2/b => b < 2/r = 2*den/num
        # And 1/b < r => b > den/num
        # Also b > a
        
        b_min = max(a + 1, den // num + 1)
        b_max = (2 * den) // num - 1  # b < 2*den/num, so b <= (2*den//num) - 1
        
        # Actually let's be more careful:
        # r = num/den
        # 1/b < r => b > 1/r = den/num
        # 2/b > r => b < 2/r = 2*den/num
        # So b ranges from floor(den/num) + 1 to floor(2*den/num) - 1, but also b > a
        
        b_lower = max(a + 1, den // num + 1)
        b_upper = (2 * den) // num  # b < 2*den/num means b <= 2*den/num - epsilon, so b <= (2*den)//num - 1 if 2*den not divisible, or (2*den)//num - 1
        
        # Let me just iterate b in a reasonable range
        # b must be > a, and 1/b < r = num/den => b > den/num
        # And 1/b > r/2 (because 1/c > 0 so 1/b < r, and 1/b + 1/c = r with c > b means 1/b > r/2)
        # So r/2 < 1/b < r => 1/r < b < 2/r => den/num < b < 2*den/num
        
        for b in range(b_lower, b_upper + 1):
            if time.time() > deadline:
                break
            # Check: 1/b < r => b > den/num (already ensured)
            # Compute 1/c = r - 1/b = num/den - 1/b = (num*b - den) / (den*b)
            # c = den*b / (num*b - den)
            # c must be a positive integer > b
            
            numerator_c = den * b
            denominator_c = num * b - den
            
            if denominator_c <= 0:
                continue
            
            if numerator_c % denominator_c != 0:
                continue
            
            c = numerator_c // denominator_c
            
            if c <= b:
                continue
            
            if c > 10000000:
                continue
            
            # Check distinctness: a, b, c are distinct (a < b < c already)
            # Verify the sum
            # 1/a + 1/b + 1/c should equal 5/31
            # Check: (b*c + a*c + a*b) / (a*b*c) == 5/31
            # 31*(b*c + a*c + a*b) == 5*a*b*c
            
            lhs = 31 * (b * c + a * c + a * b)
            rhs = 5 * a * b * c
            
            if lhs != rhs:
                continue
            
            current_max = max(a, b, c)
            if current_max < best_max:
                best_max = current_max
                best_solution = [a, b, c]
    
    if best_solution is not None:
        return best_solution
    else:
        # Fallback: use the known solution
        return [8, 31, 248]

# EVOLVE-BLOCK-END
