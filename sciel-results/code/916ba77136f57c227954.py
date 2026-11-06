import time
from fractions import Fraction

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # 55 seconds to be safe
    
    target = Fraction(5, 31)
    best_max_denom = float('inf')
    best_solution = None
    
    def update_best(candidate):
        nonlocal best_max_denom, best_solution
        if candidate is not None and len(set(candidate)) == len(candidate):
            max_d = max(candidate)
            if max_d < best_max_denom:
                best_max_denom = max_d
                best_solution = candidate
    
    # Search for 3-term solutions: 1/d1 + 1/d2 + 1/d3 = 5/31
    # For fixed d1, we need 1/d2 + 1/d3 = 5/31 - 1/d1
    # Let rem = 5/31 - 1/d1 = (5*d1 - 31)/(31*d1)
    # Then 1/d2 + 1/d3 = rem => (d2+d3)/(d2*d3) = rem
    # => d2*d3*rem = d2 + d3
    # => d2*d3*(5*d1 - 31)/(31*d1) = d2 + d3
    # => d2*d3*(5*d1 - 31) = 31*d1*(d2 + d3)
    # This is a Diophantine equation. We can rearrange:
    # (d2 - 31*d1/(5*d1-31)) * (d3 - 31*d1/(5*d1-31)) = (31*d1/(5*d1-31))^2
    # But this involves fractions. Better approach:
    # Let A = 5*d1 - 31, B = 31*d1
    # Then A*d2*d3 = B*(d2 + d3)
    # A*d2*d3 - B*d2 - B*d3 = 0
    # (A*d2 - B)*(A*d3 - B) = B^2
    # So for fixed d1, we enumerate divisors of B^2.
    
    # d1 must satisfy: 1/d1 < 5/31 => d1 > 31/5 = 6.2, so d1 >= 7
    # Also d1 must be such that 5*d1 - 31 > 0, so d1 >= 7
    # And d1 should be large enough that the remainder is positive.
    
    for d1 in range(7, 100):  # d1 up to 100 should cover optimal
        if time.time() > deadline:
            break
        
        A = 5 * d1 - 31
        B = 31 * d1
        
        if A <= 0:
            continue
        
        # (A*d2 - B) * (A*d3 - B) = B^2
        # Let X = A*d2 - B, Y = A*d3 - B
        # X * Y = B^2
        # X and Y are divisors of B^2
        # d2 = (X + B) / A, d3 = (Y + B) / A
        # We need d2 > d1, d3 > d2, and both integers
        
        B_sq = B * B
        
        # Enumerate divisors of B_sq
        # To avoid duplicates, we only consider X <= Y, so X <= sqrt(B_sq)
        # But B_sq can be large. Let's be smart:
        # We need X such that (X + B) % A == 0 and (Y + B) % A == 0 where Y = B_sq // X
        # Also X < Y to avoid duplicates, and X > 0
        
        # Find all divisors of B_sq
        divisors = []
        i = 1
        while i * i <= B_sq:
            if B_sq % i == 0:
                divisors.append(i)
                if i != B_sq // i:
                    divisors.append(B_sq // i)
            i += 1
        
        for X in divisors:
            if time.time() > deadline:
                break
            
            if B_sq % X != 0:
                continue
            
            Y = B_sq // X
            
            # Ensure X <= Y to avoid duplicate pairs
            if X > Y:
                continue
            
            d2 = (X + B)
            d3 = (Y + B)
            
            # Check divisibility by A
            if d2 % A != 0 or d3 % A != 0:
                continue
            
            d2 //= A
            d3 //= A
            
            # Ensure distinctness and ordering
            if d2 <= d1 or d3 <= d2:
                continue
            
            # Verify the solution
            s = Fraction(1, d1) + Fraction(1, d2) + Fraction(1, d3)
            if s == target:
                candidate = [d1, d2, d3]
                update_best(candidate)
    
    # If no 3-term solution found, try 2-term (unlikely but check)
    for d1 in range(7, 100):
        if time.time() > deadline:
            break
        rem = target - Fraction(1, d1)
        if rem > 0:
            d2 = (rem.denominator + rem.numerator - 1) // rem.numerator
            if rem == Fraction(1, d2) and d2 != d1:
                candidate = [d1, d2]
                update_best(candidate)
    
    # If still no solution, use a known good solution or greedy fallback
    if best_solution is None:
        # Known optimal 3-term solution for 5/31
        # 1/8 + 1/31 + 1/248 = 31/248 + 8/248 + 1/248 = 40/248 = 5/31
        candidate = [8, 31, 248]
        s = Fraction(1, 8) + Fraction(1, 31) + Fraction(1, 248)
        if s == target:
            best_solution = candidate
        else:
            # Greedy fallback
            r = target
            values = []
            while r > 0:
                d = (r.denominator + r.numerator - 1) // r.numerator
                if d in values:
                    d += 1
                values.append(d)
                r -= Fraction(1, d)
                if len(values) > 256:
                    break
            best_solution = values
    
    return best_solution
