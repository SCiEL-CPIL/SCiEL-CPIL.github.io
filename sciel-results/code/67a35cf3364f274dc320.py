import time
from fractions import Fraction

def construct():
    target = Fraction(5, 31)
    
    # Known solution from seed
    best_solution = [8, 28, 1736]
    best_max_denom = 1736
    
    start_time = time.time()
    time_limit = 55.0

    def verify(solution):
        if not solution:
            return False
        if len(solution) < 1 or len(solution) > 256:
            return False
        if len(set(solution)) != len(solution):
            return False
        if any(d < 1 or d > 10000000 for d in solution):
            return False
        s = sum(Fraction(1, d) for d in solution)
        return s == target

    # Search for 3-term solutions: 1/a + 1/b + 1/c = 5/31
    # a < b < c
    # a is in [7, 18]
    for a in range(7, 19):
        if time.time() - start_time > time_limit:
            break
            
        rem1 = Fraction(5, 31) - Fraction(1, a)
        if rem1 <= 0:
            continue
            
        # 1/b + 1/c = rem1
        # b > 1/rem1
        # 2/b > rem1 => b < 2/rem1
        b_low = int(1 / rem1) + 1
        b_high = int(2 / rem1)
        
        # Also b > a
        b_low = max(b_low, a + 1)
        
        if b_low > b_high:
            continue
            
        # We want to minimize c.
        # c = 1 / (rem1 - 1/b)
        # To minimize c, we maximize (rem1 - 1/b), which means minimizing 1/b, i.e., maximizing b.
        # So iterate b from b_high down to b_low.
        for b in range(b_high, b_low - 1, -1):
            if time.time() - start_time > time_limit:
                break
                
            rem2 = rem1 - Fraction(1, b)
            if rem2 <= 0:
                continue
                
            # c = 1 / rem2
            # c is integer if rem2.denominator % rem2.numerator == 0
            if rem2.denominator % rem2.numerator == 0:
                c = rem2.denominator // rem2.numerator
                
                if c <= b:
                    continue
                    
                if c < best_max_denom:
                    best_solution = [a, b, c]
                    best_max_denom = c
                    # Since we iterate b descending, the first valid c found for a given a
                    # might not be the global minimum if a different a yields smaller c.
                    # But for fixed a, larger b -> smaller c.
                    # So the first valid c found for this a is the best for this a.
                    # We can break inner loop?
                    # No, we should check all a.
                    # But we can break inner loop if we found a c that is very small?
                    # Just keep searching.
                    
    if verify(best_solution):
        return best_solution
        
    return [8, 28, 1736]
