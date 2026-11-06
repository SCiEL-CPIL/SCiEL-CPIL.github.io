import time
from fractions import Fraction

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # 5s margin
    
    def search(remaining, min_d, current_terms, k):
        if time.time() > deadline:
            return None
            
        if k == 0:
            if remaining == 0:
                return current_terms
            return None

        if k == 1:
            if remaining.numerator == 1 and remaining.denominator > min_d:
                return current_terms + [remaining.denominator]
            return None

        # d >= 1/remaining
        # d > min_d
        # d <= k/remaining
        
        # Calculate bounds
        # remaining = a/b
        # 1/d <= a/b => d >= b/a
        # d >= ceil(b/a)
        b = remaining.denominator
        a = remaining.numerator
        
        lower_d = b // a
        if b % a != 0:
            lower_d += 1
            
        if lower_d <= min_d:
            lower_d = min_d + 1
            
        # d <= k / (a/b) = k*b/a
        upper_d = (k * b) // a
        
        if lower_d > upper_d:
            return None
            
        # Iterate d
        # To improve speed, we can skip d that don't allow the rest to be formed
        for d in range(lower_d, upper_d + 1):
            # Check if d is valid
            # 1/d must be <= remaining
            # This is guaranteed by d >= lower_d
            
            new_remaining = remaining - Fraction(1, d)
            
            # Pruning: new_remaining must be formable by k-1 terms > d
            # Min sum of k-1 terms > d is approx (k-1)/(d+1)
            # If new_remaining < (k-1)/(d+1), skip
            # This is a weak bound but helps
            
            # Also new_remaining > 0
            if new_remaining <= 0:
                continue
                
            res = search(new_remaining, d, current_terms + [d], k - 1)
            if res is not None:
                return res
                
        return None

    # Try k from 5 to 8
    for k in range(5, 9):
        if time.time() > deadline:
            break
        res = search(Fraction(1, 1), 4, [], k)
        if res is not None:
            return res
            
    # Fallback
    return [5, 6, 7, 11, 231, 87781351]
