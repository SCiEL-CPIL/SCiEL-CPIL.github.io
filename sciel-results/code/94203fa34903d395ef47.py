from fractions import Fraction
import time

def construct():
    deadline = time.time() + 80
    
    # Precompute max_sum table for quick lookup
    # max_sum_table[k][start] = max sum of k distinct unit fractions with denominators >= start
    # We'll compute on demand and cache
    
    max_sum_cache = {}
    
    def max_sum(start, k):
        """Max sum of k distinct unit fractions with denominators >= start"""
        if k == 0:
            return Fraction(0, 1)
        key = (start, k)
        if key in max_sum_cache:
            return max_sum_cache[key]
        s = Fraction(0, 1)
        for i in range(k):
            s += Fraction(1, start + i)
        max_sum_cache[key] = s
        return s
    
    # Precompute some common max_sum values
    for k in range(1, 15):
        for start in range(5, 100):
            max_sum(start, k)
    
    def search(remaining, min_d, count, path):
        if time.time() > deadline:
            return None
        
        if count == 0:
            if remaining == 0:
                return path[:]
            return None
        
        if count == 1:
            # remaining must be 1/d
            if remaining.numerator == 1:
                d = remaining.denominator
                if d >= min_d and d <= 10000000:
                    return path + [d]
            return None
        
        # Lower bound for d: 1/d <= remaining => d >= ceil(1/remaining)
        if remaining > 0:
            # ceil(1/remaining) = ceil(remaining.denominator / remaining.numerator)
            num, den = remaining.denominator, remaining.numerator
            low_d = (num + den - 1) // den
            low_d = max(min_d, low_d)
        else:
            return None
        
        # Upper bound: we need remaining - 1/d <= max_sum(d+1, count-1)
        # As d increases, LHS increases and RHS decreases, so there's a max d
        # Find upper bound by binary search or just iterate with break condition
        
        d = low_d
        
        while True:
            if time.time() > deadline:
                return None
            
            if d > 10000000:
                break
            
            # Check if 1/d <= remaining
            if d * remaining.numerator < remaining.denominator:
                # 1/d > remaining, skip
                d += 1
                continue
            
            rest = remaining - Fraction(1, d)
            
            # Check if rest can be formed by count-1 terms starting from d+1
            if rest > 0:
                if rest > max_sum(d + 1, count - 1):
                    break
            
            # Also check: rest must be >= min possible sum of count-1 terms
            # Min sum of count-1 distinct unit fractions with denominators >= d+1
            # is at least 0, but we need exact match. 
            # A better lower bound: rest must be representable, which requires
            # rest >= sum of smallest (count-1) unit fractions? No, that's max.
            # Actually rest must be >= 0, which we checked.
            
            # Recursive call
            res = search(rest, d + 1, count - 1, path + [d])
            if res is not None:
                return res
            
            d += 1
        
        return None
    
    # Check N from 3 to 20
    for N in range(3, 21):
        if time.time() > deadline:
            break
        
        sol = search(Fraction(1, 1), 5, N, [])
        if sol is not None:
            return sol
    
    # Fallback: try a known solution or raise
    raise RuntimeError("No solution found")
