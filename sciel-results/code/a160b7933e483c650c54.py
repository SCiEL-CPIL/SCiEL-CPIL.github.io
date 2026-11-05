"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from fractions import Fraction
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin
    
    target = Fraction(5, 31)
    
    # Initial greedy solution to have a valid fallback
    def greedy_solution(target):
        denominators = []
        r = target
        while r > 0:
            d = (r.denominator + r.numerator - 1) // r.numerator
            denominators.append(d)
            r -= Fraction(1, d)
        return denominators
    
    best_solution = greedy_solution(target)
    best_max = max(best_solution)
    
    # Meet-in-the-middle for 3-term solutions
    # Precompute all pairs (d1, d2) with d1 < d2, d1 < d2 <= B
    # Store sum = 1/d1 + 1/d2 in a dict: sum -> list of (d1, d2)
    
    def find_3_term_solutions(B):
        """Find 3-term solutions where all denominators <= B"""
        nonlocal best_solution, best_max
        
        # Precompute all 2-term sums
        pair_sums = defaultdict(list)
        
        # For 1/d1 + 1/d2 = s, with d1 < d2 <= B
        # d2 = d1 / (s*d1 - 1) if s*d1 > 1
        # So for each d1, we need s*d1 - 1 to divide d1
        
        # Alternative: iterate d1 from 2 to B, d2 from d1+1 to B
        # But that's O(B^2) which is too slow for large B
        
        # Better: For each d1, the possible d2 values are constrained
        # 1/d1 + 1/d2 = s => d2 = d1 / (s*d1 - 1)
        # So s*d1 - 1 must be a positive divisor of d1
        
        # Let's use a different approach:
        # For 3 terms: 1/a + 1/b + 1/c = 5/31, a < b < c <= B
        # Fix a, then 1/b + 1/c = 5/31 - 1/a = (5a - 31) / (31a)
        # For this to be positive, we need 5a > 31, so a >= 7
        # Then b and c must satisfy: 1/b + 1/c = (5a-31)/(31a)
        # => 31a(b+c) = (5a-31)bc
        # => (5a-31)bc - 31ab - 31ac = 0
        # => (5a-31)b c - 31a(b+c) = 0
        # Add 31a^2/(5a-31) to both sides... this is getting complex
        
        # Simpler: For fixed a, we need 1/b + 1/c = r where r = (5a-31)/(31a)
        # Then c = b / (r*b - 1), so r*b - 1 must divide b
        # r*b - 1 = (5a-31)*b / (31a) - 1 = ((5a-31)*b - 31a) / (31a)
        # So ((5a-31)*b - 31a) must divide b*31a
        
        # Let's just iterate a from 7 to B, and for each a, find b > a such that
        # r = (5a-31)/(31a) and c = b/(r*b - 1) is an integer > b
        
        for a in range(7, min(B, 200) + 1):  # a can't be too large for 3 terms
            if time.time() > deadline:
                break
            
            # r = 1/b + 1/c = (5a-31)/(31a)
            num = 5 * a - 31
            den = 31 * a
            if num <= 0:
                continue
            
            # We need 1/b + 1/c = num/den
            # => c = b*den / (num*b - den)
            # So num*b - den must be positive and divide b*den
            
            # b > a, and 1/b < num/den => b > den/num
            min_b = max(a + 1, den // num + 1)
            
            # Also 1/b + 1/c > 1/b + 1/(b+1) for c > b, so:
            # num/den > 1/b + 1/(b+1) = (2b+1)/(b(b+1))
            # This gives an upper bound on b, but it's complex.
            # Instead, note that c > b, so 1/c < 1/b, thus 1/b > num/(2*den)
            # => b < 2*den/num
            max_b = min(B, 2 * den // num)
            
            if min_b > max_b:
                continue
            
            for b in range(min_b, max_b + 1):
                if time.time() > deadline:
                    break
                
                # c = b * den / (num * b - den)
                denom_c = num * b - den
                if denom_c <= 0:
                    continue
                
                if (b * den) % denom_c != 0:
                    continue
                
                c = (b * den) // denom_c
                if c <= b or c > B:
                    continue
                
                # Check: 1/a + 1/b + 1/c == target
                if Fraction(1, a) + Fraction(1, b) + Fraction(1, c) != target:
                    continue
                
                candidate = [a, b, c]
                curr_max = c
                if curr_max < best_max:
                    best_max = curr_max
                    best_solution = candidate
    
    # First, try to find a 3-term solution with small max denominator
    # Start with small B and increase
    for B in [50, 100, 200, 500, 1000, 2000, 5000]:
        if time.time() > deadline:
            break
        find_3_term_solutions(B)
    
    # If we haven't found a 3-term solution, try 4-term with meet-in-middle
    # For 4 terms: 1/a + 1/b + 1/c + 1/d = 5/31
    # Fix a, b, then 1/c + 1/d = 5/31 - 1/a - 1/b
    # This is similar to the 3-term case but with two fixed terms
    
    def find_4_term_solutions(B):
        nonlocal best_solution, best_max
        
        # Precompute all 2-term sums 1/c + 1/d with c < d <= B
        # Store in dict: sum -> (c, d)
        two_term_sums = {}
        
        # For efficiency, iterate c from 2 to B//2, d from c+1 to B
        # But that's O(B^2) which is too slow for B=5000
        
        # Instead, use the same technique as 3-term:
        # For fixed c, d = c*den / (num*c - den) where 1/c + 1/d = num/den
        
        # For 4 terms: fix a, b, then need 1/c + 1/d = r where r = target - 1/a - 1/b
        # Then d = c / (r*c - 1), so r*c - 1 must divide c
        
        for a in range(7, min(B, 100) + 1):
            if time.time() > deadline:
                break
            
            for b in range(a + 1, min(B, 200) + 1):
                if time.time() > deadline:
                    break
                
                r = target - Fraction(1, a) - Fraction(1, b)
                if r <= 0:
                    continue
                
                num = r.numerator
                den = r.denominator
                
                # Need 1/c + 1/d = num/den, c > b, d > c
                # d = c*den / (num*c - den)
                # num*c - den > 0 => c > den/num
                min_c = max(b + 1, den // num + 1)
                max_c = min(B, 2 * den // num)
                
                if min_c > max_c:
                    continue
                
                for c in range(min_c, max_c + 1):
                    if time.time() > deadline:
                        break
                    
                    denom_d = num * c - den
                    if denom_d <= 0:
                        continue
                    
                    if (c * den) % denom_d != 0:
                        continue
                    
                    d = (c * den) // denom_d
                    if d <= c or d > B:
                        continue
                    
                    # Verify
                    if Fraction(1, a) + Fraction(1, b) + Fraction(1, c) + Fraction(1, d) != target:
                        continue
                    
                    candidate = [a, b, c, d]
                    curr_max = d
                    if curr_max < best_max:
                        best_max = curr_max
                        best_solution = candidate
    
    # Try 4-term solutions if needed
    if best_max > 500:  # Only if 3-term didn't find something good
        for B in [200, 500, 1000]:
            if time.time() > deadline:
                break
            find_4_term_solutions(B)
    
    return best_solution
# EVOLVE-BLOCK-END
