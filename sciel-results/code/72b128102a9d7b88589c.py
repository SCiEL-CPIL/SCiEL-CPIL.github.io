"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
from fractions import Fraction
import math

def construct():
    """Find minimum number of distinct denominators >= 5 whose reciprocal sum is 1."""
    
    def find_two_term(num, den, min_d):
        """Find c, d > min_d such that 1/c + 1/d = num/den.
        Returns (c, d) or None."""
        # 1/c + 1/d = num/den => (c+d)/(cd) = num/den
        # den*(c+d) = num*cd => num*cd - den*c - den*d = 0
        # Multiply by num: num^2*cd - num*den*c - num*den*d = 0
        # Add den^2: (num*c - den)*(num*d - den) = den^2
        # So we need factor pairs (f1, f2) of den^2 where:
        #   num*c = f1 + den => c = (f1 + den) / num
        #   num*d = f2 + den => d = (f2 + den) / num
        #   c > min_d, d > c, both integers
        
        # Simplify num/den first
        g = math.gcd(num, den)
        num_r = num // g
        den_r = den // g
        
        # (num_r * c - den_r) * (num_r * d - den_r) = den_r^2
        target = den_r * den_r
        
        # Find all divisors of target
        # target = den_r^2, so divisors are products of divisors of den_r
        # But we can just iterate up to sqrt(target)
        # For efficiency, we'll iterate f1 from 1 to sqrt(target)
        # f1 must divide target, and f2 = target // f1
        
        # c = (f1 + den_r) / num_r must be integer > min_d
        # d = (f2 + den_r) / num_r must be integer > c
        
        # f1 + den_r must be divisible by num_r
        # f2 + den_r must be divisible by num_r
        
        # Let's find all divisors of target
        divisors = []
        i = 1
        while i * i <= target:
            if target % i == 0:
                divisors.append(i)
                if i != target // i:
                    divisors.append(target // i)
            i += 1
        
        best = None
        for f1 in divisors:
            f2 = target // f1
            # c = (f1 + den_r) / num_r
            if (f1 + den_r) % num_r != 0:
                continue
            c = (f1 + den_r) // num_r
            if (f2 + den_r) % num_r != 0:
                continue
            d = (f2 + den_r) // num_r
            
            # Ensure c <= d and both > min_d
            if c > d:
                c, d = d, c
            
            if c <= min_d or d <= c:
                continue
            
            # Verify
            if Fraction(1, c) + Fraction(1, d) != Fraction(num_r, den_r):
                continue
            
            # Prefer smaller c (greedy)
            if best is None or c < best[0]:
                best = (c, d)
        
        return best
    
    def find_three_term(num, den, min_d):
        """Find b, c, d > min_d such that 1/b + 1/c + 1/d = num/den.
        Returns (b, c, d) or None."""
        # For each possible b > min_d, check if 1/c + 1/d = num/den - 1/b
        # 1/b < num/den => b > den/num
        # Also 1/b + 1/c + 1/d = num/den, with c > b, d > c
        # So 1/b < num/den => b > den/num
        # And 1/b > num/den - 1/(b+1) - 1/(b+2) ... roughly
        # Upper bound on b: 1/b >= num/(3*den) => b <= 3*den/num
        
        # Simplify
        g = math.gcd(num, den)
        num_r = num // g
        den_r = den // g
        
        # b must satisfy: b > min_d, b > den_r/num_r
        # 1/b < num_r/den_r => b > den_r/num_r
        min_b = max(min_d + 1, den_r // num_r + 1)
        
        # Upper bound: 1/b >= num_r/(3*den_r) => b <= 3*den_r/num_r
        max_b = (3 * den_r) // num_r + 1
        max_b = min(max_b, 10000000)
        
        # Try b from min_b to max_b
        # To be efficient, we can limit the search
        # For each b, compute remaining = num_r/den_r - 1/b = (num_r*b - den_r) / (den_r*b)
        # Then find 2-term representation
        
        # We want the smallest valid b (greedy)
        for b in range(min_b, max_b + 1):
            # remaining = num_r/den_r - 1/b = (num_r*b - den_r) / (den_r*b)
            rem_num = num_r * b - den_r
            rem_den = den_r * b
            
            if rem_num <= 0:
                break  # 1/b >= num_r/den_r, so remaining <= 0
            
            # Check if remaining can be represented as 1/c + 1/d with c > b, d > c
            # rem_num/rem_den must be positive and less than 2/(b+1) roughly
            # Actually, we need c > b and d > c, so 1/c < 1/b and 1/d < 1/c
            # So rem_num/rem_den < 2/b
            if rem_num * b >= 2 * rem_den:
                continue  # Too large for 2 terms with denominators > b
            
            # Also rem_num/rem_den > 0, already checked
            # And rem_num/rem_den must be representable
            # Lower bound: 1/c + 1/d < 2/(b+1), so rem_num/rem_den < 2/(b+1)
            # Upper bound: 1/c + 1/d > 1/(b+1) + 1/(b+2) approximately
            
            result = find_two_term(rem_num, rem_den, b)
            if result is not None:
                c, d = result
                # Verify
                if Fraction(1, b) + Fraction(1, c) + Fraction(1, d) == Fraction(num_r, den_r):
                    if b < c < d and b > min_d:
                        return (b, c, d)
        
        return None
    
    def find_four_term():
        """Find a, b, c, d >= 5 such that 1/a + 1/b + 1/c + 1/d = 1."""
        # For each a >= 5, remaining = 1 - 1/a = (a-1)/a
        # Need 1/b + 1/c + 1/d = (a-1)/a with b > a, c > b, d > c
        
        # a must satisfy: 1/a < 1 => a >= 2, but a >= 5
        # Also 1/a + 1/(a+1) + 1/(a+2) + 1/(a+3) <= 1 for feasibility
        # Actually, we need the remaining to be representable by 3 terms > a
        
        # Upper bound on a: 1/a >= 1/4 roughly (since we need 3 more terms)
        # More precisely: 1/a + 1/(a+1) + 1/(a+2) + 1/(a+3) <= 1
        # For large a, this sum is ~ 4/a, so a >= 4. But we need it to be <= 1.
        # Actually, we need 1 - 1/a to be representable by 3 terms > a.
        # The maximum sum of 3 terms > a is 1/(a+1) + 1/(a+2) + 1/(a+3)
        # So we need 1 - 1/a <= 1/(a+1) + 1/(a+2) + 1/(a+3)
        
        # Let's just try a from 5 upward
        # a must be such that (a-1)/a can be written as sum of 3 unit fractions > a
        # 1/b + 1/c + 1/d = (a-1)/a
        # b > a, so 1/b < 1/a
        # Max possible: 1/(a+1) + 1/(a+2) + 1/(a+3)
        # We need (a-1)/a <= 1/(a+1) + 1/(a+2) + 1/(a+3)
        
        for a in range(5, 200):  # Upper bound on a
            rem_num = a - 1
            rem_den = a
            
            # Check feasibility
            # Max sum of 3 terms > a: 1/(a+1) + 1/(a+2) + 1/(a+3)
            max_sum = Fraction(1, a+1) + Fraction(1, a+2) + Fraction(1, a+3)
            if Fraction(rem_num, rem_den) > max_sum:
                continue
            
            # Also min sum: as denominators go to infinity, sum -> 0, so always feasible in that sense
            # But we need b > a, c > b, d > c, all finite
            
            result = find_three_term(rem_num, rem_den, a)
            if result is not None:
                b, c, d = result
                # Verify
                if Fraction(1, a) + Fraction(1, b) + Fraction(1, c) + Fraction(1, d) == Fraction(1, 1):
                    if a < b < c < d and a >= 5:
                        return [a, b, c, d]
        
        return None
    
    # Try 4 terms first (minimum possible)
    result = find_four_term()
    if result is not None:
        return result
    
    # If 4 terms don't work, try 5 terms
    # For 5 terms: 1/a + 1/b + 1/c + 1/d + 1/e = 1
    # For each a, remaining = 1 - 1/a, need 4 terms > a
    
    def find_five_term():
        for a in range(5, 50):
            rem_num = a - 1
            rem_den = a
            
            # Need 4 terms > a summing to (a-1)/a
            # Max: 1/(a+1) + 1/(a+2) + 1/(a+3) + 1/(a+4)
            max_sum = sum(Fraction(1, a+i) for i in range(1, 5))
            if Fraction(rem_num, rem_den) > max_sum:
                continue
            
            # Find 4-term representation of rem_num/rem_den with denominators > a
            # For each b > a, remaining2 = rem_num/rem_den - 1/b
            # Need 3 terms > b
            
            min_b = a + 1
            # Upper bound on b: 1/b >= rem_num/(4*rem_den) => b <= 4*rem_den/rem_num
            max_b = (4 * rem_den) // rem_num + 1
            max_b = min(max_b, 10000000)
            
            for b in range(min_b, max_b + 1):
                rem2_num = rem_num * b - rem_den
                rem2_den = rem_den * b
                
                if rem2_num <= 0:
                    break
                
                # Need 3 terms > b summing to rem2_num/rem2_den
                result3 = find_three_term(rem2_num, rem2_den, b)
                if result3 is not None:
                    c, d, e = result3
                    if Fraction(1, a) + Fraction(1, b) + Fraction(1, c) + Fraction(1, d) + Fraction(1, e) == Fraction(1, 1):
                        if a < b < c < d < e and a >= 5:
                            return [a, b, c, d, e]
        
        return None
    
    result = find_five_term()
    if result is not None:
        return result
    
    # Fallback: use a known valid representation
    # Let's construct one: 1 = 1/5 + 1/6 + 1/7 + 1/8 + 1/9 + 1/10 + 1/14 + 1/35 + 1/45 + 1/63 + 1/90 + 1/126 + 1/180 + 1/315 + 1/540 + 1/840 + 1/1260 + 1/1890 + 1/2520 + 1/4200
    # This is too many. Let me just find one computationally with a simpler approach.
    
    # Use a greedy algorithm as final fallback
    def greedy(remaining, min_d, max_terms):
        if max_terms == 0:
            return None if remaining != 0 else []
        if remaining == 0:
            return []
        
        # min_d: smallest allowed denominator
        # Try d from min_d upward
        # d must satisfy 1/d <= remaining => d >= ceil(1/remaining)
        if remaining < 1:
            min_allowed = max(min_d, (remaining.denominator + remaining.numerator - 1) // remaining.numerator)
        else:
            min_allowed = min_d
        
        max_allowed = max(min_allowed, max_terms * remaining.denominator // remaining.numerator + 1)
        max_allowed = min(max_allowed, 10000000)
        
        for d in range(min_allowed, min(max_allowed + 1, min_allowed + 5000)):
            new_rem = remaining - Fraction(1, d)
            if new_rem < 0:
                break
            if new_rem == 0:
                return [d]
            if max_terms - 1 >= 1 and new_rem > 0:
                rest = greedy(new_rem, d + 1, max_terms - 1)
                if rest is not None:
                    return [d] + rest
        return None
    
    for n in range(4, 20):
        res = greedy(Fraction(1, 1), 5, n)
        if res is not None:
            s = sum(Fraction(1, d) for d in res)
            if s == Fraction(1, 1) and len(set(res)) == len(res) and all(5 <= d <= 10000000 for d in res):
                return res
    
    raise RuntimeError("no representation found")
# EVOLVE-BLOCK-END
