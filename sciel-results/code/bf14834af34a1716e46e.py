"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
from fractions import Fraction
import time
import math

# EVOLVE-BLOCK-START

def construct():
    deadline = time.time() + 50  # 50 second safety margin
    MAX_DEN = 10**7
    
    def dfs(rem, min_den, terms_left, used):
        """
        rem: Fraction remaining to sum to 1
        min_den: next denominator must be >= min_den (and not in used)
        terms_left: how many more terms we can use
        used: set of denominators already chosen
        Returns list of chosen denominators or None
        """
        if rem == 0:
            return []
        if terms_left == 0:
            return None
        
        # Lower bound on next denominator: 1/d <= rem => d >= 1/rem
        # 1/rem = rem.den / rem.num
        den_lb = max(min_den, math.ceil(rem.denominator / rem.numerator))
        
        if den_lb > MAX_DEN:
            return None
        
        # Pruning: Max possible sum with remaining terms
        # The largest possible terms are from the smallest available denominators:
        # den_lb, den_lb+1, ..., den_lb + terms_left - 1
        # If sum of these is < rem, we can't reach rem.
        # Optimization: For small terms_left, calculate directly.
        if terms_left <= 50:
            max_possible = sum(Fraction(1, den_lb + i) for i in range(terms_left))
            if max_possible < rem:
                return None
        else:
            # Approximation for large terms_left (not expected in this problem)
            if terms_left * Fraction(1, den_lb) < rem:
                return None

        # Pruning: Min possible sum with remaining terms
        # The smallest possible terms are from the largest allowed denominators: MAX_DEN
        # If sum of these is > rem, we can't get down to rem.
        # Actually, we need sum = rem. Each term >= 1/MAX_DEN.
        # So min sum is terms_left / MAX_DEN.
        # If rem < terms_left / MAX_DEN, it's impossible.
        if rem < Fraction(terms_left, MAX_DEN):
            return None
        
        # Upper bound on next denominator:
        # We need 1/d + (sum of remaining terms) = rem
        # Max sum of remaining terms (terms_left - 1) is roughly (terms_left - 1) / (d+1)
        # So 1/d + (terms_left - 1)/(d+1) >= rem
        # This is a loose bound. A tighter one:
        # 1/d <= rem => d >= 1/rem (already used)
        # Also, the sum of the remaining terms must be positive, so 1/d < rem => d > 1/rem.
        # Furthermore, the remaining terms must sum to rem - 1/d.
        # The minimum value of the remaining terms is (terms_left - 1) / MAX_DEN.
        # So rem - 1/d >= (terms_left - 1) / MAX_DEN
        # 1/d <= rem - (terms_left - 1) / MAX_DEN
        # d >= 1 / (rem - (terms_left - 1) / MAX_DEN)
        # This gives a potentially tighter lower bound.
        rem_minus_min = rem - Fraction(terms_left - 1, MAX_DEN)
        if rem_minus_min > 0:
            den_lb_tighter = math.ceil(rem_minus_min.denominator / rem_minus_min.numerator)
            den_lb = max(den_lb, den_lb_tighter)
            
            if den_lb > MAX_DEN:
                return None

        # Upper bound on d from the requirement that the remaining terms can sum to rem - 1/d.
        # Max sum of remaining terms is roughly (terms_left - 1) / (d + 1).
        # So rem - 1/d <= (terms_left - 1) / (d + 1)
        # (rem - 1/d)(d + 1) <= terms_left - 1
        # rem*d + rem - 1 - 1/d <= terms_left - 1
        # rem*d + rem <= terms_left
        # d <= (terms_left - rem) / rem
        # This bound is valid if rem < terms_left (which is true).
        den_ub_frac = (terms_left - rem.numerator / rem.denominator) / (rem.numerator / rem.denominator)
        # Using Fraction for precision:
        # d * rem <= terms_left - rem  => d <= (terms_left - rem) / rem
        # (terms_left - rem) / rem = terms_left/rem - 1
        den_ub = min(MAX_DEN, math.floor((terms_left / rem) - 1))
        
        if den_lb > den_ub:
            return None

        # Iterate candidates
        # To find a solution quickly, we can iterate from den_lb upwards.
        # The search space is small for k=8.
        for den in range(den_lb, den_ub + 1):
            if time.time() > deadline:
                return None
            if den in used:
                continue
            
            term = Fraction(1, den)
            new_rem = rem - term
            
            # Quick check: new_rem must be achievable
            if new_rem < 0:
                break # Since den is increasing, 1/den is decreasing, new_rem is increasing. 
                      # Wait, if den increases, 1/den decreases, so rem - 1/den increases.
                      # So if new_rem < 0, it will become less negative. 
                      # But we require new_rem >= 0. 
                      # Actually, if 1/den > rem, then new_rem < 0. 
                      # As den increases, 1/den decreases, so new_rem increases towards rem.
                      # So we should continue, not break, unless we know it won't work.
                      # However, we have the bound den_ub which ensures 1/den < rem roughly.
                      # Let's just check if new_rem is 0 or positive.
            
            if new_rem == 0:
                if terms_left - 1 == 0:
                    return [den]
                else:
                    # We have terms left but rem is 0. Invalid.
                    continue
            
            if new_rem < 0:
                continue

            result = dfs(new_rem, den + 1, terms_left - 1, used | {den})
            if result is not None:
                return [den] + result
        
        return None

    # Try k from 8 upward
    # We established k < 8 is impossible.
    for k in range(8, 15):
        if time.time() > deadline:
            break
        result = dfs(Fraction(1, 1), 5, k, frozenset())
        if result is not None:
            # Verify solution
            s = sum(Fraction(1, d) for d in result)
            if s == 1 and len(set(result)) == len(result) and all(5 <= d <= MAX_DEN for d in result):
                return result
    
    # Fallback: If DFS fails (unlikely), return a valid known solution if hardcoded, 
    # or try a less optimized search. 
    # For safety, we can try k=9, 10 with a simpler approach if the above fails.
    # But k=8 should be found quickly.
    
    raise RuntimeError("no valid representation found")

# EVOLVE-BLOCK-END
