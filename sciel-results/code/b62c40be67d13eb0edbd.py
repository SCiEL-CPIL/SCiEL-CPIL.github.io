"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from fractions import Fraction

def construct():
    start = time.time()
    deadline = start + 55

    # Known minimal solution: 5 distinct denominators >= 5
    # 1/5 + 1/6 + 1/10 + 1/12 + 1/15 = 1
    # Verification:
    # LCM(5,6,10,12,15) = 60
    # 12 + 10 + 6 + 5 + 4 = 37... wait, that's not 60
    # Let me recalculate:
    # 1/5 = 12/60
    # 1/6 = 10/60
    # 1/10 = 6/60
    # 1/12 = 5/60
    # 1/15 = 4/60
    # Sum = 12+10+6+5+4 = 37/60. Not 1.
    
    # Try: 1/5 + 1/6 + 1/7 + 1/42
    # LCM = 420
    # 84 + 70 + 60 + 10 = 224. Not 420.
    
    # Try: 1/5 + 1/6 + 1/7 + 1/10 + 1/14 + 1/35
    # LCM = 210
    # 42+35+30+21+15+6 = 149. Not 210.
    
    # Try: 1/5 + 1/6 + 1/7 + 1/10 + 1/14 + 1/15 + 1/21
    # LCM = 210
    # 42+35+30+21+15+14+10 = 167. Not 210.
    
    # Let me use a proper search.
    # For 4 terms: 1/a + 1/b + 1/c + 1/d = 1 with a,b,c,d >= 5 distinct
    # Try a=5: 1/b + 1/c + 1/d = 4/5
    # b=6: 1/c + 1/d = 4/5 - 1/6 = 24/30 - 5/30 = 19/30
    # c=7: 1/d = 19/30 - 1/7 = 133/210 - 30/210 = 103/210. d=210/103. Not integer.
    # c=8: 1/d = 19/30 - 1/8 = 76/120 - 15/120 = 61/120. Not integer.
    # c=9: 1/d = 19/30 - 1/9 = 57/90 - 10/90 = 47/90. Not integer.
    # c=10: 1/d = 19/30 - 1/10 = 19/30 - 3/30 = 16/30 = 8/15. d=15/8. Not integer.
    # c=11: 1/d = 19/30 - 1/11 = 209/330 - 30/330 = 179/330. Not integer.
    # c=12: 1/d = 19/30 - 1/12 = 76/120 - 10/120 = 66/120 = 11/20. d=20/11. Not integer.
    # c=14: 1/d = 19/30 - 1/14 = 133/210 - 15/210 = 118/210 = 59/105. Not integer.
    # c=15: 1/d = 19/30 - 1/15 = 19/30 - 2/30 = 17/30. Not integer.
    # c=18: 1/d = 19/30 - 1/18 = 57/90 - 5/90 = 52/90 = 26/45. Not integer.
    # c=20: 1/d = 19/30 - 1/20 = 38/60 - 3/60 = 35/60 = 7/12. d=12/7. Not integer.
    # c=24: 1/d = 19/30 - 1/24 = 38/60 - 2.5/60... let me use LCM(30,24)=120
    # 19/30 = 76/120, 1/24 = 5/120. 71/120. Not integer.
    
    # It seems 4 terms is hard. Let me try 5 terms with a programmatic search.
    
    def find_solution(max_terms):
        def dfs(remaining, min_d, terms_left, current):
            if time.time() > deadline:
                return None
            if remaining == 0:
                return current
            if terms_left == 0:
                return None
            if remaining <= 0:
                return None
            
            # Pruning: check if max possible sum is enough
            max_possible = Fraction(0)
            for i in range(terms_left):
                max_possible += Fraction(1, min_d + i)
            if remaining > max_possible:
                return None
            
            # Lower bound for next denominator
            num, den = remaining.numerator, remaining.denominator
            lower_exact = (den + num - 1) // num
            lower = max(min_d, lower_exact)
            
            # Upper bound: we need remaining - 1/d >= 0, so 1/d <= remaining
            # Also, we need to be able to reach remaining with terms_left-1 more terms
            # The minimum additional sum is (terms_left-1) * 1/max_d
            # But max_d can be large, so this bound is weak
            # Better: 1/d < remaining, so d > 1/remaining
            # And we need remaining - 1/d to be achievable with terms_left-1 terms starting from d+1
            # For now, just use a reasonable upper bound
            
            if terms_left == 1:
                if remaining.denominator % remaining.numerator == 0:
                    d = remaining.denominator // remaining.numerator
                    if d >= min_d:
                        return current + [d]
                return None
            
            # Upper bound from the requirement that we can still reach remaining
            # with terms_left-1 terms, each <= remaining - 1/d
            # Simple bound: d < 1/remaining
            upper = int(1 / float(remaining)) + 1
            if upper < lower:
                return None
            # Cap to keep search tractable
            if upper > 100000:
                upper = 100000
            
            for d in range(lower, upper + 1):
                frac = Fraction(1, d)
                if frac >= remaining:
                    break
                new_remaining = remaining - frac
                if new_remaining <= 0:
                    continue
                result = dfs(new_remaining, d + 1, terms_left - 1, current + [d])
                if result is not None:
                    return result
            return None
        
        return dfs(Fraction(1, 1), 5, max_terms, [])

    # Try to find minimal solution
    for k in range(4, 12):
        if time.time() > deadline:
            break
        result = find_solution(k)
        if result is not None:
            # Verify
            if len(set(result)) == len(result) and all(5 <= d <= 10000000 for d in result):
                total = sum(Fraction(1, d) for d in result)
                if total == 1:
                    return result

    # Fallback: known valid solution
    # 1/5 + 1/6 + 1/7 + 1/10 + 1/12 + 1/14 + 1/15 + 1/21 + 1/35
    # Let me just return a verified one if search fails
    # For now, raise if not found
    raise RuntimeError("no solution found")
# EVOLVE-BLOCK-END
