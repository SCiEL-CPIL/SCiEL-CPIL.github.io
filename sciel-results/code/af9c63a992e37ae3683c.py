"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from fractions import Fraction
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 85  # safety margin before 90s kill
    
    def verify(sol):
        if not sol:
            return False
        if len(set(sol)) != len(sol):
            return False
        if any(d < 5 or d > 10000000 for d in sol):
            return False
        s = sum(Fraction(1, d) for d in sol)
        return s == 1
    
    # Try to find minimal solutions by increasing count
    # We know 8 is the minimum possible (5,6,7,8,9,10,11,12 sum > 1)
    
    # Search for 8-term solutions using backtracking with pruning
    # We need 1/a + 1/b + 1/c + 1/d + 1/e + 1/f + 1/g + 1/h = 1
    # with 5 <= a < b < c < d < e < f < g < h
    
    def search_k_terms(k, min_d=5):
        """Find k distinct denominators >= min_d summing to 1"""
        if k == 1:
            # Need 1/d = 1, so d = 1, but min_d >= 5, impossible
            return None
        if k == 2:
            # 1/a + 1/b = 1 => (a-1)(b-1) = 1 => a=b=2, impossible with min_d>=5
            return None
        
        # Use the identity: if we have remaining sum S = p/q and need k more terms
        # Start with S = 1/1
        return _search(1, 1, k, min_d, [])
    
    def _search(num, den, k, min_d, path):
        """Find k distinct denominators >= min_d such that sum of 1/d = num/den"""
        if time.time() > deadline:
            return None
        if k == 0:
            if num == den:
                return path[:]
            return None
        if k == 1:
            # Need 1/d = num/den => d = den/num
            if den % num == 0:
                d = den // num
                if d >= min_d and d > (path[-1] if path else min_d - 1):
                    return path + [d]
            return None
        
        # Pruning: maximum possible sum with k terms starting from min_d
        # is 1/min_d + 1/(min_d+1) + ... (k terms)
        max_sum = sum(Fraction(1, min_d + i) for i in range(k))
        if max_sum < Fraction(num, den):
            return None
        
        # Minimum possible sum: 1/(min_d) + 1/(min_d+1) + ... but we can go to infinity
        # Actually min sum approaches 0, so no lower bound pruning easily
        
        # Try each possible next denominator
        # d must satisfy: 1/d <= num/den (since remaining k-1 terms are positive)
        # So d >= den/num
        # Also d >= min_d
        # And 1/d > num/den - (k-1)/min_d... complex, use simpler bound
        # d must be > last chosen
        start = max(min_d, (den + num - 1) // num)  # ceil(den/num)
        
        # Upper bound on d: we need 1/d + (k-1)/(d+1) >= num/den (rough)
        # Simpler: 1/d >= num/den - (k-1)/(d+1) is hard. Use: 1/d > num/den / k is too loose
        # Better: remaining k-1 terms each <= 1/(d+1), so 1/d + (k-1)/(d+1) >= num/den
        # This gives an upper bound on d
        
        # Try d from start upward, with a reasonable limit
        # The largest d we'd try: if k=2, d can be large. For k>=3, bound is tighter.
        # Use: 1/d >= num/den - (k-1)/5 (since other terms <= 1/5)
        # So d <= den / (num - den*(k-1)/5) if denominator positive
        
        # Just iterate with a cap
        max_d = 10000000
        # Tighter: 1/d + (k-1)/(d+1) >= num/den
        # For large d, LHS ≈ k/d, so d <= k*den/num
        d_upper = min(max_d, k * den // num + k + 10)
        
        for d in range(start, d_upper + 1):
            if time.time() > deadline:
                return None
            if den % d == 0 and num % d == 0:
                # 1/d = num/d * den/d... let's compute new remainder
                # new_num/den - 1/d = (d*num - den)/(d*den)
                new_num = d * num - den
                new_den = d * den
                if new_num <= 0:
                    continue
                # Reduce
                from math import gcd
                g = gcd(new_num, new_den)
                rn, rd = new_num // g, new_den // g
                result = _search(rn, rd, k - 1, d + 1, path + [d])
                if result is not None:
                    return result
            else:
                # 1/d doesn't divide evenly, but we can still use fractions
                # new remainder: num/den - 1/d = (d*num - den) / (d*den)
                new_num = d * num - den
                if new_num <= 0:
                    continue
                new_den = d * den
                from math import gcd
                g = gcd(new_num, new_den)
                rn, rd = new_num // g, new_den // g
                result = _search(rn, rd, k - 1, d + 1, path + [d])
                if result is not None:
                    return result
        
        return None
    
    # Try increasing k from 8
    for k in range(8, 13):
        sol = search_k_terms(k)
        if sol is not None and verify(sol):
            return sol
    
    # Fallback: use a known valid solution constructed via splitting
    # 1 = 1/2 + 1/3 + 1/6
    # Split 1/2: 1/2 = 1/5 + 1/6 + 1/15 (verify: 1/5+1/6+1/15 = 6/30+5/30+2/30=13/30 ≠ 1/2)
    # Let me compute properly:
    # 1/2 - 1/5 = 3/10
    # 3/10 = 1/4 + 1/20 (4<5, bad)
    # 3/10 = 1/6 + 1/15? 1/6+1/15 = 5/30+2/30=7/30 ≠ 3/10=9/30
    # 3/10 - 1/6 = 18/60-10/60=8/60=2/15
    # 2/15 = 1/8 + 1/60? 1/8+1/60=15/120+2/120=17/120 ≠ 2/15=16/120
    # 2/15 = 1/9 + 1/45? 1/9+1/45=5/45+1/45=6/45=2/15 ✓
    # So 1/2 = 1/5 + 1/6 + 1/9 + 1/45
    # Verify: 1/5+1/6+1/9+1/45 = 9/45+7.5/45... let me use LCM=90
    # 18/90+15/90+10/90+2/90=45/90=1/2 ✓
    
    # Split 1/3: 1/3 - 1/7 = 4/21
    # 4/21 - 1/6 = 24/126-21/126=3/126=1/42
    # So 1/3 = 1/7 + 1/6 + 1/42? 1/7+1/6+1/42 = 6/42+7/42+1/42=14/42=1/3 ✓
    # But 6 already used!
    
    # 1/3 - 1/8 = 5/24
    # 5/24 - 1/6 = 5/24-4/24=1/24
    # 1/3 = 1/8 + 1/6 + 1/24? 3/24+4/24+1/24=8/24=1/3 ✓ but 6 used
    
    # 1/3 - 1/10 = 7/30
    # 7/30 - 1/5 = 7/30-6/30=1/30
    # 1/3 = 1/10 + 1/5 + 1/30? 3/30+6/30+1/30=10/30=1/3 ✓ but 5 used
    
    # 1/3 - 1/11 = 2/33
    # 2/33 - 1/12 = 8/132-11/132=-3/132 < 0
    # 2/33 = 1/17 + ... 2/33-1/17=(34-33)/561=1/561
    # 1/3 = 1/11 + 1/17 + 1/561? Let me verify: LCM of 3,11,17,561
    # 561=3*11*17. 1/3=187/561, 1/11=51/561, 1/17=33/561, 1/561=1/561
    # Sum=187+51+33+1=272/561 ≠ 561/561. Wrong.
    # 1/3 - 1/11 = (11-3)/33=8/33 not 2/33. Let me recompute.
    # 1/3 - 1/11 = (11-3)/33 = 8/33
    # 8/33 - 1/5 = 40/165-33/165=7/165
    # 7/165 - 1/24 = 168/3960-165/3960=3/3960=1/1320
    # So 1/3 = 1/11 + 1/5 + 1/24 + 1/1320? But 5 used.
    
    # This is getting complex. Let me just use a verified 9-term or 10-term solution.
    
    # Known: 1 = 1/5 + 1/6 + 1/7 + 1/10 + 1/14 + 1/21 + 1/30 + 1/35 + 1/42 + 1/70 + 1/105 + 1/210
    # Let me check sum with LCM 210:
    # 42+35+30+21+15+10+7+6+5+3+2+1 = 177/210 ≠ 1
    
    # Let me just return the backtracking result or a safe fallback
    # Safe fallback: use many terms
    # 1 = sum of 1/d for d in {5,6,7,10,14,15,21,30,35,42,70,105,210} plus adjustments
    # Sum = (42+35+30+21+15+14+10+7+6+5+3+2+1)/210 = 191/210
    # Need 19/210 more
    # 19/210 = 1/11 + ... 19/210 - 1/11 = (209-210)/2310 = -1/2310 < 0
    # 19/210 = 1/12 + ... 19/210 - 1/12 = (114-35)/1260... let me compute: 19/210=114/1260, 1/12=105/1260, diff=9/1260=1/140
    # So 19/210 = 1/12 + 1/140
    # Check: 1/12+1/140 = 35/420+3/420=38/420=19/210 ✓
    # So total: {5,6,7,10,12,14,15,21,30,35,42,70,105,140,210}
    # That's 15 terms. All distinct, all >= 5.
    
    fallback = [5, 6, 7, 10, 12, 14, 15, 21, 30, 35, 42, 70, 105, 140, 210]
    if verify(fallback):
        return fallback
    
    # Even safer: construct via repeated splitting
    # Start with 1 = 1/2 + 1/3 + 1/6
    # Split 1/2 into terms with denom >= 5, avoiding {3,6}
    # 1/2 = 1/5 + 3/10; 3/10 = 1/7 + 11/70; 11/70 = 1/8 + 1/56
    # So 1/2 = 1/5 + 1/7 + 1/8 + 1/56
    # Split 1/3: 1/3 = 1/11 + 2/33; 2/33 = 1/17 + 1/561
    # So 1/3 = 1/11 + 1/17 + 1/561
    # Split 1/6: 1/6 = 1/7 + 1/42
    # Total: 1/5 + 1/7 + 1/8 + 1/56 + 1/11 + 1/17 + 1/561 + 1/7 + 1/42
    # Duplicate 7! 
    
    # Just return the verified 15-term solution
    return fallback
# EVOLVE-BLOCK-END
