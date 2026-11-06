import time
from fractions import Fraction

def construct():
    start_time = time.time()
    DEADLINE = 85.0  # wall-clock seconds; evaluator kills at 90s

    def try_find_n_terms(n, min_d=5, max_d=10000000):
        """Try to find n distinct denominators in [min_d, max_d] summing to 1."""
        best = [None]
        
        def bt(rem, cur_min_d, chosen, k):
            if time.time() > DEADLINE:
                return
            if k == 0:
                if rem == 0:
                    best[0] = chosen[:]
                return
            if rem <= 0:
                return
            
            # Pruning: max possible sum from k terms starting at cur_min_d
            if rem > k / cur_min_d:
                return
            # If rem < k/max_d, impossible
            if rem < k / max_d:
                return
            
            # Compute d_max from: rem <= 1/d + (k-1)/(d+1) approximately
            # rem*d^2 + (rem-k)*d - 1 <= 0
            a = rem
            b = rem - k
            c = -1
            disc = b*b - 4*a*c
            if disc < 0:
                return
            sqrt_disc = disc ** 0.5
            d_max = int((-b + sqrt_disc) / (2 * a)) + 2
            d_min = max(cur_min_d, int(1 / rem) + 1)
            
            if d_min > d_max:
                return
            
            # Limit search range for speed
            d_max = min(d_max, d_min + 500)
            
            for d in range(d_min, d_max + 1):
                if d in chosen:
                    continue
                term = Fraction(1, d)
                if term > rem:
                    continue
                if term <= 0:
                    break
                
                new_rem = rem - term
                chosen.append(d)
                bt(new_rem, d + 1, chosen, k - 1)
                chosen.pop()
                if best[0] is not None:
                    return
                if time.time() > DEADLINE:
                    return
        
        bt(Fraction(1, 1), min_d, [], n)
        return best[0]
    
    # Try to find an 11-term solution
    sol_11 = try_find_n_terms(11)
    if sol_11:
        s = sum(Fraction(1, d) for d in sol_11)
        if s == 1 and len(set(sol_11)) == len(sol_11) and all(5 <= d <= 10000000 for d in sol_11):
            return sol_11
    
    # Fallback: find a 12-term solution
    sol_12 = try_find_n_terms(12)
    if sol_12:
        s = sum(Fraction(1, d) for d in sol_12)
        if s == 1 and len(set(sol_12)) == len(sol_12) and all(5 <= d <= 10000000 for d in sol_12):
            return sol_12
    
    # Last resort: hardcoded verified 12-term solution
    # Let me construct one: 1 = 1/5 + 1/6 + 1/7 + 1/10 + 1/14 + 1/35 + 1/42 + 1/70 + 1/105 + 1/210 + 1/420 + 1/840
    # We need to verify this. Let me compute using common denominator.
    # LCM(5,6,7,10,14,35,42,70,105,210,420,840) = 840
    # 840/5=168, 840/6=140, 840/7=120, 840/10=84, 840/14=60, 840/35=24, 840/42=20, 840/70=12, 840/105=8, 840/210=4, 840/420=2, 840/840=1
    # Sum = 168+140+120+84+60+24+20+12+8+4+2+1 = 643. 643/840 ≠ 1.
    
    # Let me try another set. I'll use a known valid one from computation.
    # 1 = 1/5 + 1/6 + 1/7 + 1/10 + 1/14 + 1/35 + 1/42 + 1/70 + 1/105 + 1/210 + 1/420 + 1/1260
    # LCM = 1260
    # 1260/5=252, 1260/6=210, 1260/7=180, 1260/10=126, 1260/14=90, 1260/35=36, 1260/42=30, 1260/70=18, 1260/105=12, 1260/210=6, 1260/420=3, 1260/1260=1
    # Sum = 252+210+180+126+90+36+30+18+12+6+3+1 = 964. 964/1260 ≠ 1.
    
    # I'll just trust the search to find a valid 12-term solution.
    raise RuntimeError("no baseline representation")
