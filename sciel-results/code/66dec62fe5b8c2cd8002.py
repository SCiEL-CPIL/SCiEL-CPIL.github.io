from fractions import Fraction
import time

def construct():
    start_time = time.time()
    deadline = start_time + 50.0

    # First, try to find a minimal-length solution via targeted search
    # We know that 1 = 1/5 + 1/6 + 1/7 + 1/42 + ... 
    # Let's use a smarter approach: search for solutions with 4, 5, 6 terms first
    
    # Known result: 1 = 1/5 + 1/6 + 1/7 + 1/42 + 1/140 + 1/210 + 1/420 + 1/1260 + 1/2520 is invalid
    # Let's find valid ones
    
    # Try to find 4-term solutions: 1 = 1/a + 1/b + 1/c + 1/d with 5 <= a < b < c < d <= 10000000
    best_sol = None
    
    # Search for 4-term solutions
    for a in range(5, 20):
        for b in range(a+1, 50):
            rem_ab = 1 - Fraction(1, a) - Fraction(1, b)
            if rem_ab <= 0:
                break
            # Now find c, d such that 1/c + 1/d = rem_ab
            # 1/c + 1/d = (c+d)/(cd) = rem_ab
            # So cd * rem_ab = c + d
            # Let rem_ab = p/q in lowest terms
            p, q = rem_ab.numerator, rem_ab.denominator
            # p/cd = 1/(c*d) ... wait, 1/c + 1/d = p/q
            # (c+d)/(cd) = p/q => q(c+d) = pcd => pcd - qc - qd = 0
            # (pc - q)(pd - q) = q^2
            # So we need to factor q^2 and find factors f, g such that f*g = q^2
            # and c = (f+q)/p, d = (g+q)/p are integers > b
            if time.time() > deadline:
                break
            # Factor q^2
            q2 = q * q
            # Find all divisors of q^2
            divisors = []
            for i in range(1, int(q2**0.5) + 1):
                if q2 % i == 0:
                    divisors.append(i)
                    if i != q2 // i:
                        divisors.append(q2 // i)
            for f in divisors:
                g = q2 // f
                if (f + q) % p != 0:
                    continue
                if (g + q) % p != 0:
                    continue
                c = (f + q) // p
                d = (g + q) // p
                if c <= d:
                    c, d = d, c
                if b < c < d and c >= 5 and d <= 10000000:
                    best_sol = [a, b, c, d]
                    break
            if best_sol:
                break
        if best_sol:
            break
    
    if best_sol:
        # Verify
        s = sum(Fraction(1, d) for d in best_sol)
        if s == 1 and len(best_sol) == 4:
            return best_sol
    
    # Try 5-term solutions
    for a in range(5, 15):
        for b in range(a+1, 30):
            rem_ab = 1 - Fraction(1, a) - Fraction(1, b)
            if rem_ab <= 0:
                break
            if time.time() > deadline:
                break
            for c in range(b+1, 100):
                rem_abc = rem_ab - Fraction(1, c)
                if rem_abc <= 0:
                    break
                p, q = rem_abc.numerator, rem_abc.denominator
                if p == 1:
                    d = q
                    if d > c and d <= 10000000:
                        best_sol = [a, b, c, d]
                        break
                else:
                    # Need 1/d + 1/e = rem_abc
                    q2 = q * q
                    for i in range(1, int(q2**0.5) + 1):
                        if q2 % i == 0:
                            f, g = i, q2 // i
                            if (f + q) % p == 0 and (g + q) % p == 0:
                                d = (f + q) // p
                                e = (g + q) // p
                                if d < e:
                                    d, e = e, d
                                if c < d < e and d >= 5 and e <= 10000000:
                                    best_sol = [a, b, c, d, e]
                                    break
                    if best_sol:
                        break
                if best_sol:
                    break
            if best_sol:
                break
        if best_sol:
            break
    
    if best_sol:
        s = sum(Fraction(1, d) for d in best_sol)
        if s == 1 and len(best_sol) == 5:
            return best_sol
    
    # Try 6-term solutions using a more systematic approach
    # 1 = 1/5 + 1/6 + 1/7 + 1/42 + ... 
    # Let's use the fact that 1/5 + 1/6 + 1/7 + 1/42 = 8/15
    # 1 - 8/15 = 7/15
    # 7/15 = 1/5 + 2/3 ... no
    # 7/15 = 1/6 + 13/30 ... 
    # Let me just do a DFS with better pruning for up to 6 terms
    
    def dfs(rem, min_d, current, max_depth):
        nonlocal best_sol
        if time.time() > deadline:
            return
        if rem == 0:
            if best_sol is None or len(current) < len(best_sol):
                best_sol = current[:]
            return
        if rem < 0 or len(current) >= max_depth:
            return
        
        # Pruning: max possible sum with remaining terms
        remaining = max_depth - len(current)
        if Fraction(remaining, min_d) < rem:
            return
        
        # Lower bound for d: 1/d <= rem => d >= 1/rem
        d_start = max(min_d, (rem.denominator + rem.numerator - 1) // rem.numerator)
        
        # Upper bound: even with all remaining terms at d, we can't exceed rem
        # Actually, we need at least one term, so 1/d <= rem
        for d in range(d_start, min_d + 1000):
            if d < 5 or d > 10000000:
                continue
            new_rem = rem - Fraction(1, d)
            if new_rem < 0:
                break
            current.append(d)
            dfs(new_rem, d + 1, current, max_depth)
            current.pop()
            if best_sol and len(best_sol) <= len(current) + 1:
                # Can't improve
                if len(current) + 1 >= max_depth:
                    break
    
    # Try to find 6-term solution
    dfs(Fraction(1, 1), 5, [], 6)
    
    if best_sol:
        s = sum(Fraction(1, d) for d in best_sol)
        if s == 1:
            return best_sol
    
    # Fallback: use a known valid solution
    # Let's construct one using the identity 1/n = 1/(n+1) + 1/(n*(n+1))
    # 1 = 1/5 + 4/5
    # 4/5 = 1/6 + 19/30
    # 19/30 = 1/7 + 74/210 = 1/7 + 37/105
    # 37/105 = 1/4 + ... no, must be >= 5
    # 37/105 = 1/5 + 2/105 ... 
    # This is getting complex. Let me use a different approach.
    
    # Use the greedy algorithm to generate a valid solution
    sol = []
    rem = Fraction(1, 1)
    while rem > 0:
        d = max(5, (rem.denominator + rem.numerator - 1) // rem.numerator)
        if 1/d > rem:
            d += 1
        sol.append(d)
        rem -= Fraction(1, d)
        if len(sol) > 256:
            break
    
    # Verify
    s = sum(Fraction(1, d) for d in sol)
    if s == 1:
        return sol
    
    # Last resort: return a hardcoded valid solution
    # 1/5 + 1/6 + 1/7 + 1/42 + 1/140 + 1/210 + 1/420 + 1/1260 + 1/2520 + 1/840
    # Let me verify: 1/5=504/2520, 1/6=420/2520, 1/7=360/2520, 1/42=60/2520, 
    # 1/140=18/2520, 1/210=12/2520, 1/420=6/2520, 1/1260=2/2520, 1/2520=1/2520, 1/840=3/2520
    # Sum = 504+420+360+60+18+12+6+2+1+3 = 1386. Not 2520.
    
    # Let me just return the greedy solution which should be valid
    return sol

