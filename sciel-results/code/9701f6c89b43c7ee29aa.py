import time
from fractions import Fraction

def construct():
    start_time = time.time()
    deadline = start_time + 80.0
    
    # Based on analysis, k=8 is impossible.
    # We search for k=9.
    # d1=5, d2=6, d3=7, d4=8, d5=9, d6=10, d7=11 are forced.
    # Remaining sum R = 1 - (1/5 + 1/6 + 1/7 + 1/8 + 1/9 + 1/10 + 1/11)
    # We need 2 denominators d8, d9 > 11 such that 1/d8 + 1/d9 = R.
    
    # Calculate R
    R = Fraction(1)
    for d in range(5, 12):
        R -= Fraction(1, d)
        
    # R = 1/d8 + 1/d9
    # d9 = 1 / (R - 1/d8)
    # d8 must be > 11
    # Also 1/d8 < R => d8 > 1/R
    # And 1/d8 + 1/(d8+1) >= R (max sum for 2 terms starting at d8)
    
    lower_d8 = max(12, int(1 / float(R)) + 1)
    
    # Upper bound: 1/d8 + 1/(d8+1) >= R
    # Approx 2/d8 >= R => d8 <= 2/R
    upper_d8 = int(2 / float(R)) + 2
    
    found = None
    for d8 in range(lower_d8, upper_d8 + 1):
        if time.time() > deadline:
            break
        rem = R - Fraction(1, d8)
        if rem <= 0:
            continue
        num, den = rem.numerator, rem.denominator
        if den % num == 0:
            d9 = den // num
            if d9 > d8:
                found = [5, 6, 7, 8, 9, 10, 11, d8, d9]
                break
    
    if found:
        # Verify
        s = sum(Fraction(1, x) for x in found)
        if s == 1 and len(set(found)) == len(found) and all(5 <= x <= 10000000 for x in found):
            return found

    # Fallback: General DFS for k=10 if k=9 fails (unlikely) or if time permits
    # But k=9 should be fast.
    # If k=9 not found, try k=10 with a short time limit.
    
    def find_solution_k(k, time_limit=5.0):
        def dfs(remaining, k_left, min_den, current):
            if time.time() > start_time + time_limit:
                return None
            if k_left == 0:
                return current[:] if remaining == 0 else None
            if k_left == 1:
                if remaining > 0:
                    num, den = remaining.numerator, remaining.denominator
                    if den % num == 0:
                        x = den // num
                        if x > min_den and x <= 10000000:
                            return current + [x]
                return None
            
            lower_a = max(min_den, int(1 / float(remaining)) + 1)
            upper_a = min(10000000, int(k_left / float(remaining)) + 2)
            
            if lower_a > upper_a:
                return None
            
            for a in range(lower_a, upper_a + 1):
                if time.time() > start_time + time_limit:
                    return None
                # Prune
                max_possible = 0.0
                for i in range(k_left):
                    max_possible += 1.0 / (a + i)
                if max_possible < float(remaining) - 1e-15:
                    break
                
                new_remaining = remaining - Fraction(1, a)
                if new_remaining <= 0:
                    continue
                res = dfs(new_remaining, k_left - 1, a + 1, current + [a])
                if res:
                    return res
            return None
        
        return dfs(Fraction(1), k, 5, [])

    # Try k=10
    if time.time() < deadline - 10:
        sol = find_solution_k(10, 5.0)
        if sol:
            return sol

    # Greedy fallback
    def greedy_solution():
        remaining = Fraction(1)
        denoms = []
        last = 4
        while remaining > 0:
            d = last + 1
            while Fraction(1, d) > remaining:
                d += 1
            denoms.append(d)
            remaining -= Fraction(1, d)
            last = d
            if len(denoms) > 256:
                break
        return denoms

    sol = greedy_solution()
    s = sum(Fraction(1, x) for x in sol)
    if s == 1 and len(set(sol)) == len(sol) and all(5 <= x <= 10000000 for x in sol):
        return sol

    raise RuntimeError("No solution found")
