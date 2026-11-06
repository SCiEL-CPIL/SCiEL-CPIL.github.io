import time
from fractions import Fraction

def construct():
    start_time = time.time()
    deadline = start_time + 80.0
    
    target = Fraction(5, 31)
    
    best_solution = None
    best_max = float('inf')
    
    def check_and_update(denoms):
        nonlocal best_solution, best_max
        if len(denoms) < 1 or len(denoms) > 256:
            return
        if len(set(denoms)) != len(denoms):
            return
        if any(d < 1 or d > 10000000 for d in denoms):
            return
        s = sum(Fraction(1, d) for d in denoms)
        if s != target:
            return
        mx = max(denoms)
        if mx < best_max:
            best_max = mx
            best_solution = list(denoms)
    
    # Direct enumeration of 3-term solutions using factoring trick
    # 1/a + 1/b + 1/c = 5/31, a < b < c
    # (5a-31)(5b-31)(5c-31) approach doesn't directly work for 3 terms
    # Instead: fix a, then solve 1/b + 1/c = 5/31 - 1/a
    # Let r = 5/31 - 1/a = (5a-31)/(31a)
    # 1/b + 1/c = r => (b+c)/(bc) = r => r*bc = b+c
    # (rb-1)(rc-1) = 1
    # Actually: r*bc - b - c = 0
    # Multiply by r: r²bc - rb - rc = 0
    # (rb-1)(rc-1) = 1
    # So rb-1 and rc-1 are positive integers whose product is 1
    # That means rb-1 = 1 and rc-1 = 1, so b = c = 2/r
    # Wait, that's only if r is such that this works. Let me redo.
    #
    # r*bc = b + c
    # r*bc - b - c = 0
    # Multiply by r: r²bc - rb - rc = 0
    # Add 1: (rb-1)(rc-1) = 1
    # Since b, c > 0 and r > 0, rb > 0 and rc > 0
    # (rb-1)(rc-1) = 1 means rb-1 = 1, rc-1 = 1 OR rb-1 = -1, rc-1 = -1
    # Case 1: rb = 2, rc = 2 => b = c = 2/r
    # Case 2: rb = 0, rc = 0 => impossible since r>0, b,c>0
    #
    # So for 2 remaining terms, b = c = 2/r. But we need b ≠ c for distinct denominators!
    # Wait, b and c must be distinct. So if b = c, that's invalid.
    # Hmm, but actually the problem says distinct denominators.
    #
    # Wait, I made an error. Let me redo:
    # 1/b + 1/c = r
    # (b+c)/(bc) = r
    # b+c = r*bc
    # r*bc - b - c = 0
    # Multiply by r: r²bc - rb - rc = 0
    # Add 1: (rb-1)(rc-1) = 1
    #
    # This gives only b=c=2/r as a solution (for positive b,c). 
    # But that contradicts known solutions like 1/7+1/56+1/1736=5/31
    # Let me check: 1/7 + 1/56 = 8/56 + 1/56 = 9/56. 5/31 - 9/56 = (280-279)/1736 = 1/1736. Yes!
    # So r = 5/31 - 1/7 = (35-31)/217 = 4/217
    # 1/b + 1/c = 4/217
    # (b+c)/(bc) = 4/217
    # 4bc = 217(b+c)
    # 4bc - 217b - 217c = 0
    # Multiply by 4: 16bc - 868b - 868c = 0
    # Add 217²: (4b-217)(4c-217) = 217² = 47089
    #
    # Ah, I see my error. The general formula for 1/b + 1/c = p/q is:
    # (pb - q)(pc - q) = q²
    # Not (rb-1)(rc-1) = 1. I was confusing myself.
    #
    # For 1/b + 1/c = p/q:
    # q(b+c) = p*bc
    # p*bc - qb - qc = 0
    # Multiply by p: p²bc - pqb - pqc = 0
    # Add q²: (pb-q)(pc-q) = q²
    #
    # So for r = p/q (in lowest terms), we need factor pairs of q² where both 
    # factors are ≡ -q (mod p), i.e., ≡ q (mod p) since we're looking at pb-q.
    # Actually: pb - q must divide q², and pb - q ≡ -q (mod p).
    
    # For 3 terms: fix a, compute r = 5/31 - 1/a = (5a-31)/(31a)
    # Let p = 5a-31, q = 31a. Reduce by gcd.
    # Then find factor pairs of q² for the 2-term completion.
    
    from math import gcd
    
    found_solutions = []
    
    # a must satisfy: 1/a < 5/31 (so a > 31/5 = 6.2), so a >= 7
    # Also a < b < c, and we want to minimize max(c)
    # Upper bound on a: since a < b < c, 3/a > 5/31 (roughly), so a < 93/5 = 18.6, a <= 18
    # But more carefully: 1/a + 1/(a+1) + 1/(a+2) > 5/31 for small a
    # Let's just iterate a from 7 upward and see when it stops being possible
    
    for a in range(7, 100):
        if time.time() > deadline:
            break
        
        p_num = 5 * a - 31
        q_den = 31 * a
        
        if p_num <= 0:
            continue
        
        g = gcd(p_num, q_den)
        p = p_num // g
        q = q_den // g
        
        # Now solve 1/b + 1/c = p/q with b > a, c > b
        # (pb - q)(pc - q) = q²
        # Let u = pb - q, v = pc - q, so u*v = q²
        # b = (u + q) / p, c = (v + q) / p
        # Need: u*v = q², b > a, c > b, b and c integers
        
        q2 = q * q
        
        # Find all divisors of q²
        # For each divisor u of q², v = q²/u
        # Check if (u + q) % p == 0 and (v + q) % p == 0
        # And b = (u+q)/p > a, c = (v+q)/p > b
        
        # Enumerate divisors of q²
        # q can be large, so we need to be efficient
        # Only iterate up to sqrt(q²) = q
        
        import math
        sqrt_q2 = int(math.isqrt(q2))
        
        for u in range(1, sqrt_q2 + 1):
            if time.time() > deadline:
                break
            if q2 % u != 0:
                continue
            v = q2 // u
            
            # Check both (u,v) and (v,u) orderings
            for u_test, v_test in [(u, v), (v, u)]:
                if (u_test + q) % p != 0:
                    continue
                if (v_test + q) % p != 0:
                    continue
                b = (u_test + q) // p
                c = (v_test + q) // p
                
                if b <= a or c <= b:
                    continue
                if b > 10000000 or c > 10000000:
                    continue
                
                # Verify
                if Fraction(1, a) + Fraction(1, b) + Fraction(1, c) == target:
                    found_solutions.append((a, b, c))
                    mx = max(a, b, c)
                    if mx < best_max:
                        best_max = mx
                        best_solution = [a, b, c]
    
    # If no 3-term solution found, try 4-term solutions
    if best_solution is None:
        # Fallback to known solution
        best_solution = [7, 56, 1736]
        best_max = 1736
    elif best_max >= 1736:
        # Try to find better with 4 terms using DFS
        def dfs(remaining, min_d, max_allowed, current, depth=0, max_depth=4):
            nonlocal best_solution, best_max
            if time.time() > deadline:
                return
            if remaining == 0:
                if len(current) >= 1:
                    check_and_update(current)
                return
            if remaining < 0:
                return
            if len(current) >= max_depth:
                return
            
            d_lo = (remaining.denominator + remaining.numerator - 1) // remaining.numerator
            if d_lo < min_d:
                d_lo = min_d
            if d_lo > max_allowed:
                return
            
            k_remaining = max_depth - len(current) - 1
            
            if k_remaining == 0:
                if remaining.numerator == 1:
                    d_exact = remaining.denominator
                    if d_exact >= d_lo and d_exact <= max_allowed:
                        dfs(Fraction(0), d_exact + 1, max_allowed, current + [d_exact], depth + 1, max_depth)
                return
            
            d_upper = min(max_allowed, d_lo + 200)
            
            for d in range(d_lo, d_upper + 1):
                if time.time() > deadline:
                    return
                if d > max_allowed:
                    break
                new_remaining = remaining - Fraction(1, d)
                if new_remaining < 0:
                    break
                if new_remaining > 0 and k_remaining > 0:
                    if new_remaining > Fraction(k_remaining, max_allowed):
                        continue
                dfs(new_remaining, d + 1, max_allowed, current + [d], depth + 1, max_depth)
        
        dfs(target, 7, best_max - 1, [], 0, 4)
    
    if best_solution is None:
        best_solution = [7, 56, 1736]
    
    return best_solution
