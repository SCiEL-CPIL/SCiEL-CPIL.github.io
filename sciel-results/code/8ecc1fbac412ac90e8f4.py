"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math
from fractions import Fraction
from itertools import combinations

def construct():
    """
    Find 1..256 distinct denominators in [1, 10000000] summing reciprocally to exactly 5/31.
    Minimize the LARGEST denominator.
    """
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin before 90s kill

    target = Fraction(5, 31)
    best_max_den = float('inf')
    best_solution = None

    # We will search for solutions with a maximum denominator M.
    # We want to find the smallest M.
    # The greedy solution gives an upper bound. Let's compute it first.
    r = target
    greedy_vals = []
    while r > 0:
        d = math.ceil(r.denominator / r.numerator)
        if d <= 0:
            break
        greedy_vals.append(d)
        r -= Fraction(1, d)
    
    if r == 0 and len(greedy_vals) <= 256 and max(greedy_vals) <= 10000000:
        best_max_den = max(greedy_vals)
        best_solution = greedy_vals[:]
    
    # If greedy fails or is not valid, we need a fallback, but greedy usually works for 5/31.
    # 5/31 = 1/8 + 1/248 + 1/... 
    # 5/31 - 1/8 = (40-31)/248 = 9/248
    # 9/248 -> ceil(248/9) = 28. 1/28.
    # 9/248 - 1/28 = (9*28 - 248) / (248*28) = (252-248)/6944 = 4/6944 = 1/1736
    # So 5/31 = 1/8 + 1/28 + 1/1736.
    # Max denom is 1736.
    
    # Can we do better?
    # We want to minimize the max denominator.
    # Let's try to find a representation with max denominator < 1736.
    # We can use a meet-in-the-middle or a bounded search.
    
    # Since the target is fixed, we can try to search for solutions with max denominator M
    # for M from some lower bound up to best_max_den - 1.
    
    # Lower bound:
    # The sum of reciprocals of 1..M is approx ln(M) + gamma.
    # We need sum = 5/31 ~ 0.161.
    # Even with many terms, the max denominator is constrained.
    # A simple lower bound: if we use k terms, the largest term 1/d_max must be <= 5/31.
    # Also, the sum of the smallest k distinct reciprocals 1/1 + ... + 1/k is the max possible sum for k terms with max denom k.
    # But we want to minimize the max denominator, so we want to use small denominators?
    # No, we want to minimize the LARGEST denominator.
    # So we want to use denominators as small as possible?
    # Actually, using smaller denominators increases the sum.
    # To minimize the max denominator, we should try to use denominators that are not too large.
    # But we are constrained by the sum being exactly 5/31.
    
    # Let's try to find a solution with max denominator < 1736.
    # We can iterate M from 1 upwards, and check if 5/31 can be represented as sum of distinct reciprocals of integers in [1, M].
    # This is a subset sum problem on rationals.
    
    # A more efficient way:
    # Iterate over the number of terms? No, term count is not fixed.
    
    # Let's try a DFS with pruning, limiting the max denominator to M.
    # We want to find the smallest M such that a solution exists.
    
    # We can binary search on M? Or just iterate M from 1 to best_max_den - 1.
    # But checking existence for a given M is NP-hard in general, but for small M it might be fast.
    # However, M could be up to 1735.
    
    # Alternative:
    # Use the fact that the denominators must divide something?
    # No, Egyptian fractions don't have to have a common denominator.
    
    # Let's try a specific search strategy:
    # 1. Pick the largest denominator d_max.
    # 2. The remaining sum is 5/31 - 1/d_max.
    # 3. This remaining sum must be represented by distinct denominators < d_max.
    
    # We can iterate d_max from 1 to 1735.
    # For each d_max, check if 5/31 - 1/d_max can be represented by distinct denominators in [1, d_max-1].
    # This is recursive.
    
    # To speed this up, we can precompute or use a memoized search.
    # But the state space is large.
    
    # Let's try a simpler heuristic first:
    # Try to find a 2-term or 3-term solution with small max denominator.
    # 2-term: 1/a + 1/b = 5/31 => 31(a+b) = 5ab => 5ab - 31a - 31b = 0 => (5a-31)(5b-31) = 31^2 = 961.
    # Factors of 961: 1, 31, 961. (31 is prime)
    # 5a - 31 = 1 => 5a = 32 => a = 6.4 (no)
    # 5a - 31 = 31 => 5a = 62 => a = 12.4 (no)
    # 5a - 31 = 961 => 5a = 992 => a = 198.4 (no)
    # So no 2-term solution.
    
    # 3-term: 1/a + 1/b + 1/c = 5/31.
    # We can iterate a from 1 to 1735.
    # Then 1/b + 1/c = 5/31 - 1/a = (5a-31)/(31a).
    # Let R = (5a-31)/(31a).
    # We need 1/b + 1/c = R.
    # This is the 2-term problem for R.
    # (b - 1/R)(c - 1/R) = (1/R)^2.
    # Or simply: 1/b + 1/c = p/q => (pb-q)(pc-q) = q^2.
    # Here p = 5a-31, q = 31a.
    # We need to check if q^2 has factors that yield integer b, c.
    # q^2 = (31a)^2.
    # This is doable for a up to 1735.
    
    # Let's implement this 3-term search.
    
    def check_2_term(p, q):
        """Check if 1/b + 1/c = p/q has distinct integer solutions b, c > 0."""
        # (pb - q)(pc - q) = q^2
        # Let x = pb - q, y = pc - q.
        # x * y = q^2.
        # x and y must be divisors of q^2.
        # Also x ≡ -q (mod p) and y ≡ -q (mod p).
        # b = (x + q) / p must be integer.
        # c = (y + q) / p must be integer.
        # b != c.
        
        # To avoid iterating all divisors of q^2 (which can be large),
        # we can iterate x up to sqrt(q^2) = q.
        # But q = 31a can be up to 31*1735 ~ 53785. q^2 ~ 2.9e9.
        # Iterating up to 53785 is fast.
        
        q2 = q * q
        # We need x * y = q2.
        # Iterate x from 1 to sqrt(q2).
        # Actually, we can just iterate potential b values?
        # 1/b = p/q - 1/c => 1/b = (pc - q) / (qc) => b = qc / (pc - q).
        # This is not simpler.
        
        # Let's stick to divisors of q^2.
        # But generating divisors of q^2 might be slow if q has many factors.
        # Alternative:
        # We know b > 1/p = q/p.
        # And b < 2q/p (since 1/c > 0).
        # So b is in (q/p, 2q/p).
        # We can iterate b in this range and check if c is integer.
        # c = 1 / (p/q - 1/b) = qb / (pb - q).
        # We need (pb - q) to divide (qb).
        
        b_min = math.floor(q / p) + 1
        b_max = math.ceil(2 * q / p)
        
        # Also b must be distinct from any previous denominators.
        # For now, just check existence.
        
        for b in range(b_min, b_max + 1):
            num = q * b
            den = p * b - q
            if den <= 0:
                continue
            if num % den == 0:
                c = num // den
                if c > 0 and b != c:
                    return (b, c)
        return None
    
    # Search for 3-term solutions
    # 1/a + 1/b + 1/c = 5/31
    # a < b < c
    # a must be such that 1/a < 5/31 => a > 31/5 = 6.2 => a >= 7.
    # Also 3/a > 5/31 => a < 93/5 = 18.6 => a <= 18.
    # So a is in [7, 18].
    
    found_3_term = None
    min_max_3_term = float('inf')
    
    for a in range(7, 19):
        p = 5 * a - 31
        q = 31 * a
        if p <= 0:
            continue
        # Check if 1/b + 1/c = p/q
        res = check_2_term(p, q)
        if res:
            b, c = res
            if b > a and c > b:
                max_den = c
                if max_den < min_max_3_term:
                    min_max_3_term = max_den
                    found_3_term = [a, b, c]
    
    if found_3_term and min_max_3_term < best_max_den:
        best_max_den = min_max_3_term
        best_solution = found_3_term
    
    # Now try 4-term solutions?
    # 1/a + 1/b + 1/c + 1/d = 5/31
    # a < b < c < d
    # a in [7, 15] (since 4/a > 5/31 => a < 124/5 = 24.8, but tighter bounds exist)
    # Actually, 1/a < 5/31 => a > 6.2 => a >= 7.
    # 4/a > 5/31 => a < 124/5 = 24.8 => a <= 24.
    # But we want to minimize max denominator d.
    # If we find a 4-term solution with d < 1736, it's better.
    
    # Let's search for 4-term solutions with d < best_max_den.
    
    # We can iterate a, b, c and solve for d.
    # 1/d = 5/31 - 1/a - 1/b - 1/c
    # d = 1 / (5/31 - 1/a - 1/b - 1/c)
    # d must be integer and d > c.
    
    # To limit the search:
    # a in [7, 24]
    # b > a, 1/b < 5/31 - 1/a
    # c > b, 1/c < 5/31 - 1/a - 1/b
    
    # This might be too slow if not pruned well.
    # Let's add a time check.
    
    def search_4_term(max_d_limit):
        nonlocal best_max_den, best_solution
        found = False
        min_max = float('inf')
        best_4 = None
        
        for a in range(7, 25):
            if time.time() > deadline:
                return found, min_max, best_4
            
            rem1 = Fraction(5, 31) - Fraction(1, a)
            if rem1 <= 0:
                continue
            
            # b > a
            # 1/b < rem1 => b > 1/rem1
            b_min = math.floor(1 / rem1) + 1
            # 3/b > rem1 => b < 3/rem1
            b_max = math.ceil(3 / rem1)
            
            for b in range(b_min, b_max + 1):
                if time.time() > deadline:
                    return found, min_max, best_4
                
                if b <= a:
                    continue
                    
                rem2 = rem1 - Fraction(1, b)
                if rem2 <= 0:
                    continue
                
                # c > b
                # 1/c < rem2 => c > 1/rem2
                c_min = math.floor(1 / rem2) + 1
                # 2/c > rem2 => c < 2/rem2
                c_max = math.ceil(2 / rem2)
                
                for c in range(c_min, c_max + 1):
                    if time.time() > deadline:
                        return found, min_max, best_4
                    
                    if c <= b:
                        continue
                        
                    rem3 = rem2 - Fraction(1, c)
                    if rem3 <= 0:
                        continue
                    
                    # 1/d = rem3
                    # d = 1/rem3
                    if rem3.numerator == 1:
                        d = rem3.denominator
                    else:
                        # Check if rem3 is a unit fraction
                        # rem3 = p/q. d = q/p. Must be integer.
                        if rem3.denominator % rem3.numerator != 0:
                            continue
                        d = rem3.denominator // rem3.numerator
                    
                    if d > c and d < max_d_limit:
                        max_den = d
                        if max_den < min_max:
                            min_max = max_den
                            best_4 = [a, b, c, d]
                            found = True
        
        return found, min_max, best_4
    
    # Try 4-term search with limit best_max_den
    found_4, min_max_4, best_4 = search_4_term(best_max_den)
    if found_4 and min_max_4 < best_max_den:
        best_max_den = min_max_4
        best_solution = best_4
    
    # Try 5-term? Might be too slow.
    # But let's try a limited 5-term search if we have time.
    
    def search_5_term(max_d_limit):
        nonlocal best_max_den, best_solution
        found = False
        min_max = float('inf')
        best_5 = None
        
        # a in [7, 20] (rough)
        for a in range(7, 21):
            if time.time() > deadline:
                return found, min_max, best_5
            
            rem1 = Fraction(5, 31) - Fraction(1, a)
            if rem1 <= 0:
                continue
            
            b_min = math.floor(1 / rem1) + 1
            b_max = math.ceil(4 / rem1) # 4 terms left
            
            for b in range(b_min, b_max + 1):
                if time.time() > deadline:
                    return found, min_max, best_5
                
                if b <= a:
                    continue
                    
                rem2 = rem1 - Fraction(1, b)
                if rem2 <= 0:
                    continue
                
                c_min = math.floor(1 / rem2) + 1
                c_max = math.ceil(3 / rem2)
                
                for c in range(c_min, c_max + 1):
                    if time.time() > deadline:
                        return found, min_max, best_5
                    
                    if c <= b:
                        continue
                        
                    rem3 = rem2 - Fraction(1, c)
                    if rem3 <= 0:
                        continue
                    
                    d_min = math.floor(1 / rem3) + 1
                    d_max = math.ceil(2 / rem3)
                    
                    for d in range(d_min, d_max + 1):
                        if time.time() > deadline:
                            return found, min_max, best_5
                        
                        if d <= c:
                            continue
                        
                        rem4 = rem3 - Fraction(1, d)
                        if rem4 <= 0:
                            continue
                        
                        # 1/e = rem4
                        if rem4.numerator == 1:
                            e = rem4.denominator
                        else:
                            if rem4.denominator % rem4.numerator != 0:
                                continue
                            e = rem4.denominator // rem4.numerator
                        
                        if e > d and e < max_d_limit:
                            max_den = e
                            if max_den < min_max:
                                min_max = max_den
                                best_5 = [a, b, c, d, e]
                                found = True
        
        return found, min_max, best_5
    
    found_5, min_max_5, best_5 = search_5_term(best_max_den)
    if found_5 and min_max_5 < best_max_den:
        best_max_den = min_max_5
        best_solution = best_5
    
    # Verify the best solution
    if best_solution:
        s = sum(Fraction(1, d) for d in best_solution)
        if s != target or len(set(best_solution)) != len(best_solution) or any(d < 1 or d > 10000000 for d in best_solution) or len(best_solution) > 256:
            # Fallback to greedy if verification fails
            best_solution = greedy_vals[:]
            best_max_den = max(greedy_vals)
    
    return best_solution

# EVOLVE-BLOCK-END
