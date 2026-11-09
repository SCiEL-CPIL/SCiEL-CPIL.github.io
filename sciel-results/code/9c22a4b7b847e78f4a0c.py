"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from fractions import Fraction

def construct():
    """Find Egyptian fraction of 5/31 minimizing the largest denominator.
    
    Uses number-theoretic divisor enumeration to solve Diophantine constraints
    exactly for 2, 3, and 4-term completions, then falls back to known solutions.
    """
    deadline = time.time() + 50  # 50s safety margin within 60s limit
    
    target = Fraction(5, 31)
    target_num = 5
    target_den = 31
    
    # Known fallback solution with max denominator 248
    fallback = [8, 31, 248]
    
    def get_divisors(n):
        """Get all divisors of n."""
        divs = []
        i = 1
        while i * i <= n:
            if n % i == 0:
                divs.append(i)
                if i != n // i:
                    divs.append(n // i)
            i += 1
        return divs
    
    def find_two_term(num, den, max_d):
        """Find b, c with 1/b + 1/c = num/den, b < c, both <= max_d.
        Returns list of (b, c) pairs."""
        # 1/b + 1/c = num/den => c = b*den / (b*num - den)
        # Let k = b*num - den, then c = b*den/k
        # k must divide b*den, and k > 0, and b*num > den => b > den/num
        results = []
        min_b = den // num + 1 if den % num != 0 else den // num + 1
        # Actually b > den/num, so b >= floor(den/num) + 1
        min_b = den // num + 1
        
        for b in range(min_b, max_d + 1):
            k = b * num - den
            if k <= 0:
                continue
            if (b * den) % k != 0:
                continue
            c = (b * den) // k
            if c <= b:
                continue
            if c > max_d:
                continue
            results.append((b, c))
        return results
    
    def try_three_term(a, remaining_num, remaining_den, max_d):
        """Given 1/a + 1/b + 1/c = target, find b, c with 1/b + 1/c = remaining_num/remaining_den,
        where b > a, c > b, all <= max_d."""
        # remaining = target - 1/a = (target_num*a - target_den) / (target_den * a)
        rem_num = target_num * a - target_den
        rem_den = target_den * a
        if rem_num <= 0:
            return None
        
        # Reduce fraction
        from math import gcd
        g = gcd(rem_num, rem_den)
        rn = rem_num // g
        rd = rem_den // g
        
        # Find b, c with 1/b + 1/c = rn/rd, b > a, c > b, c <= max_d
        # c = b*rd / (b*rn - rd), k = b*rn - rd > 0
        min_b = rd // rn + 1 if rd % rn != 0 else rd // rn + 1
        # Also b > a
        min_b = max(min_b, a + 1)
        
        for b in range(min_b, max_d + 1):
            k = b * rn - rd
            if k <= 0:
                continue
            if (b * rd) % k != 0:
                continue
            c = (b * rd) // k
            if c <= b:
                continue
            if c > max_d:
                continue
            return [a, b, c]
        
        return None
    
    def try_four_term(a, b, remaining_num, remaining_den, max_d):
        """Given 1/a + 1/b + 1/c + 1/d = target, find c, d with 1/c + 1/d = remaining,
        where c > b, d > c, all <= max_d."""
        rem_num = target_num * a * b - target_den * (a + b)
        rem_den = target_den * a * b
        if rem_num <= 0:
            return None
        
        from math import gcd
        g = gcd(rem_num, rem_den)
        rn = rem_num // g
        rd = rem_den // g
        
        # Find c, d with 1/c + 1/d = rn/rd, c > b, d > c, d <= max_d
        min_c = rd // rn + 1 if rd % rn != 0 else rd // rn + 1
        min_c = max(min_c, b + 1)
        
        for c in range(min_c, max_d + 1):
            k = c * rn - rd
            if k <= 0:
                continue
            if (c * rd) % k != 0:
                continue
            d = (c * rd) // k
            if d <= c:
                continue
            if d > max_d:
                continue
            return [a, b, c, d]
        
        return None
    
    # Try to find a solution with max denominator M < 248
    # For each M from 7 to 247, try all possible numbers of terms (2, 3, 4)
    
    best_solution = None
    
    for M in range(7, 248):
        if time.time() > deadline:
            break
        
        found = False
        
        # Try 2-term: 1/a + 1/b = 5/31
        # 5/31 = (a+b)/(ab) => 5ab = 31(a+b) => 5ab - 31a - 31b = 0
        # (5a - 31)(5b - 31) = 31^2 = 961
        # 961 = 1 * 961 = 31 * 31
        # 5a - 31 = 1 => a = 32/5 (not integer)
        # 5a - 31 = 31 => a = 62/5 (not integer)
        # 5a - 31 = 961 => a = 992/5 (not integer)
        # So no 2-term solution with integer denominators.
        
        # Try 3-term: 1/a + 1/b + 1/c = 5/31
        # a >= 7 (since 1/6 > 5/31)
        # a <= M, b > a, c > b, c <= M
        # For each a, try to find b, c
        
        for a in range(7, M + 1):
            if time.time() > deadline:
                break
            
            # remaining = 5/31 - 1/a = (5a - 31) / (31a)
            rem_num = 5 * a - 31
            if rem_num <= 0:
                continue
            
            # We need 1/b + 1/c = rem_num / (31a) with b > a, c > b, c <= M
            # c = b * 31a / (b * rem_num - 31a)
            # k = b * rem_num - 31a > 0 => b > 31a / rem_num
            
            min_b = (31 * a) // rem_num + 1
            min_b = max(min_b, a + 1)
            
            for b in range(min_b, M + 1):
                k = b * rem_num - 31 * a
                if k <= 0:
                    continue
                if (b * 31 * a) % k != 0:
                    continue
                c = (b * 31 * a) // k
                if c <= b:
                    continue
                if c > M:
                    continue
                
                # Found a valid 3-term solution
                solution = [a, b, c]
                # Verify
                s = Fraction(1, a) + Fraction(1, b) + Fraction(1, c)
                if s == target:
                    best_solution = solution
                    found = True
                    break
            
            if found:
                break
        
        if found:
            break
        
        # Try 4-term: 1/a + 1/b + 1/c + 1/d = 5/31
        # a >= 7, b > a, c > b, d > c, d <= M
        # For efficiency, only try a few values of a and b
        
        if not found:
            for a in range(7, min(M + 1, 20)):  # Limit a to keep search manageable
                if time.time() > deadline:
                    break
                
                rem1_num = 5 * a - 31
                if rem1_num <= 0:
                    continue
                
                min_b = (31 * a) // rem1_num + 1
                min_b = max(min_b, a + 1)
                
                for b in range(min_b, min(M + 1, 50)):  # Limit b
                    if time.time() > deadline:
                        break
                    
                    # remaining after a, b: 5/31 - 1/a - 1/b
                    rem2_num = 5 * a * b - 31 * (a + b)
                    if rem2_num <= 0:
                        continue
                    
                    # Need 1/c + 1/d = rem2_num / (31ab) with c > b, d > c, d <= M
                    min_c = (31 * a * b) // rem2_num + 1
                    min_c = max(min_c, b + 1)
                    
                    for c in range(min_c, M + 1):
                        k = c * rem2_num - 31 * a * b
                        if k <= 0:
                            continue
                        if (c * 31 * a * b) % k != 0:
                            continue
                        d = (c * 31 * a * b) // k
                        if d <= c:
                            continue
                        if d > M:
                            continue
                        
                        # Found a valid 4-term solution
                        solution = [a, b, c, d]
                        s = Fraction(1, a) + Fraction(1, b) + Fraction(1, c) + Fraction(1, d)
                        if s == target:
                            best_solution = solution
                            found = True
                            break
                    
                    if found:
                        break
                
                if found:
                    break
        
        if found:
            break
    
    if best_solution is not None:
        # Final verification
        s = sum(Fraction(1, d) for d in best_solution)
        if (s == target and 
            len(set(best_solution)) == len(best_solution) and 
            all(1 <= d <= 10000000 for d in best_solution) and 
            len(best_solution) <= 256):
            return best_solution
    
    return fallback
# EVOLVE-BLOCK-END
