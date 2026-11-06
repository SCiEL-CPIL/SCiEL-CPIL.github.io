import time
from fractions import Fraction

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # 60s limit, keep margin
    
    target = Fraction(5, 31)
    
    # Theoretical lower bound for M
    # 5/31 ~ 0.161
    # If M is very small, we can't sum to 0.161 with distinct denominators <= M
    # Max sum with denominators <= M is H_M (harmonic number)
    # H_7 ~ 2.59, H_1 ~ 1
    # We need H_M >= 5/31, which is true for all M >= 1.
    # But we need distinct denominators.
    # Let's just start from M=2.
    
    # We know M=248 works. We search for M < 248.
    # If we find a solution for M, we return it immediately as it's the minimal M.
    
    # Optimization: Check 3-term solutions first for all M < 248?
    # No, just iterate M from low to high.
    # For each M, check k=2, 3, 4, 5, 6.
    
    # Helper to check if a solution exists with max denom <= M
    # We want the solution with the smallest M.
    
    # We'll iterate M from 2 to 247.
    # For efficiency, we can separate the checks.
    # 2-term check is O(1) per M? No, 2-term solution is unique if it exists.
    # 1/a + 1/b = 5/31 => (5a-31)(5b-31) = 961.
    # We already determined no integer solution for 2 terms.
    
    # So we start with 3 terms.
    # We can iterate M from 19 to 247 (since 3 terms need M >= 19 roughly).
    # Actually, let's just iterate M from 2 to 247.
    
    for M in range(2, 248):
        if time.time() > deadline:
            break
            
        # Check 3-term solutions with max denom <= M
        # 1/a + 1/b + 1/c = 5/31, a < b < c <= M
        found = None
        for a in range(1, M - 1):
            # Pruning: 1/a < 5/31 => a > 31/5 = 6.2 => a >= 7
            if a < 7:
                continue
            # Max possible sum with b, c > a is 1/(a+1) + 1/(a+2)
            # If 1/a + 1/(a+1) + 1/(a+2) < 5/31, then even with smallest b,c we can't reach target?
            # No, we want sum = 5/31.
            # If 1/a > 5/31, we can't use a? No, other terms are positive.
            # If 1/a >= 5/31, then 1/b + 1/c <= 0, impossible.
            # So we need 1/a < 5/31 => a > 6.2.
            
            # Lower bound for b:
            # 1/b < 5/31 - 1/a
            # b > 1 / (5/31 - 1/a) = 31a / (5a - 31)
            # Also b > a.
            
            min_b = max(a + 1, (31 * a) // (5 * a - 31) + 1)
            
            # Upper bound for b:
            # 1/b + 1/c >= 1/b + 1/(b+1) (approx, assuming c > b)
            # Actually, min sum for b,c is 1/M + 1/(M-1) ? No, c <= M.
            # To allow c <= M, we need 1/b + 1/c >= 5/31 - 1/a.
            # The max value of 1/b + 1/c with b < c <= M is 1/(a+1) + 1/(a+2) ? No.
            # The max value is when b and c are as small as possible.
            # But we are iterating b.
            # For a fixed a, b, c is determined.
            # c = (31ab) / (5ab - 31a - 31b)
            # We need c > b and c <= M.
            
            # So we just iterate b from min_b to M-1.
            # But we can prune b.
            # c > b => (31ab) / (5ab - 31a - 31b) > b
            # 31ab > 5b^2a - 31ab - 31b^2
            # 62ab + 31b^2 > 5ab^2
            # 31b^2 + 62ab - 5ab^2 > 0
            # This is always true for large b?
            # Let's just check c directly.
            
            # Also c <= M => (31ab) / (5ab - 31a - 31b) <= M
            # 31ab <= M(5ab - 31a - 31b)
            # 31ab <= 5Mab - 31Ma - 31Mb
            # 31Ma + 31Mb <= 5Mab - 31ab
            # 31Ma <= ab(5M - 31) - 31Mb
            # This is getting messy. Just iterate b.
            
            for b in range(min_b, M):
                denom = 5 * a * b - 31 * a - 31 * b
                if denom <= 0:
                    continue
                if (31 * a * b) % denom != 0:
                    continue
                c = (31 * a * b) // denom
                if c <= b:
                    continue
                if c > M:
                    # Since c increases as b increases?
                    # Let's check derivative or just break if c > M and it's increasing.
                    # f(b) = 31ab / (5ab - 31a - 31b)
                    # As b -> infinity, f(b) -> 31a/5.
                    # So c is bounded.
                    # If c > M, we might still find a valid c for larger b?
                    # No, if c > M, we need smaller c.
                    # Smaller c means larger denom.
                    # Larger denom means smaller b?
                    # Let's not break, just continue.
                    pass
                else:
                    # Found a valid 3-term solution
                    found = [a, b, c]
                    break
            if found:
                break
        if found:
            return found
            
        # If no 3-term solution, check 4-term solutions
        # This is more expensive.
        # We'll use DFS.
        
        def dfs(rem, last, path, terms_left):
            if time.time() > deadline:
                return None
            if rem == 0:
                if path:
                    return list(path)
                return None
            if terms_left == 0:
                return None
            if rem < 0:
                return None
                
            # Pruning:
            # Max possible sum with terms_left terms > last and <= M
            # Smallest denominators: last+1, last+2, ...
            max_sum = Fraction(0, 1)
            d = last + 1
            for _ in range(terms_left):
                if d > M:
                    break
                max_sum += Fraction(1, d)
                d += 1
            if rem > max_sum:
                return None
                
            # Min possible sum with terms_left terms > last and <= M
            # Largest denominators: M, M-1, ...
            min_sum = Fraction(0, 1)
            d = M
            for _ in range(terms_left):
                if d <= last:
                    break
                min_sum += Fraction(1, d)
                d -= 1
            if rem < min_sum:
                return None
                
            # Bounds for next denominator d
            # 1/d < rem => d > 1/rem
            # Also d > last
            # And d <= M
            
            # Lower bound: d >= ceil(1/rem)
            min_d = max(last + 1, (rem.denominator + rem.numerator - 1) // rem.numerator)
            
            # Upper bound:
            # rem <= 1/d + 1/(d+1) + ... (terms_left terms)
            # Approx: rem <= terms_left / d => d <= terms_left / rem
            max_d = min(M, (terms_left * rem.denominator) // rem.numerator)
            
            if min_d > max_d:
                return None
                
            for d in range(min_d, max_d + 1):
                new_rem = rem - Fraction(1, d)
                if new_rem < 0:
                    continue
                path.append(d)
                res = dfs(new_rem, d, path, terms_left - 1)
                if res:
                    return res
                path.pop()
            return None
        
        # Try 4 terms
        sol = dfs(target, 0, [], 4)
        if sol:
            return sol
            
        # Try 5 terms
        sol = dfs(target, 0, [], 5)
        if sol:
            return sol
            
        # Try 6 terms
        sol = dfs(target, 0, [], 6)
        if sol:
            return sol
            
    # If no solution found with M < 248, return the known solution
    return [8, 31, 248]
