import time
from fractions import Fraction
from math import gcd

def construct():
    deadline = time.time() + 85  # Safety margin within 90s
    
    def _find_3_terms():
        """Find 3 distinct denominators >= 5 with reciprocal sum 1."""
        for a in range(5, 200):
            # 1/a + 1/b + 1/c = 1 => 1/b + 1/c = 1 - 1/a = (a-1)/a
            # => (b+c)/(bc) = (a-1)/a => a(b+c) = (a-1)bc
            # => (a-1)bc - ab - ac = 0
            # => (a-1)b*c - a*b - a*c = 0
            # Multiply by (a-1): (a-1)^2 bc - a(a-1)b - a(a-1)c = 0
            # Add a^2: [(a-1)b - a][(a-1)c - a] = a^2
            num = a - 1
            den = a
            # We need 1/b + 1/c = num/den
            # => (b+c)/(bc) = num/den
            # => den*(b+c) = num*bc
            # => num*bc - den*b - den*c = 0
            # => (num*b - den)*(num*c - den) = den^2
            rhs = den * den
            # Find factor pairs of rhs where both factors > den
            # Let x = num*b - den, y = num*c - den, x*y = rhs
            # b = (x + den) / num, c = (y + den) / num
            # Need b > a, c > b, b >= 5, c >= 5, integer
            # Iterate over divisors of rhs
            x = 1
            while x * x <= rhs:
                if rhs % x == 0:
                    y = rhs // x
                    # x <= y
                    b_num = x + den
                    c_num = y + den
                    if b_num % num == 0 and c_num % num == 0:
                        b = b_num // num
                        c = c_num // num
                        if b > a and c > b and b >= 5 and c >= 5:
                            return [a, b, c]
                x += 1
        return None
    
    def _find_4_terms():
        """Find 4 distinct denominators >= 5 with reciprocal sum 1."""
        for a in range(5, 100):
            if time.time() > deadline:
                return None
            # 1/b + 1/c + 1/d = 1 - 1/a = (a-1)/a
            rem_num = a - 1
            rem_den = a
            # Find 3 terms summing to rem_num/rem_den
            for b in range(a + 1, a + 200):
                # 1/c + 1/d = rem_num/rem_den - 1/b = (rem_num*b - rem_den)/(rem_den*b)
                num2 = rem_num * b - rem_den
                den2 = rem_den * b
                if num2 <= 0:
                    break
                # Find 2 terms summing to num2/den2
                # 1/c + 1/d = num2/den2
                # => (c+d)/(cd) = num2/den2
                # => den2*(c+d) = num2*cd
                # => num2*cd - den2*c - den2*d = 0
                # => (num2*c - den2)*(num2*d - den2) = den2^2
                rhs2 = den2 * den2
                x = 1
                while x * x <= rhs2:
                    if rhs2 % x == 0:
                        y = rhs2 // x
                        c_num = x + den2
                        d_num = y + den2
                        if c_num % num2 == 0 and d_num % num2 == 0:
                            c = c_num // num2
                            d = d_num // num2
                            if c > b and d > c and c >= 5 and d >= 5:
                                return [a, b, c, d]
                    x += 1
        return None
    
    def _find_5_terms():
        """Find 5 distinct denominators >= 5 with reciprocal sum 1."""
        # Use greedy-like approach with backtracking
        def dfs5(remaining_num, remaining_den, min_d, count, path):
            if time.time() > deadline:
                return None
            if count == 0:
                if remaining_num == 0:
                    return list(path)
                return None
            if remaining_num <= 0:
                return None
            # Pruning: max sum with count terms starting from min_d
            # count/min_d >= remaining_num/remaining_den
            if remaining_num * min_d > remaining_den * count:
                return None
            d_start = max(min_d, (remaining_num + remaining_den - 1) // remaining_den)
            d_end = min(int(remaining_num * count / remaining_den) + 10, 10000000)
            if d_start > d_end:
                return None
            for d in range(d_start, d_end + 1):
                if time.time() > deadline:
                    return None
                if d in path:
                    continue
                # new_remaining = remaining - 1/d
                new_num = remaining_num * d - remaining_den
                new_den = remaining_den * d
                if new_num < 0:
                    continue
                # Reduce fraction
                g = gcd(new_num, new_den)
                if g > 1:
                    new_num //= g
                    new_den //= g
                if count - 1 == 0:
                    if new_num == 0:
                        return list(path) + [d]
                    continue
                # Feasibility: (count-1)/(d+1) >= new_num/new_den
                if new_num * (d + 1) > new_den * (count - 1):
                    break
                path.append(d)
                result = dfs5(new_num, new_den, d + 1, count - 1, path)
                path.pop()
                if result is not None:
                    return result
            return None
        return dfs5(1, 1, 5, 5, [])
    
    def dfs(remaining, min_d, count, path):
        """Find 'count' more distinct denominators >= min_d that sum to 'remaining'."""
        if time.time() > deadline:
            return None
        if count == 0:
            if remaining == 0:
                return list(path)
            return None
        if remaining <= 0:
            return None
        
        # Pruning: max possible sum with 'count' terms starting from min_d
        if remaining > Fraction(count, min_d):
            return None
        
        # Lower bound for d: 1/d <= remaining -> d >= 1/remaining
        d_start = max(min_d, (remaining.numerator + remaining.denominator - 1) // remaining.denominator)
        
        # Upper bound for d:
        # We need remaining - 1/d <= (count-1)/(d+1)
        # remaining <= 1/d + (count-1)/(d+1)
        # Let k = count - 1
        # remaining <= 1/d + k/(d+1)
        # The function f(d) = 1/d + k/(d+1) is decreasing for d > 0.
        # So we can find the max d such that f(d) >= remaining.
        # Approximate d_max: 1/d ~ remaining => d ~ 1/remaining.
        # Let's just iterate and break if the condition fails.
        # But we need an upper limit to avoid infinite loops or huge ranges if start is small.
        # d_start is at least 1/remaining.
        # d_max is roughly 1/remaining as well?
        # If k is small, 1/d dominates.
        
        # Let's iterate d from d_start upwards.
        # To prevent excessive iteration, we can estimate d_max.
        # 1/d + k/(d+1) >= remaining
        # For large d, LHS ~ (1+k)/d. So d <= (1+k)/remaining.
        # So d_max <= (count) / remaining.
        # This is a safe upper bound.
        
        d_end = int((count * remaining.numerator) // remaining.denominator) + 10
        d_end = min(d_end, 10000000)
        
        # If d_start > d_end, no solution
        if d_start > d_end:
            return None

        for d in range(d_start, d_end + 1):
            if time.time() > deadline:
                return None
            if d in path:
                continue
            
            # Check if 1/d <= remaining
            if Fraction(1, d) > remaining:
                # Since d is increasing, 1/d is decreasing.
                # If 1/d > remaining, we need larger d.
                # But we started at d_start which should satisfy this.
                # However, due to integer division, d_start might be slightly off.
                continue
            
            new_remaining = remaining - Fraction(1, d)
            
            if new_remaining < 0:
                continue
                
            # Feasibility check for remaining 'count-1' terms
            if count - 1 == 0:
                if new_remaining == 0:
                    return list(path) + [d]
                else:
                    continue
            
            # Max sum of (count-1) terms from d+1 is (count-1)/(d+1)
            if new_remaining > Fraction(count - 1, d + 1):
                # If new_remaining is too large, we need smaller d?
                # No, d is increasing. As d increases, 1/d decreases, so new_remaining increases.
                # Also (count-1)/(d+1) decreases.
                # So the gap widens.
                # If it fails here, it will fail for all larger d.
                break
            
            # Min sum of (count-1) terms from d+1 is (count-1)/10000000
            # if new_remaining < Fraction(count - 1, 10000000):
            #     # If new_remaining is too small, we need larger d?
            #     # As d increases, new_remaining increases.
            #     # So we continue.
            #     pass

            path.append(d)
            result = dfs(new_remaining, d + 1, count - 1, path)
            path.pop()
            if result is not None:
                return result
        
        return None
    
    # Try n from 3 upward, using fast solvers for small n
    for n in range(3, 10):
        if time.time() > deadline:
            break
        if n == 3:
            result = _find_3_terms()
        elif n == 4:
            result = _find_4_terms()
        elif n == 5:
            result = _find_5_terms()
        else:
            result = dfs(Fraction(1), 5, n, [])
        if result is not None:
            # Verify
            s = sum(Fraction(1, d) for d in result)
            if s == 1 and len(set(result)) == n and all(5 <= d <= 10000000 for d in result):
                return result
    
    # Fallback: DP over divisors of a large number
    target = 2520
    reached = {0: []}
    for d in range(5, target + 1):
        if target % d:
            continue
        w = target // d
        for total in list(reached.keys()):
            new_total = total + w
            if new_total <= target and new_total not in reached:
                reached[new_total] = reached[total] + [d]
        if target in reached:
            sol = reached[target]
            s = sum(Fraction(1, d) for d in sol)
            if s == 1 and all(5 <= d <= 10000000 for d in sol) and len(set(sol)) == len(sol):
                return sol

    # Last resort: known valid solution with small denominators
    # 1/5 + 1/6 + 1/7 + 1/10 + 1/14 + 1/35 = (42+35+30+21+15+6)/210 = 149/210 ≠ 1
    # Let's use: 1 = 1/5 + 1/6 + 1/8 + 1/30 + 1/120 + 1/120? No, need distinct.
    # Known: 1 = 1/5 + 1/6 + 1/7 + 1/42 + 1/140 + 1/420 + ... 
    # Actually, a simple valid set: 1/5 + 1/6 + 1/10 + 1/15 + 1/30 = (6+5+3+2+1)/30 = 17/30. No.
    # 1/5 + 1/6 + 1/7 + 1/42 = (42+35+30+7)/210 = 114/210 = 19/35. No.
    # Let me just return a hardcoded valid solution:
    # 1/5 + 1/6 + 1/8 + 1/12 + 1/30 + 1/60 + 1/120 + 1/180 + 1/360 + 1/720 + 1/1440 + 1/2880 + 1/5760 + 1/11520 + 1/23040 + 1/46080 + 1/92160 + 1/184320 + 1/368640 + 1/737280 + 1/1474560 + 1/2949120 + 1/5898240 + 1/11796480 + 1/23592960 + 1/47185920 + 1/94371840 + 1/188743680 + 1/377487360 + 1/754974720 + 1/1509949440 + 1/3019898880 + 1/6039797760 + 1/12079595520...
    # This is too long. Let me find a short one.
    # 1 = 1/5 + 4/5. 4/5 = 1/6 + 14/30 = 1/6 + 7/15. 7/15 = 1/7 + 34/105 = 1/7 + 1/11 + ...
    # Actually, let's just use the DFS result if it exists, or raise.
    raise RuntimeError("no valid representation found")
