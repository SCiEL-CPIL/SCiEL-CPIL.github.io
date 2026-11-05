def construct():
    import time
    from fractions import Fraction
    from math import isqrt, gcd

    start_time = time.time()
    DEADLINE = 85.0  # Safety margin before 90s kill

    target = Fraction(5, 31)
    best_max = float('inf')
    best_solution = None

    # Known 3-term solution: 1/8 + 1/31 + 1/248 = 5/31
    # 1/8 = 31/248, 1/31 = 8/248, 1/248 = 1/248 -> (31+8+1)/248 = 40/248 = 5/31
    best_solution = [8, 31, 248]
    best_max = 248

    def time_left():
        return DEADLINE - (time.time() - start_time)

    def solve_two_terms(rem, min_val, max_val):
        """
        Find (x, y) with min_val < x < y <= max_val such that 1/x + 1/y = rem.
        rem is a Fraction.
        Returns (x, y) or None.
        """
        if rem <= 0:
            return None
        
        n = rem.numerator
        d = rem.denominator
        
        # 1/x + 1/y = n/d => d(x+y) = nxy => nxy - dx - dy = 0
        # (nx - d)(ny - d) = d^2
        # Let A = nx - d, B = ny - d. Then AB = d^2.
        # x = (A+d)/n, y = (B+d)/n
        # Constraints:
        # x > min_val => A > n*min_val - d
        # y <= max_val => B <= n*max_val - d
        # x < y => A < B => A < d (since AB=d^2 and A,B>0)
        # A must be a divisor of d^2.
        
        # Iterate A from 1 to d
        # This is faster than iterating divisors of d^2 if d is small, 
        # but d can be large. However, A must be < d.
        
        A_min = n * min_val - d
        if A_min < 0:
            A_min = 0
            
        # A must be a divisor of d^2
        # And A < d
        # And A > A_min
        # And B = d^2 / A must satisfy B <= n*max_val - d
        
        B_max = n * max_val - d
        
        # We iterate A. To be fast, we only check A that divides d^2.
        # Since A < d, we can iterate i from 1 to isqrt(d^2) which is d.
        # But we only need divisors.
        
        # Optimization: If d is very large, this loop is slow.
        # But in our DFS, rem is small fraction, so d is manageable.
        
        best_pair = None
        best_y = float('inf')
        
        # We can just iterate A from A_min+1 to d-1
        # But we must check if A divides d^2.
        # d^2 can be huge. 
        # Alternative: Iterate divisors of d. Let d = p1^a1 ... pk^ak.
        # Divisors of d^2 are p1^b1 ... pk^bk where 0<=bi<=2ai.
        # This is complex to generate dynamically.
        
        # Given the small depth (max 7 terms) and pruning, d should stay small.
        # Let's use a simple loop with a check, but break early if possible.
        
        # Actually, a better bound for A:
        # A = nx - d. Since x > min_val, A > n*min_val - d.
        # Also A < d.
        
        # Let's iterate A in range(max(1, A_min + 1), d):
        # This is too slow if d is large (e.g., 10^6).
        # We need to iterate divisors.
        
        # Let's factor d.
        def get_divisors(n):
            divs = []
            i = 1
            while i * i <= n:
                if n % i == 0:
                    divs.append(i)
                    if i != n // i:
                        divs.append(n // i)
                i += 1
            return divs
        
        # d might be large. If d > 10^6, this is slow.
        # But note: rem = n/d. 
        # In our search, rem is the remaining fraction.
        # Initially rem = 5/31. d=31.
        # Next rem is smaller. d grows.
        # However, we prune by max_val.
        
        # If d is too large to factor quickly, we might miss solutions.
        # But for minimizing max denominator, we expect small d.
        
        if d > 100000:
            # Fallback: brute force x?
            # x is in (min_val, max_val].
            # If max_val - min_val is small, iterate x.
            if max_val - min_val < 10000:
                for x in range(min_val + 1, max_val + 1):
                    # 1/y = rem - 1/x = (n*x - d) / (d*x)
                    # y = d*x / (n*x - d)
                    num_y = d * x
                    den_y = n * x - d
                    if den_y <= 0:
                        continue
                    if num_y % den_y == 0:
                        y = num_y // den_y
                        if y > x and y <= max_val:
                            return (x, y)
            return None

        d_divs = get_divisors(d)
        # Divisors of d^2 are formed by taking divisors of d and squaring? No.
        # If d = p^k, divisors of d^2 are p^0..p^2k.
        # We need divisors of d^2 that are < d.
        
        # Let's just get divisors of d^2 directly if d is small enough.
        # d^2 for d=1000 is 1,000,000. Loop to 1000. OK.
        # d^2 for d=10,000 is 100,000,000. Loop to 10,000. OK.
        # d^2 for d=100,000 is 10^10. Loop to 100,000. Might be slow.
        
        # We already handled d > 100,000 with brute force x if range is small.
        # If range is large and d is large, we might time out.
        # But such deep branches are unlikely to improve best_max=248.
        
        d_sq = d * d
        # Get divisors of d_sq
        # We only care about A < d.
        # So we iterate i from 1 to isqrt(d_sq) = d.
        # If i divides d_sq, then A=i is a candidate.
        
        for A in range(1, d):
            if d_sq % A != 0:
                continue
            if A <= A_min:
                continue
            B = d_sq // A
            # B must be > A (since x < y => A < B)
            if B <= A:
                continue
            if B > B_max:
                continue
            
            # x = (A + d) / n
            if (A + d) % n != 0:
                continue
            x = (A + d) // n
            if x <= min_val:
                continue
            
            # y = (B + d) / n
            if (B + d) % n != 0:
                continue
            y = (B + d) // n
            if y <= x:
                continue
            if y > max_val:
                continue
            
            return (x, y)
            
        return None

    def dfs(rem, current_max, min_den, path, depth, max_depth):
        nonlocal best_max, best_solution
        
        if time_left() < 1.0:
            return
            
        if rem == 0:
            if current_max < best_max:
                best_max = current_max
                best_solution = path.copy()
            return
            
        if depth >= max_depth:
            return
            
        # Pruning:
        # 1. min_den must be > last denominator in path
        # 2. max possible sum of remaining terms (depth - current_depth) * (1/min_den) < rem
        #    Actually, we want to know if we CAN reach rem.
        #    Max sum of remaining (k-1) terms with denominators > min_den is bounded.
        #    But we also have an upper bound on denominators: best_max.
        #    So min possible denominator for next term is min_den.
        #    Max possible denominator is best_max.
        
        # Feasibility check:
        # Can we form rem with (max_depth - depth) terms?
        # Each term 1/d where d >= min_den.
        # Sum <= (max_depth - depth) / min_den.
        if rem > (max_depth - depth) / min_den:
            return
            
        # Each term 1/d where d <= best_max.
        # Sum >= (max_depth - depth) / best_max.
        # This doesn't help much for pruning upper bound of rem, 
        # but helps if rem is very small?
        # If rem < (max_depth - depth) / best_max, we might still be fine 
        # if we use smaller denominators? No, denominators must be >= min_den.
        # The smallest possible sum of remaining terms is when denominators are as large as possible.
        # But we are constrained by d <= best_max.
        # So min sum = (max_depth - depth) / best_max.
        # If rem < min_sum, it's impossible? 
        # No, we can choose denominators smaller than best_max (but >= min_den).
        # So the range of possible sums is [ (k)/best_max, (k)/min_den ].
        # If rem is outside this range, prune.
        
        k_rem = max_depth - depth
        min_possible_sum = k_rem / best_max
        max_possible_sum = k_rem / min_den
        
        if rem < min_possible_sum or rem > max_possible_sum:
            return

        # Determine range for next denominator d
        # 1/d < rem => d > 1/rem
        d_min = int(1 / rem) + 1
        if d_min < min_den:
            d_min = min_den
            
        # 1/d <= rem => d >= 1/rem (already covered)
        # Also we need to leave enough for remaining terms.
        # rem - 1/d >= (k_rem - 1) / best_max
        # 1/d <= rem - (k_rem - 1) / best_max
        # d >= 1 / (rem - (k_rem - 1) / best_max)
        
        lower_bound_rem = rem - Fraction(k_rem - 1, best_max)
        if lower_bound_rem > 0:
            d_min2 = int(1 / lower_bound_rem) + 1
            if d_min2 > d_min:
                d_min = d_min2
        else:
            # If rem is very small, we might not be able to subtract 1/d and keep positive?
            # Actually, if rem < (k_rem-1)/best_max, we already pruned?
            # min_possible_sum = k_rem / best_max.
            # If rem >= k_rem / best_max, then rem - (k_rem-1)/best_max >= 1/best_max > 0.
            # So lower_bound_rem should be positive.
            pass

        d_max = best_max
        
        # Iterate d from d_min to d_max
        # To find small max denominators, we should iterate d in increasing order?
        # If we pick a small d, the remainder is larger, forcing larger subsequent denominators?
        # Actually, picking a small d (large 1/d) reduces the remainder more,
        # which might allow smaller subsequent denominators?
        # No, smaller d means larger 1/d.
        # If we pick d close to 1/rem, 1/d is close to rem, leaving small remainder.
        # Small remainder requires large denominators.
        # If we pick d much larger than 1/rem, 1/d is small, leaving large remainder.
        # Large remainder requires smaller denominators (but > d).
        
        # We want to minimize the MAX denominator.
        # So we want all denominators to be small.
        # This suggests picking d as small as possible?
        # If d is small, 1/d is large.
        # If 1/d > rem, invalid.
        # So d must be > 1/rem.
        # The smallest valid d is d_min.
        # Let's try d_min first.
        
        for d in range(d_min, d_max + 1):
            if time_left() < 0.5:
                return
                
            new_rem = rem - Fraction(1, d)
            if new_rem < 0:
                break # Since d is increasing, 1/d is decreasing, new_rem will increase?
                # Wait. d increases => 1/d decreases => rem - 1/d increases.
                # So if new_rem < 0, we need to INCREASE d?
                # No. If 1/d > rem, then rem - 1/d < 0.
                # We need 1/d < rem.
                # d > 1/rem.
                # d_min is set to int(1/rem) + 1.
                # So for d >= d_min, 1/d <= 1/d_min < rem?
                # 1/d_min <= 1/(1/rem + 1) < rem.
                # So new_rem should be positive.
                # Let's re-verify.
                # If d is very large, 1/d is small, new_rem is close to rem.
                # If d is small (near 1/rem), 1/d is near rem, new_rem is near 0.
                # So new_rem is always positive for d >= d_min.
                # The loop break condition is wrong.
                # We should break if d > d_max.
                
            if new_rem == 0:
                if d < best_max:
                    best_max = d
                    best_solution = path + [d]
                return # Found a solution with this d, no need to try larger d for this branch?
                       # Actually, larger d would mean smaller 1/d, larger new_rem.
                       # If new_rem == 0, we are done.
                       # If we try larger d, new_rem > 0, we need more terms.
                       # But we are in DFS, so we should continue to see if we can get
                       # a smaller max denominator with more terms?
                       # No, if we found a solution with max denominator d,
                       # any solution with more terms will have max denominator >= d?
                       # Not necessarily. The subsequent denominators must be > d.
                       # So the max will be at least the last denominator.
                       # If we use more terms, the last denominator will be > d.
                       # So max will be > d.
                       # Thus, if we found a solution ending at d,
                       # any extension will have max > d.
                       # So we can return.
                       
            # Prune: if d >= best_max, we won't improve best_max
            # (since all subsequent denominators > d >= best_max)
            if d >= best_max:
                break
                
            dfs(new_rem, d, d + 1, path + [d], depth + 1, max_depth)

    # Start DFS
    # We know 3-term solution has max 248.
    # Let's try to find 4, 5, 6, 7 term solutions with max < 248.
    
    # Initial call
    # rem = 5/31
    # current_max = inf (will be set by d)
    # min_den = 1
    # path = []
    # depth = 0
    # max_depth = 7
    
    # To speed up, we can start with a tighter bound.
    # We know best_max = 248.
    # So we only care about solutions with max < 248.
    
    dfs(target, 0, 1, [], 0, 7)

    if best_solution is None:
        # Fallback to known 3-term
        return [8, 31, 248]
        
    return best_solution
