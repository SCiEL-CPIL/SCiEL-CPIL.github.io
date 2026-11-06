<<<<<<< SEARCH
def _search_k8(deadline):
    """Search for 8 distinct denominators >= 5 summing to 1"""
    k = 8
    d = []
    
    def dfs(idx, current_sum_frac, current_sum_float, last_denom):
        if time.time() > deadline:
            return None
        
        if idx == k:
            if current_sum_frac == 1:
                return d[:]
            return None
        
        remaining_terms = k - idx
        remaining_frac = 1 - current_sum_frac
        remaining_float = 1.0 - current_sum_float
        
        if remaining_frac <= 0:
            return None
        
        # Pruning with float for speed
        max_possible_float = _max_sum_float(remaining_terms, last_denom + 1)
        if current_sum_float + max_possible_float < 1.0 - 1e-12:
            return None
        
        # Determine range for next denominator
        min_d_float = 1.0 / remaining_float
        min_d = max(last_denom + 1, int(math.ceil(min_d_float)))
        
        # Tighter upper bound: 1/d + max_sum(remaining_terms-1, d+1) >= remaining
        # For a valid d: remaining - 1/d >= min possible sum of remaining_terms-1 terms > d
        # And remaining - 1/d <= max possible sum of remaining_terms-1 terms > d
        # Upper bound: d must satisfy remaining - 1/d <= _max_sum_float(remaining_terms-1, d+1)
        # This is hard to invert, so use a loose bound then refine
        
        # Loose upper bound
        max_d_float = remaining_terms / remaining_float
        max_d = int(math.floor(max_d_float)) + 20
        
        if min_d > max_d:
            return None
        
        # Special case: last term
        if remaining_terms == 1:
            if remaining_frac.numerator == 1:
                d_val = remaining_frac.denominator
                if d_val > last_denom and d_val <= 10000000:
                    d.append(d_val)
                    if dfs(idx + 1, current_sum_frac + Fraction(1, d_val), current_sum_float + 1.0/d_val, d_val):
                        return d[:]
                    d.pop()
            return None
        
        # Special case: two terms left
        if remaining_terms == 2:
            # Need 1/a + 1/b = remaining_frac where a > last_denom, b > a
            # 1/a + 1/b = p/q => q(a+b) = pab => pab - qa - qb = 0 => (pa-q)(pb-q) = q^2
            p = remaining_frac.numerator
            q = remaining_frac.denominator
            # We need (pa - q)(pb - q) = q^2
            # So pa - q must be a divisor of q^2
            # And a > last_denom, b > a, both <= 10000000
            # Iterate over divisors of q^2
            q2 = q * q
            # Get all divisors of q^2
            divisors = []
            for i in range(1, int(math.isqrt(q2)) + 1):
                if q2 % i == 0:
                    divisors.append(i)
                    if i != q2 // i:
                        divisors.append(q2 // i)
            
            for div in divisors:
                if time.time() > deadline:
                    return None
                # pa - q = div => a = (div + q) / p
                num_a = div + q
                if num_a % p != 0:
                    continue
                a = num_a // p
                if a <= last_denom:
                    continue
                if a > 10000000:
                    continue
                # pb - q = q^2 / div => b = (q^2/div + q) / p
                other_div = q2 // div
                num_b = other_div + q
                if num_b % p != 0:
                    continue
                b = num_b // p
                if b <= a:
                    continue
                if b > 10000000:
                    continue
                # Verify
                if Fraction(1, a) + Fraction(1, b) == remaining_frac:
                    d.append(a)
                    d.append(b)
                    if dfs(idx + 2, current_sum_frac + Fraction(1, a) + Fraction(1, b), 
                           current_sum_float + 1.0/a + 1.0/b, b):
                        return d[:]
                    d.pop()
                    d.pop()
            return None
        
        # Iterate possible denominators
        for d_val in range(min_d, max_d + 1):
            if time.time() > deadline:
                return None
            
            if d_val <= last_denom:
                continue
            
            new_sum_float = current_sum_float + 1.0 / d_val
            if new_sum_float > 1.0:
                continue
            
            rem_float = 1.0 - new_sum_float
            if rem_float <= 0:
                continue
            
            if remaining_terms - 1 > 0:
                max_rem_float = _max_sum_float(remaining_terms - 1, d_val + 1)
                if rem_float > max_rem_float + 1e-12:
                    continue
                min_rem_float = _min_sum_float(remaining_terms - 1, d_val + 1)
                if rem_float < min_rem_float - 1e-12:
                    continue
            
            new_sum_frac = current_sum_frac + Fraction(1, d_val)
            
            d.append(d_val)
            res = dfs(idx + 1, new_sum_frac, new_sum_float, d_val)
            if res:
                return res
            d.pop()
        
        return None
    
    return dfs(0, Fraction(0), 0.0, 4)
=======
def _search_k8(deadline):
    """Search for 8 distinct denominators >= 5 summing to 1"""
    k = 8
    d = []
    
    def dfs(idx, current_sum_frac, current_sum_float, last_denom):
        if time.time() > deadline:
            return None
        
        if idx == k:
            if current_sum_frac == 1:
                return d[:]
            return None
        
        remaining_terms = k - idx
        remaining_frac = 1 - current_sum_frac
        remaining_float = 1.0 - current_sum_float
        
        if remaining_frac <= 0:
            return None
        
        # Pruning with float for speed
        max_possible_float = _max_sum_float(remaining_terms, last_denom + 1)
        if current_sum_float + max_possible_float < 1.0 - 1e-12:
            return None
        
        # Determine range for next denominator
        min_d_float = 1.0 / remaining_float
        min_d = max(last_denom + 1, int(math.ceil(min_d_float)))
        
        # Tighter upper bound
        max_d_float = remaining_terms / remaining_float
        max_d = int(math.floor(max_d_float)) + 10
        
        if min_d > max_d:
            return None
        
        # Special case: last term
        if remaining_terms == 1:
            if remaining_frac.numerator == 1:
                d_val = remaining_frac.denominator
                if d_val > last_denom and d_val <= 10000000:
                    d.append(d_val)
                    if dfs(idx + 1, current_sum_frac + Fraction(1, d_val), current_sum_float + 1.0/d_val, d_val):
                        return d[:]
                    d.pop()
            return None
        
        # Special case: two terms left - use algebraic solution
        if remaining_terms == 2:
            return _solve_two_terms(remaining_frac, last_denom, d, idx, current_sum_frac, current_sum_float, deadline)
        
        # Iterate possible denominators with tighter bounds
        for d_val in range(min_d, max_d + 1):
            if time.time() > deadline:
                return None
            
            if d_val <= last_denom:
                continue
            
            new_sum_float = current_sum_float + 1.0 / d_val
            if new_sum_float > 1.0:
                continue
            
            rem_float = 1.0 - new_sum_float
            if rem_float <= 0:
                continue
            
            if remaining_terms - 1 > 0:
                max_rem_float = _max_sum_float(remaining_terms - 1, d_val + 1)
                if rem_float > max_rem_float + 1e-12:
                    continue
            
            new_sum_frac = current_sum_frac + Fraction(1, d_val)
            
            d.append(d_val)
            res = dfs(idx + 1, new_sum_frac, new_sum_float, d_val)
            if res:
                return res
            d.pop()
        
        return None
    
    return dfs(0, Fraction(0), 0.0, 4)

def _solve_two_terms(remaining_frac, last_denom, d, idx, current_sum_frac, current_sum_float, deadline):
    """Solve 1/a + 1/b = remaining_frac with a > last_denom, b > a, both <= 10^7"""
    p = remaining_frac.numerator
    q = remaining_frac.denominator
    
    # (pa - q)(pb - q) = q^2
    # Let x = pa - q, then x divides q^2, and a = (x + q)/p
    # b = (q^2/x + q)/p
    
    q2 = q * q
    
    # Generate divisors of q^2 efficiently
    divisors = []
    sqrt_q2 = int(math.isqrt(q2))
    for i in range(1, sqrt_q2 + 1):
        if q2 % i == 0:
            divisors.append(i)
            if i != q2 // i:
                divisors.append(q2 // i)
    
    # Sort to try smaller a first
    divisors.sort()
    
    for x in divisors:
        if time.time() > deadline:
            return None
        
        # a = (x + q) / p
        num_a = x + q
        if num_a % p != 0:
            continue
        a = num_a // p
        if a <= last_denom:
            continue
        if a > 10000000:
            break  # Since divisors are sorted, larger x gives larger a
        
        # b = (q^2/x + q) / p
        y = q2 // x
        num_b = y + q
        if num_b % p != 0:
            continue
        b = num_b // p
        if b <= a:
            continue
        if b > 10000000:
            continue
        
        # Verify
        if Fraction(1, a) + Fraction(1, b) == remaining_frac:
            d.append(a)
            d.append(b)
            if dfs(idx + 2, current_sum_frac + Fraction(1, a) + Fraction(1, b), 
                   current_sum_float + 1.0/a + 1.0/b, b):
                return d[:]
            d.pop()
            d.pop()
    
    return None
>>>>>>> REPLACE

