# EVOLVE-BLOCK-START
from fractions import Fraction
import time
import math

def construct():
    start_time = time.time()
    deadline = start_time + 55
    
    # Priority: Find minimal k. k=8 is the theoretical minimum.
    # If we find k=8, we return it.
    # If we fail, we try k=9, etc.
    # If we fail all, we return a hardcoded valid solution.
    
    result = _search_k8_optimized(deadline)
    if result:
        return result
        
    # Fallback to larger k if 8 fails
    for k in range(9, 13):
        if time.time() > deadline - 5:
            break
        result = _search_k_optimized(k, deadline)
        if result:
            return result
            
    # Last resort: Hardcoded valid solution (from DP or known)
    # The DP solution from the seed is valid. We can just run it quickly.
    return _fallback_dp()

def _fallback_dp():
    """Subset dynamic programming over divisors of a locally chosen common denominator."""
    target, reached = 2520, {0: []}
    for d in range(5, target + 1):
        if target % d:
            continue
        w = target // d
        for total, values in list(reached.items()):
            if total + w <= target and total + w not in reached:
                reached[total + w] = values + [d]
        if target in reached:
            return reached[target]
    raise RuntimeError("no valid representation found")

def _search_k8_optimized(deadline):
    """Optimized search for 8 distinct denominators >= 5 summing to 1"""
    k = 8
    d = []
    
    # Use integer arithmetic for speed: track (num, den) of current sum
    # Target is 1/1
    
    def dfs(idx, num, den, last_denom):
        if time.time() > deadline:
            return None
        
        if idx == k:
            if num == den:
                return d[:]
            return None
        
        remaining_terms = k - idx
        remaining_num = den - num
        remaining_den = den
        
        if remaining_num <= 0:
            return None
        
        # Pruning 1: Max possible sum
        # Max sum of 'remaining_terms' fractions with denominators > last_denom
        # is sum_{i=1}^{remaining_terms} 1/(last_denom + i)
        # If current_sum + max_possible < 1, prune.
        # max_possible < remaining_num / remaining_den
        
        # Approximation for speed:
        # max_possible <= remaining_terms / (last_denom + 1)
        # If remaining_num * (last_denom + 1) > remaining_den * remaining_terms:
        #   Then 1/(last_denom+1) * remaining_terms < remaining
        #   So even the max possible sum is too small.
        if remaining_num * (last_denom + 1) > remaining_den * remaining_terms:
            return None

        # Pruning 2: Min possible sum (not strictly needed if we iterate from min_d)
        
        # Determine range for next denominator
        # 1/d_val <= remaining_num / remaining_den
        # d_val >= remaining_den / remaining_num
        min_d_float = remaining_den / remaining_num
        min_d = max(last_denom + 1, int(math.ceil(min_d_float)))
        
        # Upper bound:
        # We need to fill remaining_terms - 1 terms with sum >= remaining - 1/d_val
        # Max sum of remaining_terms - 1 terms is approx (remaining_terms - 1) / (d_val + 1)
        # So remaining - 1/d_val <= (remaining_terms - 1) / (d_val + 1)
        # This is complex. A simpler bound:
        # d_val must be small enough so that 1/d_val + max_sum(rest) >= remaining
        # 1/d_val + (remaining_terms - 1)/(d_val + 1) >= remaining_num/remaining_den
        # This is an inequality in d_val.
        # For now, use a loose bound: d_val <= remaining_terms * remaining_den / remaining_num + 10
        max_d = int(remaining_terms * remaining_den / remaining_num) + 10
        
        if min_d > max_d:
            return None
        
        # Special case: last term
        if remaining_terms == 1:
            if remaining_den % remaining_num == 0:
                d_val = remaining_den // remaining_num
                if d_val > last_denom and d_val <= 10000000:
                    d.append(d_val)
                    # Check if sum is exactly 1
                    # num/den + 1/d_val == 1 ?
                    # num/den + 1/d_val = (num*d_val + den) / (den*d_val)
                    # We know remaining_num/remaining_den = 1/d_val
                    # So num/den + 1/d_val = (den - remaining_num)/den + remaining_num/remaining_den = 1 - remaining + remaining = 1.
                    # So it is valid.
                    return d[:]
            return None
        
        # Special case: two terms left
        if remaining_terms == 2:
            # 1/a + 1/b = remaining_num/remaining_den
            # Let p = remaining_num, q = remaining_den
            # 1/a + 1/b = p/q => q(a+b) = pab => pab - qa - qb = 0
            # (pa - q)(pb - q) = q^2
            # Let X = pa - q, Y = pb - q. XY = q^2.
            # a = (X + q) / p, b = (Y + q) / p
            # X must be a divisor of q^2.
            # Also a > last_denom, b > a.
            p = remaining_num
            q = remaining_den
            q2 = q * q
            
            # Iterate divisors of q^2
            # Optimization: Only check divisors X such that a = (X+q)/p > last_denom
            # X > p*last_denom - q
            # And b > a => Y > X (since b > a => (Y+q)/p > (X+q)/p => Y > X)
            # So X < sqrt(q^2) = q.
            # So we only need to check divisors X of q^2 where X < q.
            
            # Get divisors of q^2 up to q
            # This can be slow if q is large. But q is the denominator of the remainder.
            # In early stages, q is small. In later stages, q can be large.
            # However, if remaining_terms is 2, we are at the end of the search.
            
            # To avoid iterating all divisors of q^2 if q is large, we can bound X.
            # X = pa - q.
            # a > last_denom => X > p*last_denom - q.
            # a <= 10000000 => X <= p*10000000 - q.
            # Also X < q.
            
            # If q is large, finding divisors is slow.
            # Alternative: Iterate a from last_denom+1 to some max.
            # 1/a + 1/b = p/q => 1/b = p/q - 1/a = (pa - q) / (qa)
            # b = qa / (pa - q)
            # We need pa - q to divide qa.
            # Let g = gcd(pa - q, q). Then (pa - q)/g must divide a.
            # This is still complex.
            
            # Let's stick to divisor enumeration but with a time check.
            # If q > 10000, divisor enumeration might be too slow.
            # In that case, fall back to iterating a.
            
            if q > 10000:
                # Iterate a
                # 1/a <= p/q => a >= q/p
                # 1/a > p/q / 2 => a < 2q/p (since 1/b > 0, 1/a < p/q)
                # Actually 1/b = p/q - 1/a > 0 => 1/a < p/q => a > q/p.
                # Also 1/b <= p/q => always true if a > q/p.
                # b = qa / (pa - q).
                # We need b > a => qa / (pa - q) > a => q / (pa - q) > 1 => q > pa - q => 2q > pa => a < 2q/p.
                # So a is in (q/p, 2q/p).
                min_a = max(last_denom + 1, int(math.ceil(q / p)))
                max_a = int(math.floor(2 * q / p))
                
                for a in range(min_a, max_a + 1):
                    if time.time() > deadline:
                        return None
                    num_b = q * a
                    den_b = p * a - q
                    if den_b <= 0:
                        continue
                    if num_b % den_b != 0:
                        continue
                    b = num_b // den_b
                    if b <= a:
                        continue
                    if b > 10000000:
                        continue
                    # Verify
                    if Fraction(1, a) + Fraction(1, b) == Fraction(p, q):
                        d.append(a)
                        d.append(b)
                        return d[:]
                return None
            else:
                # Enumerate divisors of q^2
                divisors = []
                for i in range(1, int(math.isqrt(q2)) + 1):
                    if q2 % i == 0:
                        divisors.append(i)
                        if i != q2 // i:
                            divisors.append(q2 // i)
                
                for div in divisors:
                    if time.time() > deadline:
                        return None
                    # div is X = pa - q
                    num_a = div + q
                    if num_a % p != 0:
                        continue
                    a = num_a // p
                    if a <= last_denom:
                        continue
                    if a > 10000000:
                        continue
                    # other_div is Y = pb - q
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
                    if Fraction(1, a) + Fraction(1, b) == Fraction(p, q):
                        d.append(a)
                        d.append(b)
                        return d[:]
                return None
        
        # Iterate possible denominators
        for d_val in range(min_d, max_d + 1):
            if time.time() > deadline:
                return None
            
            if d_val <= last_denom:
                continue
            
            # new_sum = num/den + 1/d_val = (num*d_val + den) / (den*d_val)
            new_num = num * d_val + den
            new_den = den * d_val
            
            # Check if new_sum > 1
            if new_num > new_den:
                continue
            
            d.append(d_val)
            res = dfs(idx + 1, new_num, new_den, d_val)
            if res:
                return res
            d.pop()
        
        return None
    
    return dfs(0, 0, 1, 4)

def _search_k_optimized(k, deadline):
    """General search for k denominators"""
    d = []
    
    def dfs(idx, num, den, last_denom):
        if time.time() > deadline:
            return None
        
        if idx == k:
            if num == den:
                return d[:]
            return None
        
        remaining_terms = k - idx
        remaining_num = den - num
        remaining_den = den
        
        if remaining_num <= 0:
            return None
        
        if remaining_num * (last_denom + 1) > remaining_den * remaining_terms:
            return None

        min_d_float = remaining_den / remaining_num
        min_d = max(last_denom + 1, int(math.ceil(min_d_float)))
        max_d = int(remaining_terms * remaining_den / remaining_num) + 10
        
        if min_d > max_d:
            return None
        
        if remaining_terms == 1:
            if remaining_den % remaining_num == 0:
                d_val = remaining_den // remaining_num
                if d_val > last_denom and d_val <= 10000000:
                    d.append(d_val)
                    return d[:]
            return None
        
        if remaining_terms == 2:
            p = remaining_num
            q = remaining_den
            if q > 10000:
                min_a = max(last_denom + 1, int(math.ceil(q / p)))
                max_a = int(math.floor(2 * q / p))
                for a in range(min_a, max_a + 1):
                    if time.time() > deadline:
                        return None
                    num_b = q * a
                    den_b = p * a - q
                    if den_b <= 0:
                        continue
                    if num_b % den_b != 0:
                        continue
                    b = num_b // den_b
                    if b <= a:
                        continue
                    if b > 10000000:
                        continue
                    if Fraction(1, a) + Fraction(1, b) == Fraction(p, q):
                        d.append(a)
                        d.append(b)
                        return d[:]
                return None
            else:
                q2 = q * q
                divisors = []
                for i in range(1, int(math.isqrt(q2)) + 1):
                    if q2 % i == 0:
                        divisors.append(i)
                        if i != q2 // i:
                            divisors.append(q2 // i)
                for div in divisors:
                    if time.time() > deadline:
                        return None
                    num_a = div + q
                    if num_a % p != 0:
                        continue
                    a = num_a // p
                    if a <= last_denom:
                        continue
                    if a > 10000000:
                        continue
                    other_div = q2 // div
                    num_b = other_div + q
                    if num_b % p != 0:
                        continue
                    b = num_b // p
                    if b <= a:
                        continue
                    if b > 10000000:
                        continue
                    if Fraction(1, a) + Fraction(1, b) == Fraction(p, q):
                        d.append(a)
                        d.append(b)
                        return d[:]
                return None
        
        for d_val in range(min_d, max_d + 1):
            if time.time() > deadline:
                return None
            
            if d_val <= last_denom:
                continue
            
            new_num = num * d_val + den
            new_den = den * d_val
            
            if new_num > new_den:
                continue
            
            d.append(d_val)
            res = dfs(idx + 1, new_num, new_den, d_val)
            if res:
                return res
            d.pop()
        
        return None
    
    return dfs(0, 0, 1, 4)
# EVOLVE-BLOCK-END
