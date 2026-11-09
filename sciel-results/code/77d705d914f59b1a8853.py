import time
from fractions import Fraction

def construct():
    start_time = time.time()
    deadline = start_time + 55  # Safety margin for 90s limit
    
    MAX_DENOM = 10000000
    MIN_DENOM = 5

    def ceil_div(a, b):
        """Ceiling division for positive integers a, b."""
        return (a + b - 1) // b

    def max_sum_from(d, count):
        """
        Calculate the maximum possible sum of 'count' distinct unit fractions
        with denominators starting from 'd' (i.e., d, d+1, ..., d+count-1).
        This is the sum of the smallest possible denominators, which yields the largest sum.
        """
        s = Fraction(0, 1)
        for i in range(count):
            s += Fraction(1, d + i)
        return s

    def dfs(rem, min_d, depth, path):
        """
        rem: remaining sum as Fraction
        min_d: minimum allowed denominator for the next term
        depth: number of terms left to choose
        path: list of chosen denominators so far
        """
        if time.time() > deadline:
            return None
        
        if depth == 0:
            if rem == 0:
                return path
            return None
        
        if rem == 0:
            return None
        
        if rem < 0:
            return None

        # Pruning 1: Max possible sum check
        # The largest sum we can add with 'depth' terms starting from min_d
        # is sum(1/(min_d + i) for i in 0..depth-1)
        # If current_rem > max_possible, it's impossible.
        # Wait, 'rem' is what we need to fill.
        # If rem > max_sum_from(min_d, depth), we cannot fill it.
        if depth <= 100: # Only do exact check for small depths to save time
            max_possible = max_sum_from(min_d, depth)
            if rem > max_possible:
                return None
        else:
            # Loose bound for large depth: sum(1/(d+i)) < 1/d + (depth-1)/(d+1) approx depth/d
            # Actually, sum(1/(d+i)) is roughly log((d+depth)/d) + 1/d
            # A simple upper bound is depth / min_d
            if Fraction(depth, min_d) < rem:
                return None

        # Base case optimization for small depths
        if depth == 1:
            # We need 1/d = rem => d = 1/rem
            # rem must be a unit fraction
            if rem.numerator == 1:
                d = rem.denominator
                if d >= min_d and d <= MAX_DENOM:
                    return path + [d]
            return None
            
        if depth == 2:
            # We need 1/a + 1/b = rem
            # 1/a + 1/b = (a+b)/(ab) = rem
            # Let rem = p/q. (a+b)/(ab) = p/q => q(a+b) = pab
            # pab - qa - qb = 0
            # (pa - q)(pb - q) = q^2
            # This is a Diophantine equation. 
            # Alternatively, iterate 'a' from min_d upwards.
            # 1/b = rem - 1/a = (rem*a - 1)/a
            # b = a / (rem*a - 1)
            # We need rem*a - 1 to be a divisor of a.
            # Also b > a => 1/b < 1/a => rem - 1/a < 1/a => rem < 2/a => a < 2/rem.
            # And 1/a <= rem => a >= 1/rem.
            
            # Lower bound for a: a >= ceil(1/rem)
            # Upper bound for a: a < 2/rem (since b > a implies 1/b < 1/a < rem/2? No. 1/a + 1/b = rem. If a <= b, then 2/a >= rem => a <= 2/rem. Also a < b implies 1/a > 1/b, so 1/a > rem/2 => a < 2/rem.)
            
            a_min = max(min_d, ceil_div(rem.denominator, rem.numerator))
            # a_max is roughly 2/rem. 
            # rem = p/q. 2/rem = 2q/p.
            a_max = (2 * rem.denominator) // rem.numerator
            
            if a_max < a_min:
                return None
                
            # We also need b <= MAX_DENOM
            # b = a / (rem*a - 1). Since rem > 0, rem*a - 1 > 0.
            # b is integer if (rem*a - 1) divides a.
            
            # To speed up, we can iterate a.
            # However, a_max can be large if rem is small.
            # But rem is the remainder after picking some terms. 
            # If depth=2 is reached, rem is likely not extremely small?
            # Actually, if we are minimizing k, and k is small, rem is significant.
            
            # Optimization: 
            # Let rem = p/q.
            # (pa - q)(pb - q) = q^2.
            # Let X = pa - q, Y = pb - q.
            # XY = q^2.
            # X must be a divisor of q^2.
            # X = pa - q => a = (X + q) / p.
            # So a is determined by a divisor X of q^2.
            # We need a >= min_d, a integer, b > a, b <= MAX_DENOM.
            
            # This divisor approach is much faster if q is small.
            # q = rem.denominator.
            # In early DFS steps, denominators are small, so q is manageable.
            
            p = rem.numerator
            q = rem.denominator
            
            # If q is too large, fall back to iteration or prune.
            # For q > 10^6, iteration might be safer or just prune.
            # But typically in minimal k searches, q stays moderate.
            
            if q > 1000000:
                # Fallback to iteration with tight bounds
                # a < 2/rem => a < 2q/p
                # Iterate a from a_min to min(a_max, 20000) ? 
                # If a_max is huge, this is slow.
                # Let's stick to the divisor method if possible, else iterate.
                # Actually, if q is huge, p is likely small (e.g., 1).
                # If p=1, 1/a + 1/b = 1/q => (a-q)(b-q)=q^2.
                # Divisors of q^2 are many.
                
                # Let's try iterating a from a_min.
                # But limit the range.
                # If a_max > 10000, we might time out.
                # However, if depth=2, we are close to the end.
                # Let's iterate.
                for a in range(a_min, a_max + 1):
                    if time.time() > deadline:
                        return None
                    # b = a / (rem*a - 1)
                    # rem*a - 1 = (p*a/q) - 1 = (p*a - q)/q
                    # b = a / ((p*a - q)/q) = a*q / (p*a - q)
                    num = a * q
                    den = p * a - q
                    if den <= 0:
                        continue
                    if num % den == 0:
                        b = num // den
                        if b > a and b <= MAX_DENOM:
                            return path + [a, b]
                return None
            else:
                # Divisor method
                # Find divisors of q^2
                # X * Y = q^2
                # X = p*a - q
                # Y = p*b - q
                # Since b > a, Y > X.
                # X < sqrt(q^2) => X < q.
                
                # Get divisors of q^2 up to q
                # Actually, just iterate divisors of q^2?
                # q^2 can be large.
                # Better: iterate divisors of q.
                # If X divides q^2, and X = p*a - q.
                
                # Let's just iterate divisors of q^2 is too slow if q is 10^6.
                # Let's iterate a from a_min to a_max, but a_max = 2q/p.
                # If p=1, a_max = 2q.
                # If q=10^6, a_max=2*10^6. This is too slow.
                
                # Let's use the property: a = (X+q)/p where X|q^2.
                # We need X < q (since Y > X).
                # So we need divisors X of q^2 such that X < q.
                
                # Factorize q? No, just trial division up to sqrt(q^2)=q? No.
                # Trial division up to sqrt(q) to find divisors of q, then combine?
                # Or just iterate a if range is small.
                
                # Let's check if a_max - a_min is small.
                if a_max - a_min < 10000:
                    for a in range(a_min, a_max + 1):
                        if time.time() > deadline:
                            return None
                        num = a * q
                        den = p * a - q
                        if den <= 0:
                            continue
                        if num % den == 0:
                            b = num // den
                            if b > a and b <= MAX_DENOM:
                                return path + [a, b]
                    return None
                else:
                    # Use divisor search on q^2
                    # We need X | q^2 and X < q.
                    # X = p*a - q.
                    # So p*a = X + q.
                    # a = (X+q)/p.
                    
                    # Find all divisors of q^2 less than q.
                    # This is equivalent to finding divisors of q^2.
                    # Let's find divisors of q first.
                    divs_q = []
                    i = 1
                    while i * i <= q:
                        if q % i == 0:
                            divs_q.append(i)
                            if i != q // i:
                                divs_q.append(q // i)
                        i += 1
                    
                    # Divisors of q^2 are products of divisors of q? No.
                    # If q = p1^e1 ... pk^ek, q^2 = p1^2e1 ...
                    # It's easier to just iterate potential a if we can bound it better?
                    # No, let's assume q is small enough for a_max - a_min to be small 
                    # in the context of minimal k DFS.
                    # If not, we might miss the solution or time out.
                    # Given the task, minimal k solutions usually involve small denominators, 
                    # so q should be manageable.
                    
                    # Fallback: Iterate a, but step by something? No, must be exact.
                    # Let's iterate a.
                    for a in range(a_min, a_max + 1):
                        if time.time() > deadline:
                            return None
                        num = a * q
                        den = p * a - q
                        if den <= 0:
                            continue
                        if num % den == 0:
                            b = num // den
                            if b > a and b <= MAX_DENOM:
                                return path + [a, b]
                    return None

        # General case: depth >= 3
        # Lower bound for next denominator d
        # 1/d <= rem => d >= 1/rem
        lower_bound = max(min_d, ceil_div(rem.denominator, rem.numerator))
        
        d = lower_bound
        
        while True:
            if time.time() > deadline:
                return None
            
            if d > MAX_DENOM:
                break
            
            # Pruning: 
            # If we pick d, new_rem = rem - 1/d.
            # We need to form new_rem with depth-1 terms starting from d+1.
            # Max sum of depth-1 terms from d+1 is max_sum_from(d+1, depth-1).
            # If new_rem > max_sum_from(d+1, depth-1), then this d is too small?
            # No, if d is small, 1/d is large, new_rem is small.
            # If new_rem is TOO SMALL to be formed by depth-1 terms?
            # Min sum of depth-1 terms from d+1 is sum(1/(d+1 + i) for i in depth-1-1)? No.
            # The terms must be distinct and > d.
            # The smallest possible sum for depth-1 terms > d is using the LARGEST denominators?
            # No, we are summing positive numbers.
            # We want to reach new_rem.
            # The MAXIMUM sum we can add with depth-1 terms starting from d+1 is max_sum_from(d+1, depth-1).
            # If new_rem > max_sum_from(d+1, depth-1), it's impossible.
            # This happens if 1/d was too small? No, if 1/d is small, new_rem is large.
            # So if new_rem is larger than the max possible sum of the rest, we need a SMALLER d.
            # But d is increasing. So if new_rem > max_possible, then for this d, it's impossible.
            # And for larger d, 1/d is smaller, new_rem is larger, so it will still be impossible.
            # So we can break.
            
            new_rem = rem - Fraction(1, d)
            
            if new_rem < 0:
                # This shouldn't happen if d >= ceil(1/rem), but just in case
                d += 1
                continue
                
            if new_rem == 0:
                if depth == 1:
                    return path + [d]
                else:
                    d += 1
                    continue

            # Check if new_rem can be formed by depth-1 terms starting from d+1
            if depth - 1 <= 100:
                max_possible_rest = max_sum_from(d + 1, depth - 1)
                if new_rem > max_possible_rest:
                    break # d is too large (1/d too small), remaining sum too big to cover
            else:
                if Fraction(depth - 1, d + 1) < new_rem:
                    break

            # Also check if new_rem is too small?
            # Min sum of depth-1 terms from d+1 is achieved by largest denominators?
            # No, we don't have an upper bound on denominators other than MAX_DENOM.
            # So min sum is approx 0.
            # So no lower bound pruning on new_rem other than > 0.

            res = dfs(new_rem, d + 1, depth - 1, path + [d])
            if res is not None:
                return res
            
            d += 1
            
        return None

    # Search for minimal k
    # k=1..7 are impossible.
    for k in range(8, 257):
        # Quick check if max possible sum < 1
        if k <= 20:
            max_sum = max_sum_from(MIN_DENOM, k)
            if max_sum < 1:
                continue
        else:
            # For large k, sum(1/(5+i)) grows like log(k).
            # log(256) + log(5) approx 5.5 + 1.6 = 7.1 > 1.
            pass

        result = dfs(Fraction(1), MIN_DENOM, k, [])
        if result is not None:
            return result
            
    # Fallback: Greedy algorithm
    def greedy(rem, min_d):
        path = []
        while rem > 0:
            if time.time() > deadline:
                return None
            d = max(min_d, ceil_div(rem.denominator, rem.numerator))
            if d > MAX_DENOM:
                return None
            path.append(d)
            rem -= Fraction(1, d)
            min_d = d + 1
        return path

    return greedy(Fraction(1), MIN_DENOM)
