"""Model-and-optimize: search for Egyptian fraction of 5/31 minimizing max denominator."""
import time
import math

def construct():
    start = time.time()
    deadline = start + 85  # safety margin before 90s kill

    # Target: 5/31
    # We'll work with (num, den) pairs for speed

    best_max = float('inf')
    best_result = None

    def update_best(denoms):
        nonlocal best_max, best_result
        mx = max(denoms)
        if mx < best_max:
            best_max = mx
            best_result = list(denoms)

    def gcd(a, b):
        while b:
            a, b = b, a % b
        return a

    def reduce(num, den):
        if num == 0:
            return (0, 1)
        g = gcd(abs(num), abs(den))
        num //= g
        den //= g
        if den < 0:
            num = -num
            den = -den
        return (num, den)

    def sub_frac(n1, d1, n2, d2):
        """(n1/d1) - (n2/d2)"""
        num = n1 * d2 - n2 * d1
        den = d1 * d2
        return reduce(num, den)

    def is_unit_fraction(n, d):
        """Check if n/d is a unit fraction 1/x for integer x > 0"""
        if d == 0:
            return None
        g = gcd(abs(n), abs(d))
        n //= g
        d //= g
        if d < 0:
            n = -n
            d = -d
        if n == 1 and d > 0:
            return d
        if n == -1 and d > 0:
            return -d
        return None

    def find_2term(n, d, min_denom, max_denom):
        """Find b < c with b >= min_denom, c <= max_denom such that 1/b + 1/c = n/d.
        Returns list of (b, c) pairs."""
        results = []
        # 1/b + 1/c = n/d => d(b+c) = n*b*c => n*b*c - d*b - d*c = 0
        # Multiply by n: n²*b*c - n*d*b - n*d*c = 0
        # (n*b - d)(n*c - d) = d²
        # Let u = n*b - d, v = n*c - d, then u*v = d²
        # b = (u + d) / n, c = (v + d) / n
        # Need u*v = d², u < v (since b < c), b >= min_denom, c <= max_denom
        # Also b > c is not allowed, we want b < c so u < v
        
        d_sq = d * d
        # Enumerate divisors u of d_sq where u <= sqrt(d_sq)
        # But d can be large... let's be smart
        # b = (u + d) / n must be integer, so (u + d) % n == 0
        # c = (v + d) / n where v = d_sq // u, must be integer, (v + d) % n == 0
        
        # For efficiency, iterate u from 1 to sqrt(d_sq)
        sqrt_d_sq = int(math.isqrt(d_sq))
        
        for u in range(1, sqrt_d_sq + 1):
            if d_sq % u != 0:
                continue
            v = d_sq // u
            # u <= v
            if u > v:
                continue
            # b = (u + d) / n
            if (u + d) % n != 0:
                continue
            b = (u + d) // n
            if b < min_denom or b >= max_denom:
                continue
            # c = (v + d) / n
            if (v + d) % n != 0:
                continue
            c = (v + d) // n
            if c <= b or c >= max_denom:
                continue
            # Verify: 1/b + 1/c == n/d
            if b * c * n == d * (b + c):
                results.append((b, c))
        
        return results

    # 3-term search: a < b < c
    # 1/a + 1/b + 1/c = 5/31
    # a in [7, 18] (since 3/a >= 5/31 => a <= 18.6, and 1/a < 5/31 => a >= 7)
    for a in range(7, 19):
        if time.time() > deadline:
            break
        # rem = 5/31 - 1/a = (5a - 31) / (31a)
        rem_n = 5 * a - 31
        rem_d = 31 * a
        if rem_n <= 0:
            continue
        rem_n, rem_d = reduce(rem_n, rem_d)
        if rem_n <= 0:
            continue
        
        # 1/b + 1/c = rem_n / rem_d
        # b > 1/rem = rem_d/rem_n, b <= 2/rem = 2*rem_d/rem_n
        min_b = max(a + 1, rem_d // rem_n + 1)
        max_b = min(2 * rem_d // rem_n, best_max - 1)
        
        for b in range(min_b, max_b + 1):
            if time.time() > deadline:
                break
            # rem2 = rem_n/rem_d - 1/b = (rem_n*b - rem_d) / (rem_d*b)
            rem2_n = rem_n * b - rem_d
            rem2_d = rem_d * b
            if rem2_n <= 0:
                continue
            rem2_n, rem2_d = reduce(rem2_n, rem2_d)
            if rem2_n <= 0:
                continue
            # c = rem2_d / rem2_n must be integer
            if rem2_d % rem2_n == 0:
                c = rem2_d // rem2_n
                if c > b and c <= 10000000 and c < best_max:
                    update_best([a, b, c])

    # 4-term search: a < b < c < d
    # Use divisor-based approach for the last 2 terms
    for a in range(7, 126):
        if time.time() > deadline:
            break
        if a >= best_max:
            break
        # rem = 5/31 - 1/a
        rem_n = 5 * a - 31
        rem_d = 31 * a
        if rem_n <= 0:
            continue
        rem_n, rem_d = reduce(rem_n, rem_d)
        if rem_n <= 0:
            continue
        
        # 1/b + 1/c + 1/d = rem_n/rem_d
        # b > 1/rem, b <= 3/rem
        min_b = max(a + 1, rem_d // rem_n + 1)
        max_b = min(3 * rem_d // rem_n, best_max - 1)
        
        for b in range(min_b, max_b + 1):
            if time.time() > deadline:
                break
            if b >= best_max:
                break
            # rem2 = rem_n/rem_d - 1/b
            rem2_n = rem_n * b - rem_d
            rem2_d = rem_d * b
            if rem2_n <= 0:
                continue
            rem2_n, rem2_d = reduce(rem2_n, rem2_d)
            if rem2_n <= 0:
                continue
            
            # 1/c + 1/d = rem2_n/rem2_d
            # c > 1/rem2, c <= 2/rem2
            min_c = max(b + 1, rem2_d // rem2_n + 1)
            max_c = min(2 * rem2_d // rem2_n, best_max - 1)
            
            for c in range(min_c, max_c + 1):
                if time.time() > deadline:
                    break
                if c >= best_max:
                    break
                # rem3 = rem2_n/rem2_d - 1/c
                rem3_n = rem2_n * c - rem2_d
                rem3_d = rem2_d * c
                if rem3_n <= 0:
                    continue
                rem3_n, rem3_d = reduce(rem3_n, rem3_d)
                if rem3_n <= 0:
                    continue
                # d = rem3_d / rem3_n must be integer
                if rem3_d % rem3_n == 0:
                    d = rem3_d // rem3_n
                    if d > c and d <= 10000000 and d < best_max:
                        update_best([a, b, c, d])

    # 5-term search: a < b < c < d < e
    for a in range(7, 32):
        if time.time() > deadline:
            break
        if a >= best_max:
            break
        rem_n = 5 * a - 31
        rem_d = 31 * a
        if rem_n <= 0:
            continue
        rem_n, rem_d = reduce(rem_n, rem_d)
        if rem_n <= 0:
            continue
        
        min_b = max(a + 1, rem_d // rem_n + 1)
        max_b = min(4 * rem_d // rem_n, best_max - 1)
        
        for b in range(min_b, max_b + 1):
            if time.time() > deadline:
                break
            if b >= best_max:
                break
            rem2_n = rem_n * b - rem_d
            rem2_d = rem_d * b
            if rem2_n <= 0:
                continue
            rem2_n, rem2_d = reduce(rem2_n, rem2_d)
            if rem2_n <= 0:
                continue
            
            min_c = max(b + 1, rem2_d // rem2_n + 1)
            max_c = min(3 * rem2_d // rem2_n, best_max - 1)
            
            for c in range(min_c, max_c + 1):
                if time.time() > deadline:
                    break
                if c >= best_max:
                    break
                rem3_n = rem2_n * c - rem2_d
                rem3_d = rem2_d * c
                if rem3_n <= 0:
                    continue
                rem3_n, rem3_d = reduce(rem3_n, rem3_d)
                if rem3_n <= 0:
                    continue
                
                # 1/d + 1/e = rem3_n/rem3_d
                # d > 1/rem3, d <= 2/rem3
                min_d = max(c + 1, rem3_d // rem3_n + 1)
                max_d = min(2 * rem3_d // rem3_n, best_max - 1)
                
                for d in range(min_d, max_d + 1):
                    if time.time() > deadline:
                        break
                    if d >= best_max:
                        break
                    rem4_n = rem3_n * d - rem3_d
                    rem4_d = rem3_d * d
                    if rem4_n <= 0:
                        continue
                    rem4_n, rem4_d = reduce(rem4_n, rem4_d)
                    if rem4_n <= 0:
                        continue
                    # e = rem4_d / rem4_n must be integer
                    if rem4_d % rem4_n == 0:
                        e = rem4_d // rem4_n
                        if e > d and e <= 10000000 and e < best_max:
                            update_best([a, b, c, d, e])

    # 6-term search (only if best_max is still large)
    if best_max > 200:
        for a in range(7, 32):
            if time.time() > deadline:
                break
            if a >= best_max:
                break
            rem_n = 5 * a - 31
            rem_d = 31 * a
            if rem_n <= 0:
                continue
            rem_n, rem_d = reduce(rem_n, rem_d)
            if rem_n <= 0:
                continue
            
            min_b = max(a + 1, rem_d // rem_n + 1)
            max_b = min(5 * rem_d // rem_n, best_max - 1)
            
            for b in range(min_b, max_b + 1):
                if time.time() > deadline:
                    break
                if b >= best_max:
                    break
                rem2_n = rem_n * b - rem_d
                rem2_d = rem_d * b
                if rem2_n <= 0:
                    continue
                rem2_n, rem2_d = reduce(rem2_n, rem2_d)
                if rem2_n <= 0:
                    continue
                
                min_c = max(b + 1, rem2_d // rem2_n + 1)
                max_c = min(4 * rem2_d // rem2_n, best_max - 1)
                
                for c in range(min_c, max_c + 1):
                    if time.time() > deadline:
                        break
                    if c >= best_max:
                        break
                    rem3_n = rem2_n * c - rem2_d
                    rem3_d = rem2_d * c
                    if rem3_n <= 0:
                        continue
                    rem3_n, rem3_d = reduce(rem3_n, rem3_d)
                    if rem3_n <= 0:
                        continue
                    
                    min_d = max(c + 1, rem3_d // rem3_n + 1)
                    max_d = min(3 * rem3_d // rem3_n, best_max - 1)
                    
                    for d in range(min_d, max_d + 1):
                        if time.time() > deadline:
                            break
                        if d >= best_max:
                            break
                        rem4_n = rem3_n * d - rem3_d
                        rem4_d = rem3_d * d
                        if rem4_n <= 0:
                            continue
                        rem4_n, rem4_d = reduce(rem4_n, rem4_d)
                        if rem4_n <= 0:
                            continue
                        
                        # 1/e + 1/f = rem4_n/rem4_d
                        min_e = max(d + 1, rem4_d // rem4_n + 1)
                        max_e = min(2 * rem4_d // rem4_n, best_max - 1)
                        
                        for e in range(min_e, max_e + 1):
                            if time.time() > deadline:
                                break
                            if e >= best_max:
                                break
                            rem5_n = rem4_n * e - rem4_d
                            rem5_d = rem4_d * e
                            if rem5_n <= 0:
                                continue
                            rem5_n, rem5_d = reduce(rem5_n, rem5_d)
                            if rem5_n <= 0:
                                continue
                            # f = rem5_d / rem5_n must be integer
                            if rem5_d % rem5_n == 0:
                                f = rem5_d // rem5_n
                                if f > e and f <= 10000000 and f < best_max:
                                    update_best([a, b, c, d, e, f])

    # 7-term search (only if best_max is still large)
    if best_max > 500:
        for a in range(7, 20):
            if time.time() > deadline:
                break
            if a >= best_max:
                break
            rem_n = 5 * a - 31
            rem_d = 31 * a
            if rem_n <= 0:
                continue
            rem_n, rem_d = reduce(rem_n, rem_d)
            if rem_n <= 0:
                continue
            
            min_b = max(a + 1, rem_d // rem_n + 1)
            max_b = min(6 * rem_d // rem_n, best_max - 1)
            
            for b in range(min_b, max_b + 1):
                if time.time() > deadline:
                    break
                if b >= best_max:
                    break
                rem2_n = rem_n * b - rem_d
                rem2_d = rem_d * b
                if rem2_n <= 0:
                    continue
                rem2_n, rem2_d = reduce(rem2_n, rem2_d)
                if rem2_n <= 0:
                    continue
                
                min_c = max(b + 1, rem2_d // rem2_n + 1)
                max_c = min(5 * rem2_d // rem2_n, best_max - 1)
                
                for c in range(min_c, max_c + 1):
                    if time.time() > deadline:
                        break
                    if c >= best_max:
                        break
                    rem3_n = rem2_n * c - rem2_d
                    rem3_d = rem2_d * c
                    if rem3_n <= 0:
                        continue
                    rem3_n, rem3_d = reduce(rem3_n, rem3_d)
                    if rem3_n <= 0:
                        continue
                    
                    min_d = max(c + 1, rem3_d // rem3_n + 1)
                    max_d = min(4 * rem3_d // rem3_n, best_max - 1)
                    
                    for d in range(min_d, max_d + 1):
                        if time.time() > deadline:
                            break
                        if d >= best_max:
                            break
                        rem4_n = rem3_n * d - rem3_d
                        rem4_d = rem3_d * d
                        if rem4_n <= 0:
                            continue
                        rem4_n, rem4_d = reduce(rem4_n, rem4_d)
                        if rem4_n <= 0:
                            continue
                        
                        min_e = max(d + 1, rem4_d // rem4_n + 1)
                        max_e = min(3 * rem4_d // rem4_n, best_max - 1)
                        
                        for e in range(min_e, max_e + 1):
                            if time.time() > deadline:
                                break
                            if e >= best_max:
                                break
                            rem5_n = rem4_n * e - rem4_d
                            rem5_d = rem4_d * e
                            if rem5_n <= 0:
                                continue
                            rem5_n, rem5_d = reduce(rem5_n, rem5_d)
                            if rem5_n <= 0:
                                continue
                            
                            # 1/f + 1/g = rem5_n/rem5_d
                            min_f = max(e + 1, rem5_d // rem5_n + 1)
                            max_f = min(2 * rem5_d // rem5_n, best_max - 1)
                            
                            for f in range(min_f, max_f + 1):
                                if time.time() > deadline:
                                    break
                                if f >= best_max:
                                    break
                                rem6_n = rem5_n * f - rem5_d
                                rem6_d = rem5_d * f
                                if rem6_n <= 0:
                                    continue
                                rem6_n, rem6_d = reduce(rem6_n, rem6_d)
                                if rem6_n <= 0:
                                    continue
                                # g = rem6_d / rem6_n must be integer
                                if rem6_d % rem6_n == 0:
                                    g = rem6_d // rem6_n
                                    if g > f and g <= 10000000 and g < best_max:
                                        update_best([a, b, c, d, e, f, g])

    if best_result is not None:
        return best_result

    # Fallback
    return [7, 8, 9, 10000000]
