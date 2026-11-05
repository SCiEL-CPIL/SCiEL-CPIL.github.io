from fractions import Fraction
import time
import math

def construct():
    start_time = time.time()
    deadline = start_time + 50  # 50s safety margin
    
    # Try k=5 first with optimized search
    # We need 1/a + 1/b + 1/c + 1/d + 1/e = 1 with 5 <= a < b < c < d < e <= 10000000
    
    def solve_three_terms(p, q, min_d):
        """
        Find distinct integers d, e > min_d such that 1/d + 1/e = p/q
        where p/q is in lowest terms.
        Returns (d, e) or None.
        """
        if p <= 0 or q <= 0:
            return None
        
        # 1/d + 1/e = p/q
        # q(d+e) = p*de
        # p*de - q*d - q*e = 0
        # Multiply by p: p^2*de - pq*d - pq*e = 0
        # (pd - q)(pe - q) = q^2
        # Let x = pd - q, y = pe - q
        # Then x*y = q^2
        # d = (x + q) / p, e = (y + q) / p
        # We need d, e to be integers, distinct, > min_d, and in range
        
        q2 = q * q
        
        # Find all factor pairs (x, y) with x * y = q2, x <= y
        # x must divide q2
        # d = (x + q) / p must be integer, so (x + q) % p == 0
        # Similarly for e
        
        best = None
        # Iterate over divisors x of q2 where x <= sqrt(q2)
        sqrt_q2 = int(math.isqrt(q2))
        
        for x in range(1, sqrt_q2 + 1):
            if q2 % x != 0:
                continue
            y = q2 // x
            
            # Check x first
            if (x + q) % p == 0:
                d = (x + q) // p
                if d > min_d and d <= 10000000:
                    if (y + q) % p == 0:
                        e = (y + q) // p
                        if e > d and e <= 10000000:
                            return (d, e)
            
            # Check y first (if y != x)
            if y != x:
                if (y + q) % p == 0:
                    d = (y + q) // p
                    if d > min_d and d <= 10000000:
                        if (x + q) % p == 0:
                            e = (x + q) // p
                            if e > d and e <= 10000000:
                                return (d, e)
        
        return None
    
    # Search for 5-term solution
    # a from 5, b from a+1, c from b+1
    # For each (a,b,c), compute remaining = 1 - 1/a - 1/b - 1/c
    # Then solve 1/d + 1/e = remaining
    
    for a in range(5, 10):  # a is small
        if time.time() > deadline:
            break
        for b in range(a + 1, 20):  # b is also relatively small
            if time.time() > deadline:
                break
            # 1/a + 1/b < 1 required
            if a * b >= a + b:  # 1/a + 1/b >= 1, impossible
                continue
            for c in range(b + 1, 50):  # c is moderate
                if time.time() > deadline:
                    break
                # Compute remaining = 1 - 1/a - 1/b - 1/c
                # = (abc - bc - ac - ab) / abc
                num = a * b * c - b * c - a * c - a * b
                den = a * b * c
                if num <= 0:
                    break  # remaining <= 0, too large c
                if num >= den:
                    continue  # remaining >= 1, not possible for two more terms
                
                # Reduce fraction
                g = math.gcd(num, den)
                p = num // g
                q = den // g
                
                # Now solve 1/d + 1/e = p/q with d > c
                res = solve_three_terms(p, q, c + 1)
                if res:
                    d, e = res
                    return [a, b, c, d, e]
    
    # Fallback: try k=6 with a simpler approach
    # Known 6-term solutions exist
    # Let's use a known one: 1/5+1/6+1/7+1/10+1/14+1/35... let me verify
    # Actually, let me just return a known valid 5-term solution
    # 1/5 + 1/6 + 1/7 + 1/10 + 1/210
    # = 42/210 + 35/210 + 30/210 + 21/210 + 1/210
    # = (42+35+30+21+1)/210 = 129/210... not 1
    
    # Let me find a valid one:
    # 1/5 + 1/6 + 1/7 + 1/14 + 1/35
    # = 42/210 + 35/210 + 30/210 + 15/210 + 6/210 = 128/210... no
    
    # 1/5 + 1/6 + 1/7 + 1/10 + 1/42 + 1/... 
    # Let me just use the DFS fallback but with integer arithmetic
    
    # Actually, let me try a different known decomposition:
    # 1 = 1/5 + 1/6 + 1/7 + 1/10 + 1/210 + 1/... no
    
    # Let me compute: 1 - 1/5 - 1/6 - 1/7 - 1/10
    # = 1 - (42+35+30+21)/210 = 1 - 128/210 = 82/210 = 41/105
    # So we need 1/d + 1/e = 41/105
    # (41d - 105)(41e - 105) = 105^2 = 11025
    # Divisors of 11025: 1, 3, 5, 7, 9, 15, 21, 25, 35, 45, 49, 63, 75, 105, 147, 175, 225, 315, 441, 525, 735, 1225, 1575, 2450, 3675, 11025
    # x = 41d - 105, so d = (x+105)/41
    # Try x=105: d = 210/41 not int
    # x=441: d = 546/41 not int  
    # x=735: d = 840/41 not int
    # x=1225: d = 1330/41 not int
    # x=2450: d = 2555/41 not int
    # x=3675: d = 3780/41 not int
    # x=11025: d = 11130/41 = 271.46... not int
    
    # Let me try a different set. Use the search but with better bounds.
    
    # Fallback: use a known valid solution
    # 1/5 + 1/6 + 1/7 + 1/10 + 1/14 + 1/35 = ?
    # LCD = 210: 42+35+30+21+15+6 = 149/210... no
    
    # 1/5 + 1/6 + 1/7 + 1/10 + 1/14 + 1/35
    # Let me use a computational approach for fallback
    
    # Simple fallback: return a known valid 6-term solution
    # 1/5 + 1/6 + 1/7 + 1/10 + 1/14 + 1/35... 
    # Actually, let me just do a quick DFS for k=6 with integer arithmetic
    
    def dfs_int(rem_num, rem_den, min_d, depth, max_depth, path):
        if time.time() > deadline:
            return None
        if depth == max_depth:
            if rem_num == 0:
                return path[:]
            return None
        if rem_num == 0:
            return path[:]
        if rem_num <= 0:
            return None
        
        # rem = rem_num / rem_den
        # 1/d < rem => d > rem_den / rem_num
        # d >= min_d
        # Also need rem <= (max_depth - depth) / min_d
        max_left = max_depth - depth
        if rem_num * min_d > rem_den * max_left:
            return None
        
        d_lower = max(min_d, rem_den // rem_num + 1)
        # 1/d >= rem / max_left => d <= max_left * rem_den / rem_num
        d_upper = (max_left * rem_den) // rem_num
        if d_upper > 10000000:
            d_upper = 10000000
        
        for d in range(d_lower, min(d_upper + 1, d_lower + 100)):  # Limit iterations
            if d in path:
                continue
            # new_rem = rem_num/rem_den - 1/d = (rem_num*d - rem_den) / (rem_den*d)
            new_num = rem_num * d - rem_den
            if new_num < 0:
                break
            new_den = rem_den * d
            g = math.gcd(new_num, new_den)
            if g > 1:
                new_num //= g
                new_den //= g
            
            res = dfs_int(new_num, new_den, d + 1, depth + 1, max_depth, path + [d])
            if res:
                return res
        return None
    
    res = dfs_int(1, 1, 5, 0, 6, [])
    if res:
        return res
    
    # Final fallback: return a hardcoded valid solution
    # Let me verify: 1/5+1/6+1/7+1/10+1/14+1/35
    # = (42+35+30+21+15+6)/210 = 149/210... no
    
    # Try: 1/5+1/6+1/7+1/10+1/15+1/42
    # LCD = 210: 42+35+30+21+14+5 = 147/210... no
    
    # I'll just raise an error if we get here, but the search should find a solution
    raise RuntimeError("no solution found")

