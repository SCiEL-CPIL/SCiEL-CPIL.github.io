from fractions import Fraction
import time
from math import gcd

_DEADLINE = 0

def construct():
    global _DEADLINE
    start_time = time.time()
    _DEADLINE = start_time + 55
    
    for k in range(8, 257):
        if time.time() > _DEADLINE:
            break
        result = _find_k_terms(k)
        if result:
            return result
            
    return greedy_solution()

def _find_k_terms(k):
    """Find k distinct denominators >= 5 summing to 1 using fast DFS."""
    global _DEADLINE
    
    def dfs(terms_rem, t_num, t_den, min_d, chosen):
        global _DEADLINE
        if time.time() > _DEADLINE:
            return None
        
        if terms_rem == 0:
            return chosen if t_num == 0 else None
        
        if terms_rem == 1:
            if t_num == 0:
                return None
            if t_den % t_num == 0:
                d = t_den // t_num
                if d >= min_d and d <= 10000000:
                    return chosen + [d]
            return None
        
        if terms_rem == 2:
            # Solve 1/a + 1/b = t_num/t_den with a > min_d, b > a
            # 1/a + 1/b = N/D => a+b = ab*N/D => D(a+b) = N*ab => N*ab - D*a - D*b = 0
            # (Na - D)(Nb - D) = D^2
            # Let X = Na - D, Y = Nb - D. Then XY = D^2.
            # a = (X + D) / N, b = (Y + D) / N.
            # We need a, b integers, a > min_d, b > a.
            # X must be a divisor of D^2.
            # Also a > min_d => (X + D) / N > min_d => X > N*min_d - D.
            # And b > a => Y > X.
            # So we iterate over divisors X of D^2.
            # To avoid iterating over all divisors of D^2 (which can be large), 
            # we can iterate x from 1 to sqrt(D^2) = D.
            # But D can be large.
            # Alternative: iterate a from min_d+1 upwards?
            # a must satisfy: 1/a < t_num/t_den => a > t_den/t_num.
            # Also 1/a + 1/(a+1) >= t_num/t_den (max sum for remaining 2 terms starting at a+1? No, b > a).
            # Max sum for 1/a + 1/b with b > a is 1/a + 1/(a+1).
            # So we need 1/a + 1/(a+1) >= t_num/t_den.
            # This gives an upper bound on a.
            
            # Let's try iterating a.
            # a > t_den / t_num
            # a < 1 / (t_num/t_den - 1/(a+1)) ... circular.
            # Use: 1/a + 1/(a+1) >= t_num/t_den
            # => (2a+1)/(a(a+1)) >= t_num/t_den
            # => t_den(2a+1) >= t_num(a^2+a)
            # => t_num*a^2 + (t_num - 2*t_den)*a - t_den <= 0
            # Roots of t_num*a^2 + (t_num - 2*t_den)*a - t_den = 0
            # a = [-(t_num - 2*t_den) + sqrt((t_num - 2*t_den)^2 + 4*t_num*t_den)] / (2*t_num)
            # a_max = ceil( root )
            
            disc = (t_num - 2*t_den)**2 + 4*t_num*t_den
            if disc < 0:
                return None
            # sqrt(disc)
            # Use integer sqrt
            import math
            sqrt_disc = math.isqrt(disc)
            if sqrt_disc * sqrt_disc != disc:
                sqrt_disc += 1 # ceil sqrt
            # a_max = (-(t_num - 2*t_den) + sqrt_disc) // (2*t_num)
            # We need a such that t_num*a^2 + (t_num - 2*t_den)*a - t_den <= 0
            # The positive root is R. a must be <= R.
            # R = (2*t_den - t_num + sqrt_disc) / (2*t_num)
            numerator = 2*t_den - t_num + sqrt_disc
            a_max = numerator // (2*t_num)
            
            a_min = max(min_d, t_den // t_num + 1) # ceil(t_den/t_num)
            
            for a in range(a_min, a_max + 1):
                if time.time() > _DEADLINE:
                    return None
                # b = 1 / (t_num/t_den - 1/a) = 1 / ((t_num*a - t_den)/(t_den*a)) = t_den*a / (t_num*a - t_den)
                num_b = t_den * a
                den_b = t_num * a - t_den
                if den_b <= 0:
                    continue
                if num_b % den_b == 0:
                    b = num_b // den_b
                    if b > a and b <= 10000000:
                        return chosen + [a, b]
            return None

        # Pruning: max sum with smallest denominators must be >= target
        # Use float for speed
        max_s = 0.0
        for i in range(terms_rem):
            max_s += 1.0 / (min_d + i)
        if max_s < t_num / t_den - 1e-15:
            return None
        
        # d bounds
        d_lo = max(min_d, -(-t_den // t_num))  # ceil(t_den/t_num)
        d_hi = min(10000000, (terms_rem * t_den) // t_num + 2)
        
        for d in range(d_lo, d_hi + 1):
            if time.time() > _DEADLINE:
                return None
            
            new_num = t_num * d - t_den
            if new_num <= 0:
                continue
            new_den = t_den * d
            
            # Check if remaining is achievable (float for speed)
            max_rest = 0.0
            for i in range(terms_rem - 1):
                max_rest += 1.0 / (d + 1 + i)
            if new_num / new_den > max_rest + 1e-15:
                break
            
            g = gcd(new_num, new_den)
            if g > 1:
                new_num //= g
                new_den //= g
            
            res = dfs(terms_rem - 1, new_num, new_den, d + 1, chosen + [d])
            if res is not None:
                return res
        
        return None
    
    return dfs(k, 1, 1, 5, [])

def greedy_solution():
    R = Fraction(1, 1)
    D = 5
    denoms = []
    while R > 0:
        inv_R = Fraction(R.denominator, R.numerator)
        d = max(D, inv_R.numerator // inv_R.denominator)
        if (inv_R.numerator % inv_R.denominator) != 0:
            d += 1
        if d < D:
            d = D
        denoms.append(d)
        R -= Fraction(1, d)
        D = d + 1
        if len(denoms) > 256:
            break
    return denoms
