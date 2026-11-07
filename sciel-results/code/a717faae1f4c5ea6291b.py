import math
import time
from itertools import combinations

P = {'n': 210, 'maximum_window': 10210, 'maximum_size': 100}


def _is_square(n):
    if n < 0:
        return False
    r = math.isqrt(n)
    return r * r == n


def _squarefree_kernel(n):
    """Return the squarefree part of n (product of primes with odd exponent)."""
    k = n
    d = 2
    kernel = 1
    while d * d <= k:
        if k % d == 0:
            exp = 0
            while k % d == 0:
                k //= d
                exp += 1
            if exp % 2 == 1:
                kernel *= d
        d += 1 if d == 2 else 2
    if k > 1:
        kernel *= k
    return kernel


def _parity_set(n):
    """Return the set of primes with odd exponent in n's factorization.
    Used for XOR-based matching."""
    k = n
    d = 2
    pset = set()
    while d * d <= k:
        if k % d == 0:
            exp = 0
            while k % d == 0:
                k //= d
                exp += 1
            if exp % 2 == 1:
                pset.add(d)
        d += 1 if d == 2 else 2
    if k > 1:
        pset.add(k)
    return frozenset(pset)


def _xor_pset(p1, p2):
    """Symmetric difference of two frozensets."""
    return frozenset(p1.symmetric_difference(p2))


def _validate(lst, n=210, max_window=10210, max_size=100):
    """Check all constraints."""
    if not isinstance(lst, (list, tuple)):
        return False
    if len(lst) < 1 or len(lst) > max_size:
        return False
    if len(set(lst)) != len(lst):
        return False
    for x in lst:
        if not isinstance(x, int) or isinstance(x, bool):
            return False
        if x <= n or x > max_window:
            return False
    prod = n
    for x in lst:
        prod *= x
    return _is_square(prod)


def _find_best_pair_algebraic(deadline):
    """Find a pair (a, b) such that 210 * a * b is a perfect square.
    Minimizes max(a, b).
    Uses the property that a * b = 210 * t^2 for some integer t.
    """
    n = P['n']
    max_window = P['maximum_window']
    best_val = float('inf')
    best_pair = None
    
    # t=1 gives a*b = 210. Factors: (1,210), (2,105), (3,70), (5,42), (6,35), (7,30), (10,21), (14,15).
    # None have both > 210.
    
    # We need a > 210 and b > 210.
    # So a*b > 210*210 = 44100.
    # 210 * t^2 > 44100 => t^2 > 210 => t >= 15.
    
    for t in range(1, 1000):
        if time.time() > deadline:
            break
            
        M = n * t * t
        
        # If the smallest possible max(a,b) for this M is already worse than best_val, stop.
        # min max(a,b) is ceil(sqrt(M)).
        sqrt_M = math.isqrt(M)
        if sqrt_M * sqrt_M != M:
            sqrt_M += 1
        # If ceil(sqrt(M)) - n >= best_val, then any pair from this M will have max(a,b) >= ceil(sqrt(M)) > n + best_val
        # So max(a,b) - n >= best_val. We want strictly smaller.
        if sqrt_M - n >= best_val:
            break
            
        # Search for factors a, b of M close to sqrt(M)
        # Start from floor(sqrt(M)) and go down
        a = math.isqrt(M)
        while a > n:
            if M % a == 0:
                b = M // a
                if b > n and b != a:
                    # Valid pair
                    val = max(a, b) - n
                    if val < best_val:
                        best_val = val
                        best_pair = [a, b]
                        # If we found a very good one, we might stop early?
                        # But we need to be sure.
                        # Since we iterate t increasing, M increases, sqrt(M) increases.
                        # So subsequent t will have larger min-max.
                        # So if we find a pair, it's the best for this t.
                        # But could a larger t have a smaller max(a,b)?
                        # No, because min max(a,b) for M is sqrt(M).
                        # And sqrt(M) increases with t.
                        # So the first valid pair found (smallest t) might not be the global min max(a,b)
                        # if a larger t allows a pair very close to sqrt(M) while small t had no pair close to sqrt(M).
                        # Example: t=15, M=210*225=47250. sqrt=217. 
                        # If no factors near 217, max(a,b) could be large.
                        # t=16, M=210*256=53760. sqrt=231.
                        # If factors near 231, max(a,b) ~ 231.
                        # So we cannot stop at first t with a solution.
                        # We must continue until sqrt(M) - n >= best_val.
                        pass
            a -= 1
            if a <= n:
                break
                
    if best_pair:
        return best_pair, best_val
    return None, float('inf')


def construct():
    t0 = time.time()
    deadline = t0 + 80.0  # safety margin before 90s kill

    n = P['n']
    
    # Baseline: single element 210 * 4 = 840 (value 630)
    # Or 210 * 2^2 = 840.
    # 210 * s^2. s=2 -> 840. val = 630.
    # s=3 -> 1890. val = 1680.
    # Smallest single element is 840.
    best_val = 630
    best_list = [840]
    
    # Try algebraic pair search
    pair, val = _find_best_pair_algebraic(deadline)
    if pair and _validate(pair):
        if val < best_val:
            best_val = val
            best_list = pair
    
    # If we have time, try triples?
    # Triples are harder to optimize algebraically in the same simple way.
    # But pairs are likely sufficient to beat 630 significantly.
    # The previous score was 21, which is much better than 630.
    # Wait, the score is 21.0. This implies max(a,b) - 210 = 21.
    # So max(a,b) = 231.
    # My algebraic search should find this quickly.
    
    # Let's check if t=15 works.
    # M = 210 * 225 = 47250.
    # sqrt(47250) ≈ 217.37.
    # Factors of 47250:
    # 47250 / 210 = 225.
    # 47250 = 2 * 3 * 5^3 * 7 * 3^2 = 2 * 3^3 * 5^3 * 7.
    # We need a, b > 210.
    # If a=210, b=225. But a must be > 210.
    # Next factor?
    # Divisors of 47250.
    # 47250 / 225 = 210.
    # 47250 / 210 = 225.
    # 47250 / 189 = 250. (189 < 210)
    # 47250 / 250 = 189.
    # 47250 / 210 is not integer? 47250/210 = 225.
    # Is there a pair > 210?
    # 210 * 225.
    # Factors:
    # 47250 / 210 = 225.
    # 47250 / 216? No.
    # Let's check t=16.
    # M = 210 * 256 = 53760.
    # sqrt(53760) ≈ 231.86.
    # 53760 / 231 ? 53760/231 = 232.7. No.
    # 53760 / 232 ? 53760/232 = 231.7. No.
    # 53760 / 240 = 224. (224 < 210? No, 224 > 210).
    # So a=224, b=240.
    # max(224, 240) = 240.
    # val = 240 - 210 = 30.
    # This is better than 630.
    # Can we do better?
    # t=15 gave no pair > 210?
    # 47250 factors:
    # 210, 225.
    # 189, 250.
    # 175, 270.
    # 150, 315.
    # 140, 337.5 (No)
    # 125, 378.
    # 105, 450.
    # 90, 525.
    # 75, 630.
    # 70, 675.
    # 63, 750.
    # 50, 945.
    # 45, 1050.
    # 42, 1125.
    # 35, 1350.
    # 30, 1575.
    # 25, 1890.
    # 21, 2250.
    # 18, 2625.
    # 15, 3150.
    # 14, 3375.
    # 10, 4725.
    # 9, 5250.
    # 7, 6750.
    # 6, 7875.
    # 5, 9450.
    # 3, 15750.
    # 2, 23625.
    # 1, 47250.
    # None have both > 210 except (210, 225) which fails a>210.
    
    # So t=16 gives 30.
    # Can we get 21?
    # max(a,b) = 231.
    # a*b = 210 * t^2.
    # If max is 231, min is 231 or less.
    # Product approx 231^2 = 53361.
    # 210 * t^2 ≈ 53361 => t^2 ≈ 254 => t ≈ 15.9.
    # So t=16 is the likely candidate.
    # For t=16, M=53760.
    # We found 224, 240. Max 240. Val 30.
    # Are there other factors?
    # 53760 / 224 = 240.
    # 53760 / 231? No.
    # 53760 / 232? No.
    # 53760 / 235? No.
    # 53760 / 240 = 224.
    # 53760 / 252? 53760/252 = 213.3.
    # 53760 / 210 = 256. (210 not > 210).
    # 53760 / 216? 53760/216 = 248.8.
    # 53760 / 224 = 240.
    # 53760 / 225? No.
    # 53760 / 228? No.
    # 53760 / 231? No.
    # 53760 / 232? No.
    # 53760 / 235? No.
    # 53760 / 236? No.
    # 53760 / 240 = 224.
    # 53760 / 245? No.
    # 53760 / 252? No.
    # 53760 / 256 = 210.
    # 53760 / 280 = 192.
    # 53760 / 288 = 186.6.
    # 53760 / 315? No.
    # 53760 / 320 = 168.
    # 53760 / 336 = 160.
    # 53760 / 360 = 149.3.
    # 53760 / 378? No.
    # 53760 / 384 = 140.
    # 53760 / 420 = 128.
    # 53760 / 448 = 120.
    # 53760 / 480 = 112.
    # 53760 / 504 = 106.6.
    # 53760 / 560 = 96.
    # 53760 / 630? No.
    # 53760 / 672 = 80.
    # 53760 / 720 = 74.6.
    # 53760 / 768 = 70.
    # 53760 / 840 = 64.
    # 53760 / 896 = 60.
    # 53760 / 960 = 56.
    # 53760 / 1008 = 53.3.
    # 53760 / 1120 = 48.
    # 53760 / 1260? No.
    # 53760 / 1344 = 40.
    # 53760 / 1440 = 37.3.
    # 53760 / 1568? No.
    # 53760 / 1680 = 32.
    # 53760 / 1920 = 28.
    # 53760 / 2016 = 26.6.
    # 53760 / 2240 = 24.
    # 53760 / 2520 = 21.3.
    # 53760 / 2688 = 20.
    # 53760 / 3360 = 16.
    # 53760 / 3840 = 14.
    # 53760 / 4480 = 12.
    # 53760 / 5040 = 10.6.
    # 53760 / 5600 = 9.6.
    # 53760 / 6720 = 8.
    # 53760 / 8960 = 6.
    # 53760 / 11200 = 4.8.
    # 53760 / 13440 = 4.
    # 53760 / 17920 = 3.
    # 53760 / 26880 = 2.
    # 53760 / 53760 = 1.
    
    # It seems 224, 240 is the best for t=16.
    # What about t=17?
    # M = 210 * 289 = 60690.
    # sqrt = 246.3.
    # 60690 / 246 ? No.
    # 60690 / 245 ? No.
    # 60690 / 240 ? No.
    # 60690 / 238 ? No.
    # 60690 / 210 = 289.
    # 60690 / 238? 238 * 255 = 60690.
    # 238 > 210, 255 > 210.
    # Max 255. Val 45. Worse.
    
    # What about t=14?
    # M = 210 * 196 = 41160.
    # sqrt = 202.8.
    # 41160 / 210 = 196. (196 < 210).
    # 41160 / 196 = 210.
    # 41160 / 195? No.
    # 41160 / 192? No.
    # 41160 / 189? No.
    # 41160 / 180? No.
    # 41160 / 175? No.
    # 41160 / 168 = 245.
    # 168 < 210.
    # 41160 / 150? No.
    # 41160 / 140 = 294.
    # 140 < 210.
    # 41160 / 126 = 326.6.
    # 41160 / 120 = 343.
    # 120 < 210.
    # 41160 / 105 = 392.
    # 105 < 210.
    # 41160 / 98 = 420.
    # 98 < 210.
    # 41160 / 84 = 490.
    # 84 < 210.
    # 41160 / 70 = 588.
    # 70 < 210.
    # 41160 / 60 = 686.
    # 60 < 210.
    # 41160 / 56 = 735.
    # 56 < 210.
    # 41160 / 49 = 840.
    # 49 < 210.
    # 41160 / 42 = 980.
    # 42 < 210.
    # 41160 / 35 = 1176.
    # 35 < 210.
    # 41160 / 30 = 1372.
    # 30 < 210.
    # 41160 / 28 = 1470.
    # 28 < 210.
    # 41160 / 24 = 1715.
    # 24 < 210.
    # 41160 / 21 = 1960.
    # 21 < 210.
    # 41160 / 20 = 2058.
    # 20 < 210.
    # 41160 / 15 = 2744.
    # 15 < 210.
    # 41160 / 14 = 2940.
    # 14 < 210.
    # 41160 / 12 = 3430.
    # 12 < 210.
    # 41160 / 10 = 4116.
    # 10 < 210.
    # 41160 / 7 = 5880.
    # 7 < 210.
    # 41160 / 6 = 6860.
    # 6 < 210.
    # 41160 / 5 = 8232.
    # 5 < 210.
    # 41160 / 3 = 13720.
    # 3 < 210.
    # 41160 / 2 = 20580.
    # 2 < 210.
    # 41160 / 1 = 41160.
    # 1 < 210.
    
    # So t=16 giving 30 seems to be the best pair.
    # The current score is 21.
    # How did they get 21?
    # Maybe a triple?
    # Or maybe I missed a pair.
    # Let's check t=15 again.
    # M = 47250.
    # Factors:
    # 210, 225.
    # 189, 250.
    # 175, 270.
    # 150, 315.
    # 140, 337.5.
    # 125, 378.
    # 105, 450.
    # 90, 525.
    # 75, 630.
    # 70, 675.
    # 63, 750.
    # 50, 945.
    # 45, 1050.
    # 42, 1125.
    # 35, 1350.
    # 30, 1575.
    # 25, 1890.
    # 21, 2250.
    # 18, 2625.
    # 15, 3150.
    # 14, 3375.
    # 10, 4725.
    # 9, 5250.
    # 7, 6750.
    # 6, 7875.
    # 5, 9450.
    # 3, 15750.
    # 2, 23625.
    # 1, 47250.
    
    # No pair > 210.
    
    # Maybe t=13?
    # M = 210 * 169 = 35490.
    # sqrt = 188.3.
    # Max(a,b) >= 189.
    # 35490 / 210 = 169.
    # 35490 / 195? No.
    # 35490 / 189? No.
    # 35490 / 180? No.
    # 35490 / 175? No.
    # 35490 / 169? No.
    # 35490 / 150? No.
    # 35490 / 140? No.
    # 35490 / 130? No.
    # 35490 / 126? No.
    # 35490 / 120? No.
    # 35490 / 105? No.
    # 35490 / 98? No.
    # 35490 / 90? No.
    # 35490 / 78? No.
    # 35490 / 70? No.
    # 35490 / 65? No.
    # 35490 / 60? No.
    # 35490 / 56? No.
    # 35490 / 52? No.
    # 35490 / 42? No.
    # 35490 / 39? No.
    # 35490 / 30? No.
    # 35490 / 26? No.
    # 35490 / 21? No.
    # 35490 / 15? No.
    # 35490 / 13? No.
    # 35490 / 10? No.
    # 35490 / 7? No.
    # 35490 / 6? No.
    # 35490 / 5? No.
    # 35490 / 3? No.
    # 35490 / 2? No.
    # 35490 / 1? No.
    
    # It seems 30 is the best pair.
    # The score 21 must come from a triple or I am missing something.
    # But 30 is a significant improvement over 630.
    # And 30 is likely the best pair.
    
    # I will implement the algebraic pair search.
    # It is fast and should find 30.
    
    return [int(x) for x in best_list]
