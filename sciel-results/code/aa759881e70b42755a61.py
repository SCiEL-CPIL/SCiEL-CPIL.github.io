import time
import math
import sys
from collections import defaultdict

def construct():
    deadline = time.time() + 55.0  # safety margin before 60s
    n = 210
    max_val = 10210

    def is_square(m):
        if m < 0:
            return False
        s = math.isqrt(m)
        return s * s == m

    # Key insight: For a set S of distinct integers in (210, 10210],
    # product(S) * 210 must be a perfect square.
    #
    # For pairs: a*b*210 = s^2 => a*b = 210*t^2 where s=210*t
    # We need 210 < a < b <= 10210, a*b = 210*t^2
    #
    # For triples: a*b*c*210 = s^2
    # More generally, we can think in terms of square-free kernels.
    #
    # Strategy 1: Pairs via factorization of 210*t^2
    # Strategy 2: Triples via factorization
    # Strategy 3: Use the structure that sf(a)*sf(b) = 210 (mod squares) for pairs
    #
    # For pairs: a*b = 210*t^2
    # To minimize max(a,b), we want a and b close to sqrt(210)*t
    # 
    # Let's precompute: for each t, M = 210*t^2
    # We need divisors a of M with 210 < a < b = M/a <= 10210
    # So a must be in (210, min(sqrt(M), M/10210))
    # And b = M/a must be > a and <= 10210 and > 210
    #
    # To minimize b (which is max(a,b) since b>a), we want a as large as possible
    # (close to sqrt(M)), subject to a dividing M and b <= 10210.
    #
    # Better approach: iterate a from 211 upward, and for each a, check if there
    # exists b such that a*b*210 is a square.
    # a*b*210 = s^2 => b = s^2/(210*a) must be integer > a, > 210, <= 10210
    # => 210*a must divide s^2, and b = s^2/(210*a)
    #
    # Alternative: For each a in (210, 10210], compute sf(a) (square-free part).
    # Then we need sf(a) * sf(b) * 210 to be a square, i.e., sf(a)*sf(b) = 210 (since sf(210)=210)
    # Wait: sf(a*b*210) = 1 means the XOR of prime exponents mod 2 is 0.
    # Let's represent each number by its square-free kernel as a bitmask of primes.
    # But primes up to 10210... that's too many for a bitmask.
    #
    # Simpler: just iterate t and find factorizations.
    #
    # For efficiency: iterate t from 1 upward. For each t, M = 210*t^2.
    # Find all divisors a of M in range (210, 10210] such that b=M/a is also in (210, 10210] and a<b.
    # The best pair for this t has the smallest b, which means the largest a (closest to sqrt(M)).
    #
    # Even better: for each a in (211, 10210], we can compute what b values work.
    # a*b = 210*t^2 => b = 210*t^2/a must be integer.
    # So a must divide 210*t^2.
    #
    # Let me just do the t-iteration but more efficiently.

    best_max = max_val + 1
    best_result = None

    # Try pairs first
    # For t from 1 upward:
    #   M = 210 * t^2
    #   sqrt_M = isqrt(M)
    #   a must be in (max(211, ceil(M/10210)), min(sqrt_M, 10210)]
    #   and M % a == 0
    #   b = M // a, need b > a, b <= 10210, b > 210
    #   To minimize b, pick largest valid a.

    # Precompute divisors more efficiently
    # For a given M, to find the largest divisor a <= sqrt(M) with a > 210:
    # Iterate downward from min(isqrt(M), 10210) to max(211, ceil(M/10210))

    t = 1
    # We can bound t: b = M/a >= M/10210 > 210 => M > 210*10210 => 210*t^2 > 2144100 => t^2 > 10210 => t > 101
    # Actually b > 210 means M/a > 210, which is usually satisfied.
    # The real constraint is a > 210 and b <= 10210.
    # a > 210 and a <= sqrt(M) => M > 210^2 = 44100 => t^2 > 210 => t > 14
    # b = M/a <= 10210 => a >= M/10210
    # So we need max(211, ceil(M/10210)) <= min(isqrt(M), 10210)
    
    while time.time() < deadline:
        M = 210 * t * t
        sqrt_M = math.isqrt(M)
        
        # a must satisfy: max(211, ceil(M/10210)) <= a <= min(sqrt_M, 10210)
        lo = max(211, (M + 10209) // 10210)  # ceil(M/10210)
        hi = min(sqrt_M, 10210)
        
        if lo <= hi:
            # Search for divisors from hi downward (to find largest a first, giving smallest b)
            found_for_t = False
            for a in range(hi, lo - 1, -1):
                if time.time() >= deadline:
                    break
                if M % a == 0:
                    b = M // a
                    if b > a and b <= 10210 and b > 210:
                        candidate_max = b
                        if candidate_max < best_max:
                            best_max = candidate_max
                            best_result = (a, b)
                        found_for_t = True
                        # Since we're going downward in a, b = M/a is going upward.
                        # So the first valid a (largest a) gives the smallest b for this t.
                        # We can break since we want the smallest b for this t.
                        break
            if found_for_t and best_max - 210 <= 5:
                break
        
        t += 1
        if t > 20000:
            break

    # Also try 3-element sets: a*b*c*210 = s^2
    # This is harder. Let's think: a*b*c = s^2/210 = 210*t^2 (with s=210*t)
    # So we need a*b*c = 210*t^2 with 210 < a < b < c <= 10210
    # To minimize c, we want a and b as large as possible (close to cube root).
    
    # For 3 elements, it's more complex. Let's try a different approach:
    # Fix a and b, then c = 210*t^2/(a*b) must be integer > b, <= 10210.
    # This is a 2D search which is expensive.
    
    # Alternative for 3 elements: 
    # a*b*c*210 = s^2
    # Let's parameterize: pick a, b in (210, 10210], then c = s^2/(210*a*b)
    # For c to be integer, 210*a*b must divide s^2.
    # This is complex. Let's skip 3 elements for now and focus on optimizing pairs.
    
    # Actually, let's try a smarter pair search:
    # For each a in (211, 10210], what b values make a*b*210 a square?
    # a*b*210 = s^2 => b = s^2/(210*a)
    # s must be a multiple of sqrt(210*a/gcd(210*a, ...))... this is getting complex.
    
    # Let's stick with the t-iteration but also try starting from different angles.
    
    # Another approach: enumerate a from 211 to 10210, and for each a,
    # compute the "required" sf(b) such that sf(a)*sf(b)*210 is a square.
    # sf(a)*sf(b)*210 = square => sf(a)*sf(b) = 210 (in the sense of square-free multiplication)
    # More precisely: let a = x^2 * sa, b = y^2 * sb where sa, sb are square-free.
    # Then a*b*210 = x^2*y^2*sa*sb*210. For this to be a square, sa*sb*210 must be a square.
    # Since sa, sb, 210 are square-free, sa*sb*210 is a square iff sa*sb = 210 (as square-free numbers).
    # So we need sf(a) * sf(b) = 210 (with cancellation of common factors giving a square).
    # Actually: sf(a)*sf(b)*210 = square. Let sf(a)=A, sf(b)=B, both square-free.
    # A*B*210 = square. Since A, B, 210 are square-free, this means the product A*B*210
    # has all prime exponents even.
    # 210 = 2*3*5*7.
    # So A*B must have the same prime factors as 210 (with odd exponents matching).
    # Specifically, for each prime p in {2,3,5,7}, the exponent in A*B must be 1 (odd) to make
    # A*B*210 have even exponent (1+1=2).
    # For primes not in {2,3,5,7}, the exponent in A*B must be 0 (even).
    # So A*B = 210 * (square of some number)... no, A and B are square-free, so A*B is square-free
    # only if gcd(A,B)=1. If they share factors, those cancel.
    # This is getting complicated. The t-iteration approach is cleaner.
    
    if best_result is None:
        # Fallback: single element 840 (840*210 = 176400 = 420^2)
        return [840]
    
    a, b = best_result
    prod = a * b * 210
    if is_square(prod):
        return [int(a), int(b)]
    else:
        return [840]
