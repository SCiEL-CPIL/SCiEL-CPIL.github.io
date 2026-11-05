import time
from fractions import Fraction
from math import gcd

def construct():
    start_time = time.time()
    deadline = start_time + 85.0
    
    target = Fraction(5, 31)
    best_max = float('inf')
    best_solution = None
    
    def update_best(sol):
        nonlocal best_max, best_solution
        if not sol:
            return
        if len(set(sol)) != len(sol):
            return
        if any(d < 1 or d > 10000000 for d in sol):
            return
        s = sum(Fraction(1, d) for d in sol)
        if s == target:
            m = max(sol)
            if m < best_max:
                best_max = m
                best_solution = sorted(sol)
    
    def get_divisors(n):
        if n == 0:
            return []
        divisors = set([1])
        i = 2
        while i * i <= n:
            if n % i == 0:
                while n % i == 0:
                    n //= i
                for d in list(divisors):
                    power = 1
                    while d * power <= n * 10**10:  # safe bound
                        divisors.add(d * power)
                        # Check if we can multiply by more i
                        if (d * power) * i <= n * 10**10:
                            divisors.add(d * power * i)
                        power *= i
            i += 1
        if n > 1:
            for d in list(divisors):
                divisors.add(d * n)
        return sorted(divisors)
    
    def solve_2_term(p, q):
        """Find (x,y) with x<y, 1/x+1/y=p/q, p/q in lowest terms."""
        if p == 0:
            return []
        q2 = q * q
        divs = get_divisors(q2)
        solutions = []
        for u in divs:
            if q2 % u != 0:
                continue
            v = q2 // u
            if (u + q) % p == 0 and (v + q) % p == 0:
                x = (u + q) // p
                y = (v + q) // p
                if x < 1 or y < 1:
                    continue
                if x == y:
                    continue
                if x > y:
                    x, y = y, x
                solutions.append((x, y))
        return solutions
    
    # Phase 1: 3-term search - exhaustive
    # 1/a + 1/b + 1/c = 5/31, a < b < c
    # a >= 7 (since 1/6 > 5/31)
    # For given a, b: c = 31ab / (5ab - 31a - 31b)
    # Need 5ab - 31a - 31b > 0 and divides 31ab
    
    max_a = 20  # 1/20 is small enough
    for a in range(7, max_a + 1):
        if time.time() > deadline:
            break
        rem_a = target - Fraction(1, a)
        if rem_a <= 0:
            continue
        # 1/b + 1/c = rem_a = p/q
        # b > a, and b < 2/rem_a (since 1/b + 1/c < 2/b <= rem_a means b > 2/rem_a... wait)
        # Actually 1/b + 1/c = rem_a, with b < c, so 1/b > rem_a/2, thus b < 2/rem_a
        # And b > a
        p = rem_a.numerator
        q = rem_a.denominator
        g = gcd(p, q)
        p //= g
        q //= g
        
        # b must satisfy: b > a and b < 2*q/p (since 1/b > rem_a/2)
        # Also from (pb-q)(pc-q) = q^2, we need pb > q, so b > q/p
        b_min = max(a + 1, q // p + 1)
        b_max = min(2 * q // p, 500)  # cap for safety
        
        for b in range(b_min, b_max + 1):
            if time.time() > deadline:
                break
            rem_ab = rem_a - Fraction(1, b)
            if rem_ab <= 0:
                continue
            # c = 1 / rem_ab
            if rem_ab.numerator != 1:
                continue
            c = rem_ab.denominator
            if c > b:
                update_best([a, b, c])
    
    # Phase 2: 4-term search
    # 1/a + 1/b + 1/c + 1/d = 5/31, a < b < c < d
    for a in range(7, 12):
        if time.time() > deadline:
            break
        rem_a = target - Fraction(1, a)
        if rem_a <= 0:
            continue
        for b in range(a + 1, 25):
            if time.time() > deadline:
                break
            rem_ab = rem_a - Fraction(1, b)
            if rem_ab <= 0:
                continue
            p = rem_ab.numerator
            q = rem_ab.denominator
            g = gcd(p, q)
            p //= g
            q //= g
            # c > b, c < 2*q/p
            c_min = b + 1
            c_max = min(2 * q // p, 80)
            for c in range(c_min, c_max + 1):
                if time.time() > deadline:
                    break
                rem_abc = rem_ab - Fraction(1, c)
                if rem_abc <= 0:
                    continue
                if rem_abc.numerator != 1:
                    continue
                d = rem_abc.denominator
                if d > c:
                    update_best([a, b, c, d])
    
    # Phase 3: 5-term search (limited)
    for a in range(7, 10):
        if time.time() > deadline:
            break
        rem_a = target - Fraction(1, a)
        if rem_a <= 0:
            continue
        for b in range(a + 1, 15):
            if time.time() > deadline:
                break
            rem_ab = rem_a - Fraction(1, b)
            if rem_ab <= 0:
                continue
            for c in range(b + 1, 25):
                if time.time() > deadline:
                    break
                rem_abc = rem_ab - Fraction(1, c)
                if rem_abc <= 0:
                    continue
                p = rem_abc.numerator
                q = rem_abc.denominator
                g = gcd(p, q)
                p //= g
                q //= g
                d_min = c + 1
                d_max = min(2 * q // p, 50)
                for d in range(d_min, d_max + 1):
                    if time.time() > deadline:
                        break
                    rem_abcd = rem_abc - Fraction(1, d)
                    if rem_abcd <= 0:
                        continue
                    if rem_abcd.numerator != 1:
                        continue
                    e = rem_abcd.denominator
                    if e > d:
                        update_best([a, b, c, d, e])
    
    if best_solution is not None:
        return best_solution
    
    # Fallback
    return [8, 31, 248]

# EVOLVE-BLOCK-END
