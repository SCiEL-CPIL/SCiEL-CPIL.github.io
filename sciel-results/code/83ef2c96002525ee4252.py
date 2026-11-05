import time
from fractions import Fraction
from math import gcd

def construct():
    start_time = time.time()
    deadline = start_time + 85  # Safety margin before 90s limit

    def time_check():
        return time.time() > deadline

    # Try 3-term: 1/a + 1/b + 1/c = 1, a < b < c, a >= 5
    # For each a >= 5: 1/b + 1/c = 1 - 1/a = (a-1)/a
    # c = ab/(ab - a - b)
    for a in range(5, 100):
        if time_check():
            break
        for b in range(a + 1, 10000):
            if time_check():
                break
            num = a * b
            den = a * b - a - b
            if den <= 0:
                continue
            if num % den == 0:
                c = num // den
                if c > b and c >= 5 and c <= 10000000:
                    # Verify using Fraction
                    if Fraction(1, a) + Fraction(1, b) + Fraction(1, c) == 1:
                        return [a, b, c]

    # Try 4-term: 1/a + 1/b + 1/c + 1/d = 1
    for a in range(5, 20):
        if time_check():
            break
        for b in range(a + 1, 100):
            if time_check():
                break
            rem_ab = Fraction(1) - Fraction(1, a) - Fraction(1, b)
            if rem_ab <= 0:
                continue
            # 1/c + 1/d = rem_ab
            # d = c / (c*rem_ab - 1)
            # But let's use the formula: c = ab/(ab - a - b) type
            # Actually, for two terms summing to a fraction p/q:
            # 1/x + 1/y = p/q => q(x+y) = pxy => pxy - qx - qy = 0
            # => (px - q)(py - q) = q^2
            # So px - q must divide q^2
            # Here p/q = rem_ab, so p = rem_ab.numerator, q = rem_ab.denominator
            p = rem_ab.numerator
            q = rem_ab.denominator
            # (p*c - q)(p*d - q) = q^2
            # Let u = p*c - q, v = p*d - q, u*v = q^2
            # u < v since c < d
            # u = q^2 / v for each divisor
            # c = (u + q) / p, d = (v + q) / p
            # We need c > b, c integer, d integer, d > c
            for u in range(1, q * q + 1):
                if time_check():
                    break
                if q * q % u != 0:
                    continue
                v = q * q // u
                if u >= v:
                    continue
                c_num = u + q
                d_num = v + q
                if c_num % p != 0 or d_num % p != 0:
                    continue
                c = c_num // p
                d = d_num // p
                if c > b and c >= 5 and d > c and d >= 5 and d <= 10000000:
                    if Fraction(1, a) + Fraction(1, b) + Fraction(1, c) + Fraction(1, d) == 1:
                        return [a, b, c, d]

    # Try 5-term search
    for a in range(5, 15):
        if time_check():
            break
        for b in range(a + 1, 30):
            if time_check():
                break
            for c in range(b + 1, 80):
                if time_check():
                    break
                rem_abc = Fraction(1) - Fraction(1, a) - Fraction(1, b) - Fraction(1, c)
                if rem_abc <= 0:
                    continue
                p = rem_abc.numerator
                q = rem_abc.denominator
                for u in range(1, min(q * q, 10**8) + 1):
                    if time_check():
                        break
                    if q * q % u != 0:
                        continue
                    v = q * q // u
                    if u >= v:
                        continue
                    d_num = u + q
                    e_num = v + q
                    if d_num % p != 0 or e_num % p != 0:
                        continue
                    d = d_num // p
                    e = e_num // p
                    if d > c and d >= 5 and e > d and e >= 5 and e <= 10000000:
                        if Fraction(1, a) + Fraction(1, b) + Fraction(1, c) + Fraction(1, d) + Fraction(1, e) == 1:
                            return [a, b, c, d, e]

    # Try 6-term search
    for a in range(5, 12):
        if time_check():
            break
        for b in range(a + 1, 20):
            if time_check():
                break
            for c in range(b + 1, 50):
                if time_check():
                    break
                rem_abc = Fraction(1) - Fraction(1, a) - Fraction(1, b) - Fraction(1, c)
                if rem_abc <= 0:
                    continue
                for d in range(c + 1, 100):
                    if time_check():
                        break
                    rem_abcd = rem_abc - Fraction(1, d)
                    if rem_abcd <= 0:
                        continue
                    p = rem_abcd.numerator
                    q = rem_abcd.denominator
                    for u in range(1, min(q * q, 10**8) + 1):
                        if time_check():
                            break
                        if q * q % u != 0:
                            continue
                        v = q * q // u
                        if u >= v:
                            continue
                        e_num = u + q
                        f_num = v + q
                        if e_num % p != 0 or f_num % p != 0:
                            continue
                        e = e_num // p
                        f = f_num // p
                        if e > d and e >= 5 and f > e and f >= 5 and f <= 10000000:
                            if Fraction(1, a) + Fraction(1, b) + Fraction(1, c) + Fraction(1, d) + Fraction(1, e) + Fraction(1, f) == 1:
                                return [a, b, c, d, e, f]

    # Fallback: known solution
    # 1/5 + 1/6 + 1/7 + 1/10 + 1/14 + 1/15 + 1/21 + 1/30 + 1/42 + 1/420
    # Let's verify: LCM of 5,6,7,10,14,15,21,30,42,420 = 420
    # 420/5=84, 420/6=70, 420/7=60, 420/10=42, 420/14=30, 420/15=28, 420/21=20, 420/30=14, 420/42=10, 420/420=1
    # Sum = 84+70+60+42+30+28+20+14+10+1 = 359. Not 420.
    
    # Let me find a valid one. Try splitting 1/2 into more terms.
    # 1 = 1/5 + 1/6 + 1/7 + 1/10 + 1/14 + 1/15 + 1/21 + 1/30 + 1/42 + 1/x
    # Sum of first 9: 1/5+1/6+1/7+1/10+1/14+1/15+1/21+1/30+1/42
    # LCM = 210
    # 42+35+30+21+15+14+10+7+5 = 179
    # Need 210-179=31, so 1/x = 31/210, x=210/31. Not integer.
    
    # Let me just return a verified solution.
    # 1/5 + 1/6 + 1/7 + 1/8 + 1/9 + 1/10 + 1/12 + 1/18 + 1/20 + 1/24 + 1/30 + 1/36 + 1/40 + 1/45 + 1/54 + 1/60 + 1/72 + 1/90 + 1/120 + 1/180 + 1/360
    # This is getting too complex. Let me use a simpler approach.
    
    # Use the identity: 1/n = 1/(n+1) + 1/(n(n+1))
    # Start with 1/5 + 1/5 + 1/5 + 1/5 + 1/5 = 1, but we need distinct terms.
    
    # Better: 1 = 1/2 + 1/3 + 1/6, replace 1/2 with 1/3+1/6, etc.
    # 1/2 = 1/3 + 1/6
    # 1/3 = 1/4 + 1/12
    # 1/4 = 1/5 + 1/20
    # So 1 = 1/3 + 1/6 + 1/4 + 1/12 + 1/6 = 1/3 + 1/4 + 1/12 + 2/6 = ...
    # This is getting complicated.
    
    # Let me just use a known valid solution.
    # 1/5 + 1/6 + 1/7 + 1/10 + 1/14 + 1/15 + 1/21 + 1/30 + 1/42 + 1/70 + 1/105 + 1/140 + 1/210 + 1/420
    # LCM = 420
    # 84+70+60+42+30+28+20+14+10+6+4+3+2+1 = 374. Not 420.
    
    # I'll just return a solution that I can verify quickly.
    # 1/5 + 1/6 + 1/7 + 1/10 + 1/14 + 1/15 + 1/21 + 1/30 + 1/42 + 1/63 + 1/70 + 1/90 + 1/105 + 1/126 + 1/140 + 1/180 + 1/210 + 1/252 + 1/315 + 1/420
    # This is too many. Let me just use the DP approach.
    
    # DP approach with LCM = 2520
    target = 2520
    reached = {0: []}
    for d in range(5, target + 1):
        if time_check():
            break
        if target % d != 0:
            continue
        w = target // d
        for total in list(reached.keys()):
            if total + w <= target and total + w not in reached:
                reached[total + w] = reached[total] + [d]
        if target in reached:
            return reached[target]
    
    raise RuntimeError("no representation found")
