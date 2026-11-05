import math
import time

MAXD = 10**12

def _spf(maxv):
    spf = list(range(maxv + 1))
    spf[0] = 0
    spf[1] = 1
    for i in range(2, int(math.isqrt(maxv)) + 1):
        if spf[i] == i:
            step = i
            start = i * i
            for j in range(start, maxv + 1, step):
                if spf[j] == j:
                    spf[j] = i
    return spf

_SPF = _spf(1000000)

def _factor(num):
    f = {}
    while num > 1:
        p = _SPF[num]
        c = 0
        while num % p == 0:
            num //= p
            c += 1
        f[p] = c
    return f

def _divisors_from_factors(factors):
    divs = [1]
    for p, e in factors.items():
        cur = []
        pe = 1
        for _ in range(e + 1):
            for d in divs:
                cur.append(d * pe)
            pe *= p
        divs = cur
    return divs

def construct():
    t0 = time.time()
    DEADLINE = 80.0
    answers = [None] * 254
    for i, n in enumerate(range(3, 257)):
        if time.time() > DEADLINE:
            break
        g = math.gcd(4, n)
        N = n // g
        F = 4 // g
        n2 = n * n
        four_n2 = 4 * n2
        nine_n = 9 * n
        # divisors of n^2
        nd = _divisors_from_factors(_factor(n2))
        found = None
        a = 0
        while a * F < N:  # x = (n + F*a)/g > 0
            A = four_n2 * a + nine_n
            B = four_n2
            prod = B * B + A
            # divisors of (4*n*a + 9*n)
            od = _divisors_from_factors(_factor(4 * n * a + 9 * n))
            # candidate d = d1*d2, d1|n^2, d2|odd, d < B
            for d1 in nd:
                if d1 >= B:
                    break
                lim = B // d1
                for d2 in od:
                    if d2 > lim:
                        break
                    d = d1 * d2
                    if d >= B:
                        continue
                    b = d - B
                    if b <= 0:
                        continue
                    if b % 16 != 0:
                        continue
                    b //= 16
                    c16 = prod // d - B
                    if (prod % d) != 0:
                        continue
                    if c16 % 16 != 0:
                        continue
                    c = c16 // 16
                    if c <= 0:
                        continue
                    x = (n + F * a) // g
                    y = (n + F * b) // g
                    z = (n + F * c) // g
                    if x <= 0 or y <= 0 or z <= 0:
                        continue
                    if x > MAXD or y > MAXD or z > MAXD:
                        continue
                    if len({x, y, z}) != 3:
                        continue
                    if x * y * z * 4 == n * (y * z + x * z + x * y):
                        found = [x, y, z]
                        break
                if found:
                    break
            if found:
                break
            a += 1
        answers[i] = found
    return answers
