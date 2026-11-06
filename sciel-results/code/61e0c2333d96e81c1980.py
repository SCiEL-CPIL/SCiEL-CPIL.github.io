import time
from fractions import Fraction
from math import gcd

def construct():
    deadline = time.time() + 85  # Safety margin within 90s

    # Strategy 1: Fast DP baseline to get a valid solution quickly
    def dp_baseline():
        target = 2520
        # Use a list-based DP for speed: dp[s] = list of denominators to reach sum s
        dp = [None] * (target + 1)
        dp[0] = []
        for d in range(5, target + 1):
            if target % d:
                continue
            w = target // d
            # Iterate in reverse to avoid using d multiple times
            for s in range(target - w, -1, -1):
                if dp[s] is not None and dp[s + w] is None:
                    dp[s + w] = dp[s] + [d]
            if dp[target] is not None:
                sol = dp[target]
                if all(5 <= x <= 10000000 for x in sol) and len(set(sol)) == len(sol):
                    return sol
        return None

    # Strategy 2: Targeted search for small n using integer arithmetic
    def search_n4():
        """Find 4 distinct denominators >= 5 summing to 1."""
        # 1/a + 1/b + 1/c + 1/d = 1
        # For small a, b: r = a*b - a - b, s = a*b
        # 1/c + 1/d = r/s
        # c must satisfy: c > s/r (since 1/c < r/s) and c < 2s/r (since 1/c > r/(2s))
        # Also c > b, d > c
        for a in range(5, 20):
            if time.time() > deadline:
                return None
            for b in range(a + 1, 50):
                if time.time() > deadline:
                    return None
                r = a * b - a - b
                if r <= 0:
                    continue
                s = a * b
                c_min = max(b + 1, int(s / r) + 1)
                c_max = min(10000000, int(2 * s / r) + 2)
                if c_min > c_max:
                    continue
                for c in range(c_min, c_max + 1):
                    if time.time() > deadline:
                        return None
                    num = r * c - s
                    den = s * c
                    if num <= 0:
                        continue
                    if den % num != 0:
                        continue
                    d = den // num
                    if d <= c or d < 5 or d > 10000000:
                        continue
                    if Fraction(1, a) + Fraction(1, b) + Fraction(1, c) + Fraction(1, d) == 1:
                        return [a, b, c, d]
        return None

    def search_n5():
        """Find 5 distinct denominators >= 5 summing to 1."""
        for a in range(5, 15):
            if time.time() > deadline:
                return None
            for b in range(a + 1, 25):
                if time.time() > deadline:
                    return None
                for c in range(b + 1, 40):
                    if time.time() > deadline:
                        return None
                    rem = 1 - Fraction(1, a) - Fraction(1, b) - Fraction(1, c)
                    if rem <= 0:
                        continue
                    d_min = max(c + 1, int(1 / rem) + 1)
                    d_max = min(10000000, int(2 / rem) + 2)
                    if d_min > d_max:
                        continue
                    for d in range(d_min, d_max + 1):
                        if time.time() > deadline:
                            return None
                        rem2 = rem - Fraction(1, d)
                        if rem2 <= 0:
                            continue
                        if rem2.denominator % rem2.numerator != 0:
                            continue
                        e = rem2.denominator // rem2.numerator
                        if e <= d or e < 5 or e > 10000000:
                            continue
                        if Fraction(1, a) + Fraction(1, b) + Fraction(1, c) + Fraction(1, d) + Fraction(1, e) == 1:
                            return [a, b, c, d, e]
        return None

    # Try smallest n first (n=4 is impossible for denominators >=5, but check anyway)
    for n, search_fn in [(4, search_n4), (5, search_n5)]:
        result = search_fn()
        if result is not None:
            return result

    # Fallback to DP
    result = dp_baseline()
    if result is not None:
        return result

    # Last resort: hardcoded known valid solution
    # Verify: 1/5 + 1/6 + 1/7 + 1/10 + 1/14 + 1/35
    # = 42/210 + 35/210 + 30/210 + 21/210 + 15/210 + 6/210 = 149/210 != 1
    # Use a verified solution: 1 = 1/5 + 1/6 + 1/7 + 1/42 + 1/14 + 1/35
    # Let's just rely on DP or raise if it fails
    raise RuntimeError("no valid representation found")
