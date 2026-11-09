import time
from fractions import Fraction

def construct():
    deadline = time.time() + 50

    def search_k(k, min_den=5, time_limit=10):
        target = Fraction(1, 1)
        found = [None]
        local_deadline = time.time() + time_limit
        
        def dfs(start, remaining, chosen):
            if found[0] is not None or time.time() > local_deadline:
                return
            terms_left = k - len(chosen)
            if terms_left == 0:
                if remaining == 0:
                    found[0] = chosen[:]
                return
            if terms_left == 1:
                if remaining > 0:
                    num, den = remaining.numerator, remaining.denominator
                    if den % num == 0:
                        d = den // num
                        if d > start and d >= 5 and d <= 10000000:
                            found[0] = chosen + [d]
                return

            d_min = max(start + 1, (1 / remaining).numerator // (1 / remaining).denominator + 1)
            d_max = int(terms_left / remaining)
            d_max = min(d_max, 10000000)
            
            if d_min > d_max:
                return

            for d in range(d_min, d_max + 1):
                if found[0] is not None or time.time() > local_deadline:
                    return
                new_remaining = remaining - Fraction(1, d)
                if new_remaining < 0:
                    break
                dfs(d, new_remaining, chosen + [d])

        dfs(min_den - 1, target, [])
        return found[0]

    # Try k=6
    res6 = search_k(6, time_limit=15)
    if res6:
        s = Fraction(0, 1)
        for d in res6:
            s += Fraction(1, d)
        if s == 1 and len(set(res6)) == len(res6) and all(5 <= d <= 10000000 for d in res6):
            return res6

    # Try k=7
    res7 = search_k(7, time_limit=15)
    if res7:
        s = Fraction(0, 1)
        for d in res7:
            s += Fraction(1, d)
        if s == 1 and len(set(res7)) == len(res7) and all(5 <= d <= 10000000 for d in res7):
            return res7

    # Fallback
    return [5, 6, 7, 10, 14, 21, 35]
