import time
from fractions import Fraction
import math

def construct():
    start_time = time.time()
    deadline = start_time + 58

    def max_sum_from(start, count):
        s = Fraction(0)
        for i in range(count):
            s += Fraction(1, start + i)
        return s

    def dfs(current_sum, last_d, k_remaining, path):
        if time.time() > deadline:
            return None

        if k_remaining == 0:
            if current_sum == 1:
                return path
            return None

        if current_sum >= 1:
            return None

        # Prune: max possible sum of remaining terms
        max_possible = max_sum_from(last_d + 1, k_remaining)
        if current_sum + max_possible < 1:
            return None

        # Determine lower bound for next denominator
        lb = last_d + 1
        rem = 1 - current_sum
        if rem > 0:
            min_d_frac = Fraction(1, rem)
            # Ceiling of min_d_frac
            num, den = min_d_frac.numerator, min_d_frac.denominator
            ceil_val = -(-num // den)
            if ceil_val > lb:
                lb = ceil_val

        path = list(path)
        d = lb

        while True:
            if time.time() > deadline:
                return None

            if d > 10000000:
                break

            if k_remaining == 1:
                rem = 1 - current_sum
                if rem > 0:
                    d_val = Fraction(1, rem)
                    if d_val.denominator == 1:
                        d_int = d_val.numerator
                        if d_int > last_d and d_int <= 10000000:
                            return path + [d_int]
                break

            # Upper bound check using floats for speed
            # We need 1/d + max_sum_from(d+1, k_remaining-1) >= rem
            # If float check fails, we break.
            # max_sum_from is expensive, so we approximate or calculate once?
            # Since d changes, max_sum_from changes.
            # But for large d, it's small.
            
            # Optimization: if d is large, 1/d is small.
            # Let's just calculate the float bound.
            # max_sum_approx = sum(1/(d+i) for i in range(k_remaining-1))
            max_next_float = 0.0
            for i in range(k_remaining - 1):
                max_next_float += 1.0 / (d + i)
                
            if (1.0 / d + max_next_float) < float(rem):
                break

            next_sum = current_sum + Fraction(1, d)
            res = dfs(next_sum, d, k_remaining - 1, path + [d])
            if res:
                return res

            d += 1

        return None

    # Start with k=8
    res = dfs(Fraction(0), 4, 8, [])
    if res:
        return res

    # If k=8 fails (e.g. timeout or no solution), try k=9
    res = dfs(Fraction(0), 4, 9, [])
    if res:
        return res

    # If still no solution, raise error
    raise RuntimeError("No solution found")
