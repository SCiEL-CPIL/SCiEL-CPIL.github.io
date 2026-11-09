import time
from math import gcd

def construct():
    start_time = time.time()
    deadline = start_time + 80.0  # Safety margin before 90s kill

    def is_valid_solution(denoms):
        """Verify denominators are distinct, in [5, 10^7], and sum of reciprocals = 1."""
        if len(denoms) < 1 or len(denoms) > 256:
            return False
        if len(set(denoms)) != len(denoms):
            return False
        for d in denoms:
            if d < 5 or d > 10000000:
                return False
        # Check sum of reciprocals = 1 using exact arithmetic
        num, den = 0, 1
        for d in denoms:
            g = gcd(den, d)
            den //= g
            d2 = d // g
            num = num * d2 + den
            den = den * d2
        return num == den

    def dfs(numer, denom, min_d, remaining, current, deadline):
        """
        Find 'remaining' distinct denominators >= min_d such that
        sum(1/d_i) = numer/denom.
        Uses exact integer arithmetic.
        """
        if time.time() > deadline:
            return None

        if remaining == 0:
            if numer == 0:
                return current
            return None

        if remaining == 1:
            # Need 1/d = numer/denom => d = denom/numer
            if numer == 0:
                return None
            if denom % numer == 0:
                d = denom // numer
                if d >= min_d and d >= 5 and d <= 10000000:
                    return current + [d]
            return None

        if remaining == 2:
            # Need 1/x + 1/y = numer/denom, x < y, x >= min_d
            # numer/denom = (x+y)/(xy)
            # numer*xy = denom*(x+y)
            # numer*x*y - denom*x - denom*y = 0
            # (numer*x - denom)*(numer*y - denom) = denom^2
            b2 = denom * denom
            # Find factor pairs of b2
            # numer*x - denom = a, numer*y - denom = b, a*b = b2
            # x = (a + denom) / numer, y = (b + denom) / numer
            # Need x, y integers, x >= min_d, x < y, both in [5, 10^7]
            i = 1
            while i * i <= b2:
                if time.time() > deadline:
                    return None
                if b2 % i == 0:
                    a, b = i, b2 // i
                    # x = (a + denom) / numer, y = (b + denom) / numer
                    if (a + denom) % numer == 0 and (b + denom) % numer == 0:
                        x = (a + denom) // numer
                        y = (b + denom) // numer
                        if x >= min_d and y > x and x >= 5 and y <= 10000000:
                            return current + [x, y]
                        if y >= min_d and x > y and y >= 5 and x <= 10000000:
                            return current + [y, x]
                i += 1
            return None

        # Pruning: check if max possible sum with 'remaining' terms starting at min_d is >= numer/denom
        # Max sum = 1/min_d + 1/(min_d+1) + ... + 1/(min_d+remaining-1)
        # Quick check using floating point
        max_sum = 0.0
        for i in range(remaining):
            max_sum += 1.0 / (min_d + i)
        if max_sum < numer / denom:
            return None

        # Lower bound on d: 1/d <= numer/denom => d >= denom/numer
        d_min = max(min_d, (denom + numer - 1) // numer)  # ceil(denom/numer)
        # Also d must be at least min_d

        # Upper bound on d: we need remaining-1 more terms after d, each at most 1/(d+1), 1/(d+2), ...
        # So numer/denom - 1/d <= 1/(d+1) + 1/(d+2) + ... + 1/(d+remaining-1)
        # This gives an upper bound on d, but it's complex. Use a simpler bound.
        # The smallest remaining sum is achieved with the largest denominators.
        # For a given d, the max sum of remaining terms is 1/(d+1) + ... + 1/(d+remaining-1)
        # We need numer/denom - 1/d <= that max sum
        # If numer/denom - 1/d < 0, skip (1/d > numer/denom)
        # If numer/denom - 1/d > max remaining sum, then d is too small... wait no,
        # if 1/d is too small, we can't reach the target. So we need 1/d <= numer/denom.
        # And numer/denom - 1/d <= sum of 1/(d+1) + ... + 1/(d+remaining-1)

        d_max = 10000000
        # Tighten d_max: for the remaining terms to sum to at least numer/denom - 1/d,
        # we need at minimum 1/(d+1) + ... + 1/(d+remaining-1) >= numer/denom - 1/d
        # The left side decreases as d increases, so there's a max d.
        # Simple bound: d + remaining - 1 < 10^7, so d < 10^7 - remaining + 1
        d_max = min(d_max, 10000000 - remaining + 1)

        for d in range(d_min, d_max + 1):
            if time.time() > deadline:
                return None

            # Check 1/d <= numer/denom
            if d * numer > denom:
                break  # 1/d > numer/denom, too large

            # New remainder: numer/denom - 1/d = (numer*d - denom) / (denom*d)
            new_numer = numer * d - denom
            if new_numer < 0:
                break  # Shouldn't happen if we checked above
            new_denom = denom * d

            # Reduce fraction
            g = gcd(new_numer, new_denom)
            if g > 1:
                new_numer //= g
                new_denom //= g

            # Pruning: check if max sum of (remaining-1) terms starting at d+1 is >= new_numer/new_denom
            if remaining - 1 > 0:
                max_rem = 0.0
                for i in range(remaining - 1):
                    max_rem += 1.0 / (d + 1 + i)
                if max_rem < new_numer / new_denom:
                    continue

            # Pruning: min possible sum of (remaining-1) terms starting at d+1
            # Min sum is with largest denominators: 1/(d+remaining-1) + ... but they must be distinct and >= d+1
            # The minimum sum for (remaining-1) distinct terms >= d+1 is 1/(d+remaining-1) + 1/(d+remaining) + ... 
            # Actually the minimum is when we take the largest possible: but we have no upper bound effectively.
            # Skip this pruning for now.

            res = dfs(new_numer, new_denom, d + 1, remaining - 1, current + [d], deadline)
            if res is not None:
                return res

        return None

    # Try 5 terms first
    deadline_5 = start_time + 20.0
    res = dfs(1, 1, 5, 5, [], deadline_5)
    if res is not None and is_valid_solution(res):
        return res

    # Try 6 terms
    deadline_6 = start_time + 50.0
    res = dfs(1, 1, 5, 6, [], deadline_6)
    if res is not None and is_valid_solution(res):
        return res

    # Try 7 terms
    deadline_7 = start_time + 75.0
    res = dfs(1, 1, 5, 7, [], deadline_7)
    if res is not None and is_valid_solution(res):
        return res

    # Fallback: subset DP over divisors of a highly composite number
    # Use LCM(5,6,7,8,9,10) = 2520
    target = 2520
    reached = {0: []}
    for d in range(5, target + 1):
        if target % d != 0:
            continue
        w = target // d
        for total, values in list(reached.items()):
            new_total = total + w
            if new_total <= target and new_total not in reached:
                reached[new_total] = values + [d]
        if target in reached:
            return reached[target]

    raise RuntimeError("no solution found")
