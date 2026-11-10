import time
import numpy as np
from math import isqrt

def construct():
    MAX_DENOM = 10**12
    DEADLINE = time.time() + 85.0
    answers = [None] * 254

    def factorize(num):
        """Return dict of prime->exponent for num."""
        if num == 0:
            return {0: 1}
        factors = {}
        d = 2
        temp = num
        while d * d <= temp:
            while temp % d == 0:
                factors[d] = factors.get(d, 0) + 1
                temp //= d
            d += 1
        if temp > 1:
            factors[temp] = factors.get(temp, 1)
        return factors

    def get_divisors_from_factors(factors):
        """Generate all divisors from a prime factorization dict."""
        divisors = [1]
        for prime, exp in factors.items():
            new_divs = []
            for dd in divisors:
                pwr = 1
                for e in range(exp + 1):
                    new_divs.append(dd * pwr)
                    pwr *= prime
            divisors = new_divs
        return divisors

    def merge_factors(f1, f2):
        """Merge two prime factorization dicts."""
        merged = dict(f1)
        for p, e in f2.items():
            merged[p] = merged.get(p, 0) + e
        return merged

    # Precompute factorizations for all n in 3..256
    n_factors = {}
    n2_factors = {}
    for n in range(3, 257):
        n_factors[n] = factorize(n)
        n2_factors[n] = {p: 2*e for p, e in n_factors[n].items()}

    for n in range(3, 257):
        idx = n - 3
        if time.time() > DEADLINE:
            answers[idx] = None
            continue

        found = None
        n2f = n2_factors[n]

        # x_min: need 1/x < 4/n, so x > n/4
        x_min = n // 4 + 1

        # Try x values: first a wide range, then extend if needed
        # For 1/x + 1/y + 1/z = 4/n with x<=y<=z, we have 3/x >= 4/n so x <= 3n/4
        # But we also need x > n/4. So x in (n/4, 3n/4]
        # Let's try x from x_min up to 3*n//4 + 10
        x_max = 3 * n // 4 + 10

        for x in range(x_min, x_max + 1):
            if time.time() > DEADLINE:
                break

            a = 4 * x - n
            if a <= 0:
                continue
            b = n * x

            # We need (ay-b)(az-b) = b^2
            # Let d = ay - b, then d | b^2 and d > 0 (since y > b/a for z > 0)
            # y = (d+b)/a, z = (b^2/d + b)/a
            # Need (d+b) % a == 0 and (b^2/d + b) % a == 0

            # Factor b^2 = n^2 * x^2
            x2_factors = {p: 2*e for p, e in factorize(x).items()}
            b2_factors = merge_factors(n2f, x2_factors)
            b_sq = b * b

            divs_b2 = get_divisors_from_factors(b2_factors)

            # Filter divisors d where (d + b) % a == 0
            # Vectorize this check
            d_arr = np.array(divs_b2, dtype=np.int64)
            mask = (d_arr + b) % a == 0
            valid_d = d_arr[mask]

            if len(valid_d) == 0:
                continue

            # For each valid d, check (b_sq // d + b) % a == 0
            # b_sq // d must be integer, which it is since d | b_sq
            rem = b_sq // valid_d
            z_mask = (rem + b) % a == 0
            valid_d2 = valid_d[z_mask]
            rem2 = rem[z_mask]

            if len(valid_d2) == 0:
                continue

            y_vals = (valid_d2 + b) // a
            z_vals = (rem2 + b) // a

            # Filter: distinct, positive, <= MAX_DENOM
            y_arr = y_vals.astype(np.int64)
            z_arr = z_vals.astype(np.int64)

            mask2 = (y_arr > 0) & (z_arr > 0)
            mask2 &= (y_arr != x) & (z_arr != x) & (y_arr != z_arr)
            mask2 &= (y_arr <= MAX_DENOM) & (z_arr <= MAX_DENOM)

            if not np.any(mask2):
                continue

            y_cand = y_arr[mask2]
            z_cand = z_arr[mask2]

            # Verify equation for first few candidates
            for i in range(min(len(y_cand), 20)):
                yv = int(y_cand[i])
                zv = int(z_cand[i])
                if x * yv * zv * 4 == n * (yv * zv + x * zv + x * yv):
                    found = [x, yv, zv]
                    break

            if found:
                break

        # If not found in primary range, try extended range
        if not found and time.time() < DEADLINE - 5:
            # Try larger x values
            for x in range(x_max + 1, min(2 * n + 100, x_max + 500)):
                if time.time() > DEADLINE:
                    break

                a = 4 * x - n
                if a <= 0:
                    continue
                b = n * x
                b_sq = b * b

                x2_factors = {p: 2*e for p, e in factorize(x).items()}
                b2_factors = merge_factors(n2f, x2_factors)
                divs_b2 = get_divisors_from_factors(b2_factors)

                d_arr = np.array(divs_b2, dtype=np.int64)
                mask = (d_arr + b) % a == 0
                valid_d = d_arr[mask]

                if len(valid_d) == 0:
                    continue

                rem = b_sq // valid_d
                z_mask = (rem + b) % a == 0
                valid_d2 = valid_d[z_mask]
                rem2 = rem[z_mask]

                if len(valid_d2) == 0:
                    continue

                y_vals = (valid_d2 + b) // a
                z_vals = (rem2 + b) // a

                y_arr = y_vals.astype(np.int64)
                z_arr = z_vals.astype(np.int64)

                mask2 = (y_arr > 0) & (z_arr > 0)
                mask2 &= (y_arr != x) & (z_arr != x) & (y_arr != z_arr)
                mask2 &= (y_arr <= MAX_DENOM) & (z_arr <= MAX_DENOM)

                if not np.any(mask2):
                    continue

                y_cand = y_arr[mask2]
                z_cand = z_arr[mask2]

                for i in range(min(len(y_cand), 20)):
                    yv = int(y_cand[i])
                    zv = int(z_cand[i])
                    if x * yv * zv * 4 == n * (yv * zv + x * zv + x * yv):
                        found = [x, yv, zv]
                        break

                if found:
                    break

        answers[idx] = found

    return answers
