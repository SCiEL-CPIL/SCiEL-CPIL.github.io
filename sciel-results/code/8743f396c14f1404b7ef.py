import time
import math

def construct():
    deadline = time.time() + 85.0
    answers = [None] * 254
    
    _factor_cache = {}
    _divs_cache = {}
    
    def factorize(num):
        if num in _factor_cache:
            return _factor_cache[num]
        orig = num
        factors = {}
        d = 2
        while d * d <= num:
            while num % d == 0:
                factors[d] = factors.get(d, 0) + 1
                num //= d
            d += 1
        if num > 1:
            factors[num] = factors.get(num, 0) + 1
        _factor_cache[orig] = factors
        return factors
    
    def divisors_of_square(b):
        if b in _divs_cache:
            return _divs_cache[b]
        if b <= 0:
            _divs_cache[b] = [1]
            return [1]
        factors = factorize(b)
        divs = [1]
        for p, e in factors.items():
            exp = 2 * e
            new_divs = []
            for d in divs:
                val = d
                for _ in range(exp + 1):
                    new_divs.append(val)
                    val *= p
            divs = new_divs
        _divs_cache[b] = divs
        return divs
    
    def try_parameterization(n, k):
        """Try the parametric solution: x=(n+k)/4, y=(n+k)/2, z=4n(n+k)/(k(4k-n))"""
        if k <= 0 or 4 * k == n:
            return None
        num_x = n + k
        if num_x % 4 != 0:
            return None
        x = num_x // 4
        if x <= 0:
            return None
        num_y = n + k
        if num_y % 2 != 0:
            return None
        y = num_y // 2
        if y <= 0:
            return None
        denom_z = k * (4 * k - n)
        if denom_z == 0:
            return None
        num_z = 4 * n * (n + k)
        if num_z % denom_z != 0:
            return None
        z = num_z // denom_z
        if z <= 0:
            return None
        # Check distinctness
        if x == y or y == z or x == z:
            return None
        # Check bounds
        if x > 10**12 or y > 10**12 or z > 10**12:
            return None
        # Verify by cross multiplication
        lhs = n * (y * z + x * z + x * y)
        rhs = 4 * x * y * z
        if lhs == rhs:
            # Sort for consistency
            result = sorted([x, y, z])
            return result
        return None
    
    def solve_n_divisor(n, x, tlimit):
        """Use the divisor-based algebraic approach for a specific x"""
        a = 4 * x - n
        if a <= 0:
            return None
        b = n * x
        divs = divisors_of_square(b)
        neg_b_mod_a = (-b) % a
        for k in divs:
            if time.time() > tlimit:
                break
            if k % a != neg_b_mod_a:
                continue
            y = (k + b) // a
            if y <= x:
                continue
            bq = b * b // k
            if (bq + b) % a != 0:
                continue
            z = (bq + b) // a
            if z <= y:
                continue
            if z > 10**12 or y > 10**12:
                continue
            lhs = n * (y * z + x * z + x * y)
            rhs = 4 * x * y * z
            if lhs == rhs:
                return [x, y, z]
        return None
    
    def solve_n_direct(n, tlimit):
        """Direct search over x and y ranges"""
        x_min = n // 4 + 1
        x_max = (3 * n) // 4
        
        for x in range(x_min, x_max + 1):
            if time.time() > tlimit:
                break
            a = 4 * x - n
            if a <= 0:
                continue
            b = n * x
            y_lo = b // a + 1
            y_hi = 2 * b // a
            if y_hi < y_lo:
                continue
            if y_lo <= x:
                y_lo = x + 1
            if y_hi < y_lo:
                continue
            # Cap the inner loop to avoid getting stuck
            max_y_iter = 5000
            for y in range(y_lo, min(y_lo + max_y_iter, y_hi + 1)):
                if time.time() > tlimit:
                    break
                denom = a * y - b
                if denom <= 0:
                    continue
                if (b * b) % denom != 0:
                    continue
                z = (b * b) // denom
                if (b * b // denom + b) % a != 0:
                    continue
                z = (b * b // denom + b) // a
                if z <= y:
                    continue
                if z > 10**12 or y > 10**12:
                    continue
                lhs = n * (y * z + x * z + x * y)
                rhs = 4 * x * y * z
                if lhs == rhs:
                    return [x, y, z]
        return None
    
    # Pass 0: Try parameterization for all n
    for n in range(3, 257):
        if time.time() > deadline:
            break
        idx = n - 3
        found = None
        # Try k = 1, 2, 4, n/2, n, and divisors of n
        candidates_k = set()
        candidates_k.add(1)
        candidates_k.add(2)
        candidates_k.add(4)
        if n % 2 == 0:
            candidates_k.add(n // 2)
        candidates_k.add(n)
        # Add small divisors of n
        for d in range(1, int(math.isqrt(n)) + 1):
            if n % d == 0:
                candidates_k.add(d)
                if d != n // d:
                    candidates_k.add(n // d)
        for k in sorted(candidates_k):
            if time.time() > deadline:
                break
            result = try_parameterization(n, k)
            if result is not None:
                found = result
                break
        if found is None:
            # Also try k values around n/4
            for k in range(max(1, n // 4 - 5), n // 4 + 6):
                if time.time() > deadline:
                    break
                result = try_parameterization(n, k)
                if result is not None:
                    found = result
                    break
        answers[idx] = found
    
    # Pass 1: Divisor-based search for unsolved
    unsolved = [n for n in range(3, 257) if answers[n - 3] is None]
    if unsolved:
        time_per_n = (deadline - time.time()) / len(unsolved) * 0.5
        for n in unsolved:
            if time.time() > deadline:
                break
            idx = n - 3
            x_min = n // 4 + 1
            x_max = (3 * n) // 4
            # Try a few x values
            x_limit = min(x_min + 15, x_max)
            found = None
            for x in range(x_min, x_limit + 1):
                if time.time() > deadline:
                    break
                result = solve_n_divisor(n, x, min(deadline, time.time() + time_per_n))
                if result is not None:
                    found = result
                    break
            if found is None:
                # Try direct search with limited range
                result = solve_n_direct(n, min(deadline, time.time() + time_per_n))
                if result is not None:
                    found = result
            answers[idx] = found
    
    # Pass 2: Extended search for still-unsolved
    unsolved = [n for n in range(3, 257) if answers[n - 3] is None]
    if unsolved:
        remaining = deadline - time.time()
        if remaining > 0.5:
            time_per_n = remaining / len(unsolved)
            for n in unsolved:
                if time.time() > deadline:
                    break
                idx = n - 3
                x_min = n // 4 + 1
                x_max = (3 * n) // 4
                found = None
                # Try more x values
                x_limit = min(x_min + 50, x_max)
                for x in range(x_min, x_limit + 1):
                    if time.time() > deadline:
                        break
                    result = solve_n_divisor(n, x, min(deadline, time.time() + time_per_n))
                    if result is not None:
                        found = result
                        break
                if found is None:
                    result = solve_n_direct(n, min(deadline, time.time() + time_per_n))
                    if result is not None:
                        found = result
                answers[idx] = found
    
    return answers
