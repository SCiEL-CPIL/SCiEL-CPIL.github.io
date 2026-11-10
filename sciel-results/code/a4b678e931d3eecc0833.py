"""Rewritten program: divisor-enumeration approach for 4/n unit fractions."""
import time

# EVOLVE-BLOCK-START
def construct():
    MAX_DENOM = 10**12
    DEADLINE = 80.0  # safety margin before 90s kill
    t0 = time.time()
    answers = []

    def factorize(num):
        """Trial-division factorization."""
        factors = {}
        d = 2
        while d * d <= num:
            while num % d == 0:
                factors[d] = factors.get(d, 0) + 1
                num //= d
            d += 1
        if num > 1:
            factors[num] = factors.get(num, 1)
        return factors

    def divisors_from_factors(factors):
        """Generate all divisors from prime factorization."""
        divs = [1]
        for p, e in factors.items():
            new_divs = []
            pe = 1
            for i in range(e + 1):
                for d in divs:
                    new_divs.append(d * pe)
                pe *= p
            divs = new_divs
        return divs

    def solve_n(n):
        """Find distinct x,y,z with 1/x+1/y+1/z = 4/n, all <= MAX_DENOM."""
        # Try increasing x starting from smallest possible
        # x must satisfy 1/x < 4/n => x > n/4, so x >= n//4 + 1
        # Also need 4x - n > 0 for a > 0
        for x in range(n // 4 + 1, n // 4 + 200):  # bounded x search
            if time.time() - t0 > DEADLINE:
                return None
            a = 4 * x - n
            if a <= 0:
                continue
            b = n * x
            # (a*y - b)(a*z - b) = b^2
            # Enumerate divisors d of b^2
            # y = (d + b) / a, z = (b^2//d + b) / a
            # Need d + b divisible by a, and b^2//d + b divisible by a
            # Also y, z > 0, y != z, y != x, z != x, y,z <= MAX_DENOM
            
            # Factor b^2 = (n*x)^2 = n^2 * x^2
            # Factor n^2 and x^2 separately then combine
            fn = factorize(n)
            fx = factorize(x)
            # b^2 = n^2 * x^2
            factors_b2 = {}
            for p, e in fn.items():
                factors_b2[p] = factors_b2.get(p, 0) + 2 * e
            for p, e in fx.items():
                factors_b2[p] = factors_b2.get(p, 0) + 2 * e
            
            divs = divisors_from_factors(factors_b2)
            
            for d in divs:
                if time.time() - t0 > DEADLINE:
                    return None
                # y = (d + b) / a
                num_y = d + b
                if num_y % a != 0:
                    continue
                y = num_y // a
                if y <= 0 or y > MAX_DENOM:
                    continue
                # z = (b^2 // d + b) / a
                if b * b % d != 0:
                    continue
                num_z = b * b // d + b
                if num_z % a != 0:
                    continue
                z = num_z // a
                if z <= 0 or z > MAX_DENOM:
                    continue
                # Check distinctness
                if y == x or z == x or y == z:
                    continue
                # All positive, distinct, within bound
                return [x, y, z]
        return None

    for n in range(3, 257):
        if time.time() - t0 > DEADLINE - 1.0:
            # Not enough time for remaining cases; fill with None
            answers.append(None)
            continue
        found = solve_n(n)
        answers.append(found)
        # If we still have plenty of time, continue; otherwise pad
        remaining = 256 - n
        if time.time() - t0 > DEADLINE - remaining * 0.05:
            # Tight; just keep going but check each iteration
            pass

    return answers
# EVOLVE-BLOCK-END
