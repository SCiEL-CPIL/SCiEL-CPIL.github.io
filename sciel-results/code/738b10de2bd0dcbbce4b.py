import time
import math

def construct():
    start = time.time()
    deadline = start + 85.0
    LIMIT = 10**12
    answers = [None] * 254

    # Precompute primes up to 50000 for factorization fallback
    primes = []
    for i in range(2, 50001):
        is_prime = True
        for p in primes:
            if p * p > i:
                break
            if i % p == 0:
                is_prime = False
                break
        if is_prime:
            primes.append(i)

    def factorize(num):
        factors = {}
        temp = num
        for p in primes:
            if p * p > temp:
                break
            while temp % p == 0:
                factors[p] = factors.get(p, 0) + 1
                temp //= p
        if temp > 1:
            factors[temp] = factors.get(temp, 0) + 1
        return factors

    def get_divisors_of_square(b):
        """Get all divisors of b^2 efficiently"""
        factors = factorize(b)
        divisors = [1]
        for p, e in factors.items():
            max_exp = 2 * e
            p_powers = [1]
            for k in range(1, max_exp + 1):
                p_powers.append(p_powers[-1] * p)
            new_divs = []
            for d in divisors:
                for pp in p_powers:
                    new_divs.append(d * pp)
            divisors = new_divs
        return divisors

    def find_solution_direct(n):
        """Direct y-search for a given n"""
        x_min = n // 4 + 1
        x_max = (3 * n) // 4

        if x_max < x_min:
            return None

        # Build a priority list of x values: try small a = 4x-n first
        # a = 4x - n, so small x gives small a
        # Also try heuristic x values
        heuristic_xs = set()
        for frac_num, frac_den in [(1,2), (2,3), (3,4), (1,3), (2,5), (3,5), (1,4), (3,8), (5,8), (4,5), (5,6), (7,8)]:
            x_val = n * frac_num // frac_den
            if x_min <= x_val <= x_max:
                heuristic_xs.add(x_val)
            x_val2 = n * frac_num // frac_den + 1
            if x_min <= x_val2 <= x_max:
                heuristic_xs.add(x_val2)
            x_val3 = n * frac_num // frac_den - 1
            if x_min <= x_val3 <= x_max:
                heuristic_xs.add(x_val3)

        # Sort: heuristic first, then by a = 4x-n ascending
        all_xs = set(range(x_min, x_max + 1))
        sorted_xs = sorted(all_xs, key=lambda x: (x not in heuristic_xs, 4*x - n))

        for x in sorted_xs:
            if time.time() > deadline:
                return None

            a = 4 * x - n
            if a <= 0:
                continue
            b = n * x

            # y must satisfy: 1/y < a/b => y > b/a
            # and for z > y: 1/z < 1/y, so (a*y - b) > 0 and z = b*y/(a*y-b) > y
            # z > y => b*y > y*(a*y - b) => b > a*y - b => 2b > a*y => y < 2b/a
            y_min = b // a + 1
            y_max = (2 * b) // a

            if y_max < y_min:
                continue

            # Direct y search
            for y in range(y_min, y_max + 1):
                if y == x:
                    continue

                denom = a * y - b
                if denom <= 0:
                    continue

                num = b * y
                if num % denom != 0:
                    continue

                z = num // denom

                if z <= 0 or z > LIMIT:
                    continue
                if z == x or z == y:
                    continue
                if z <= y:
                    continue

                # Verify with cross multiplication
                if n * (y * z + x * z + x * y) == 4 * x * y * z:
                    return [x, y, z]

        return None

    def find_solution_divisor(n):
        """Divisor enumeration fallback for harder n values"""
        x_min = n // 4 + 1
        x_max = (3 * n) // 4

        if x_max < x_min:
            return None

        for x in range(x_min, x_max + 1):
            if time.time() > deadline:
                return None

            a = 4 * x - n
            if a <= 0:
                continue
            b = n * x
            b2 = b * b

            # Enumerate divisors of b^2
            divs = get_divisors_of_square(b)

            for u in divs:
                if u >= b:
                    continue
                if (u + b) % a != 0:
                    continue
                v = b2 // u
                if (v + b) % a != 0:
                    continue

                y = (u + b) // a
                z = (v + b) // a

                if y <= 0 or z <= 0:
                    continue
                if y > LIMIT or z > LIMIT:
                    continue
                if x == y or x == z or y == z:
                    continue

                if n * (y * z + x * z + x * y) == 4 * x * y * z:
                    return [x, y, z]

        return None

    # First pass: try direct search for all n
    for idx, n in enumerate(range(3, 257)):
        if time.time() > deadline:
            break
        answers[idx] = find_solution_direct(n)

    # Second pass: try divisor enumeration for unsolved n
    solved = sum(1 for a in answers if a is not None)
    if solved < 254:
        remaining = [n for idx, n in enumerate(range(3, 257)) if answers[idx] is None]
        for n in remaining:
            if time.time() > deadline - 2:
                break
            answers[n - 3] = find_solution_divisor(n)

    return answers
