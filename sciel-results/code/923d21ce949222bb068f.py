import time
import math

def construct():
    # Precompute solutions for n=3..256
    # We do this once. If the function is called multiple times, cache it.
    if not hasattr(construct, '_cache'):
        construct._cache = {}
        deadline = time.time() + 85.0  # Safety margin before 90s kill

        for n in range(3, 257):
            if time.time() > deadline:
                break

            sol = None
            # x > n/4 and x <= 3n/4
            x_min = n // 4 + 1
            x_max = 3 * n // 4

            for x in range(x_min, x_max + 1):
                if sol:
                    break
                if time.time() > deadline:
                    break

                A = 4 * x - n
                if A <= 0:
                    continue
                B = n * x

                # We need u * v = B^2
                # u = A*y - B
                # v = A*z - B
                # u < v => u < B
                # u > A*x - B
                L = A * x - B + 1
                if L < 1:
                    L = 1
                R = B - 1

                if L > R:
                    continue

                # Find divisors of B^2 in [L, R]
                # such that u % A == (-B) % A

                # Factorize B
                def factorize(num):
                    factors = {}
                    d = 2
                    while d * d <= num:
                        while num % d == 0:
                            factors[d] = factors.get(d, 0) + 1
                            num //= d
                        d += 1
                    if num > 1:
                        factors[num] = factors.get(num, 0) + 1
                    return factors

                # Get prime factors of B
                b_factors = factorize(B)

                # Generate divisors of B^2
                # If B = p1^e1 * p2^e2 ..., then B^2 = p1^(2e1) * ...
                # We can generate all divisors.
                # Number of divisors of B^2 is product (2*ei + 1).
                # Max B ~ 50000.
                # Worst case B is highly composite.
                # 50000 = 5 * 10^4 = 5 * (2*5)^4 = 2^4 * 5^5.
                # B^2 = 2^8 * 5^10.
                # Divisors: (8+1)*(10+1) = 99. Very small.

                divisors = [1]
                for p, exp in b_factors.items():
                    new_divs = []
                    pow_p = 1
                    for _ in range(2 * exp):  # Exponent in B^2 is 2*exp
                        for d in divisors:
                            new_divs.append(d * pow_p)
                        pow_p *= p
                    divisors = new_divs

                # Sort divisors to check in order? 
                # We just need one valid u.
                target_mod = (-B) % A

                for u in divisors:
                    if L <= u <= R:
                        if u % A == target_mod:
                            v = (B * B) // u
                            # Check distinctness
                            y = (u + B) // A
                            z = (v + B) // A
                            # Ensure y, z are integers (guaranteed by mod check)
                            # Ensure distinctness x, y, z
                            if x != y and y != z and x != z:
                                # Ensure positive
                                if y > 0 and z > 0:
                                    # Ensure <= 10^12
                                    if y <= 10**12 and z <= 10**12:
                                        sol = (x, y, z)
                                        break
            construct._cache[n] = sol

    # Retrieve results
    results = []
    for n in range(3, 257):
        results.append(construct._cache.get(n, None))

    return results
