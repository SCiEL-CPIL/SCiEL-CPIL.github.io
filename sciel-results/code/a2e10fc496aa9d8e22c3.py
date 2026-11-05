"""Distinct three-term unit-fraction representations of 4/n."""
import time
import math

def construct():
    deadline = time.time() + 80.0
    answers = []
    BOUND = 10**12

    for n in range(3, 257):
        if time.time() > deadline:
            answers.append(None)
            continue

        found = None
        x_min = n // 4 + 1

        # Try increasing x values
        for x in range(x_min, x_min + 20000):
            if time.time() > deadline:
                break
            a = 4 * x - n
            if a <= 0:
                continue
            b = n * x

            # (ay - b)(az - b) = b^2
            # u = ay - b > 0, v = az - b > 0, u*v = b^2
            # u ≡ -b (mod a), v ≡ -b (mod a)
            # u = a*k - b for integer k, need u > 0 => k >= ceil((b+1)/a)
            # u <= b (since u <= v wlog and u*v=b^2) => a*k - b <= b => k <= floor(2b/a)
            # Also need u | b^2

            k_lo = (b + a) // a  # smallest k with a*k - b > 0, i.e., a*k > b
            if k_lo * a <= b:
                k_lo += 1
            k_hi = (2 * b) // a
            if k_hi < k_lo:
                continue

            # For each k in [k_lo, k_hi], check if u = a*k - b divides b^2
            # Then v = b^2 // u, check v ≡ -b (mod a)
            # y = (u + b) // a, z = (v + b) // a
            # Check distinctness: x, y, z all distinct

            # Optimization: u = a*k - b. We need u | b^2.
            # Note that gcd(u, b) = gcd(a*k - b, b) = gcd(a*k, b) = gcd(a, b) * gcd(k, b/gcd(a,b))
            # But let's just iterate k and check divisibility.

            # Further optimization: if a is small, range is large. Let's cap.
            # For very large ranges, we can use a smarter approach.
            # But first let's try the direct approach with a reasonable cap.

            # Actually, for most cases the solution is found at small k values.
            # Let's iterate k from k_lo upward, but cap the number of iterations.
            max_k_iters = 500000
            k_count = 0
            for k in range(k_lo, k_hi + 1):
                if time.time() > deadline:
                    break
                k_count += 1
                if k_count > max_k_iters:
                    break
                u = a * k - b
                if u <= 0:
                    continue
                if b * b % u != 0:
                    continue
                v = b * b // u
                # Check v ≡ -b (mod a)
                if (v + b) % a != 0:
                    continue
                y = (u + b) // a
                z = (v + b) // a
                if y <= 0 or z <= 0:
                    continue
                if y > BOUND or z > BOUND:
                    continue
                # Ensure y <= z for canonical form (not strictly needed but helps)
                if y > z:
                    y, z = z, y
                # Check distinctness
                if x != y and y != z and x != z:
                    found = [x, y, z]
                    break

            if found:
                break

        answers.append(found)

    return answers
