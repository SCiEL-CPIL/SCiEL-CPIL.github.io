import time
import random
from math import gcd
from functools import reduce

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin for 60s limit

    n = 30

    def lcm(a, b):
        return a * b // gcd(a, b)

    L = reduce(lcm, range(1, n + 1))

    # W[i] = L // (i+1), so sum(c[i]/(i+1)) = (1/L) * sum(c[i]*W[i])
    # We want to minimize |sum(c[i]*W[i])| subject to it being nonzero
    W = [L // (i + 1) for i in range(n)]

    best_N = None
    best_coeffs = None
    best_abs_N = float('inf')

    def calc_N(coeffs):
        s = 0
        for i in range(n):
            c = coeffs[i]
            if c != 0:
                s += c * W[i]
        return s

    # Greedy initialization: start with all zeros, add terms one by one
    # choosing the coefficient that minimizes |N|
    def greedy_init():
        coeffs = [0] * n
        # Process in order of largest weight first for better greedy choices
        order = sorted(range(n), key=lambda i: -W[i])
        for i in order:
            best_c = 0
            best_abs = abs(calc_N(coeffs))
            for c in [-1, 0, 1]:
                if c == 0:
                    continue
                coeffs[i] = c
                val = abs(calc_N(coeffs))
                if val < best_abs:
                    best_abs = val
                    best_c = c
                coeffs[i] = 0
            coeffs[i] = best_c
        return coeffs

    def random_init():
        return [random.choice([-1, 0, 1]) for _ in range(n)]

    def local_search(coeffs, N):
        """Best-improvement local search. Returns (coeffs, N)."""
        improved = True
        while improved and time.time() < deadline:
            improved = False
            indices = list(range(n))
            random.shuffle(indices)

            for i in indices:
                old_val = coeffs[i]
                current_abs = abs(N)
                best_new_val = None
                best_new_N = None
                best_new_abs = current_abs

                for new_val in [-1, 0, 1]:
                    if new_val == old_val:
                        continue
                    delta = (new_val - old_val) * W[i]
                    new_N = N + delta
                    if new_N == 0:
                        continue
                    new_abs = abs(new_N)
                    if new_abs < best_new_abs:
                        best_new_abs = new_abs
                        best_new_val = new_val
                        best_new_N = new_N

                if best_new_val is not None:
                    coeffs[i] = best_new_val
                    N = best_new_N
                    improved = True

        return coeffs, N

    # Escape from local minima via random perturbation
    def perturb(coeffs, N):
        """Apply random flips to escape local minima. Returns (coeffs, N)."""
        num_flips = random.randint(2, 5)
        for _ in range(num_flips):
            idx = random.randint(0, n - 1)
            old = coeffs[idx]
            new = random.choice([-1, 0, 1])
            if new != old:
                delta = (new - old) * W[idx]
                new_N = N + delta
                if new_N != 0:
                    coeffs[idx] = new
                    N = new_N
                else:
                    # If it became 0, try another value
                    new2 = random.choice([-1, 0, 1])
                    if new2 != old:
                        delta2 = (new2 - old) * W[idx]
                        new_N2 = N + delta2
                        if new_N2 != 0:
                            coeffs[idx] = new2
                            N = new_N2
        # Ensure non-zero
        if N == 0:
            for k in range(n):
                old_k = coeffs[k]
                for v in [-1, 0, 1]:
                    if v == old_k:
                        continue
                    d = (v - old_k) * W[k]
                    ns = N + d
                    if ns != 0:
                        coeffs[k] = v
                        N = ns
                        break
                if N != 0:
                    break
        return coeffs, N

    # Main search loop
    # First, try greedy initialization
    coeffs = greedy_init()
    N = calc_N(coeffs)
    if N == 0:
        coeffs, N = perturb(coeffs, N)
    coeffs, N = local_search(coeffs, N)
    if abs(N) < best_abs_N:
        best_abs_N = abs(N)
        best_coeffs = coeffs[:]
        best_N = N

    # Random restarts with local search
    while time.time() < deadline:
        # Random initialization
        current_coeffs = random_init()
        current_N = calc_N(current_coeffs)

        if current_N == 0:
            current_coeffs, current_N = perturb(current_coeffs, current_N)

        # Local Search
        current_coeffs, current_N = local_search(current_coeffs, current_N)

        # If we made progress, check if it's the best
        if abs(current_N) < best_abs_N:
            best_abs_N = abs(current_N)
            best_coeffs = current_coeffs[:]
            best_N = current_N

            # If we've found a very small value, do more local search
            if best_abs_N <= 10:
                current_coeffs, current_N = local_search(current_coeffs, current_N)
                if abs(current_N) < best_abs_N:
                    best_abs_N = abs(current_N)
                    best_coeffs = current_coeffs[:]
                    best_N = current_N
        else:
            # Escape local minimum: random perturbation
            current_coeffs, current_N = perturb(current_coeffs, current_N)
            current_coeffs, current_N = local_search(current_coeffs, current_N)
            if abs(current_N) < best_abs_N:
                best_abs_N = abs(current_N)
                best_coeffs = current_coeffs[:]
                best_N = current_N

    # If no valid non-zero sum was found (extremely unlikely), return a default
    if best_coeffs is None:
        best_coeffs = [1] + [0] * (n - 1)

    return best_coeffs
