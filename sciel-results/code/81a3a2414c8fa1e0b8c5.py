import time
from math import gcd
from functools import reduce
from fractions import Fraction

def construct():
    start = time.time()
    deadline = start + 80  # safety margin before 90s kill

    n = 30
    # LCM of 1..30
    H = reduce(lambda a, b: a * b // gcd(a, b), range(1, n + 1))
    W = [H // i for i in range(1, n + 1)]

    # Split into two halves for meet-in-the-middle
    mid = 15
    weights_a = W[:mid]
    weights_b = W[mid:]

    # Generate all 3^15 sums for each half
    def gen_sums(weights):
        sums = [(0, [0] * len(weights))]
        for i, w in enumerate(weights):
            new_sums = []
            for s, coeffs in sums:
                for c in (-1, 0, 1):
                    new_s = s + c * w
                    new_coeffs = coeffs[:]
                    new_coeffs[i] = c
                    new_sums.append((new_s, new_coeffs))
            sums = new_sums
        return sums

    sums_a = gen_sums(weights_a)
    if time.time() > deadline:
        return _fallback(W, n)

    sums_b = gen_sums(weights_b)
    if time.time() > deadline:
        return _fallback(W, n)

    # Sort B by sum value for binary search
    sorted_b = sorted(sums_b, key=lambda x: x[0])
    b_vals = [x[0] for x in sorted_b]
    b_coeffs = [x[1] for x in sorted_b]

    import bisect

    best_abs = None
    best_total = None
    best_coeffs = None

    for a_sum, a_coeffs in sums_a:
        if time.time() > deadline:
            break
        target = -a_sum
        idx = bisect.bisect_left(b_vals, target)

        candidates = []
        if idx < len(b_vals):
            candidates.append(idx)
        if idx > 0:
            candidates.append(idx - 1)

        for ci in candidates:
            total = a_sum + b_vals[ci]
            if total == 0:
                continue
            abs_total = abs(total)
            if best_abs is None or abs_total < best_abs:
                best_abs = abs_total
                best_total = total
                best_coeffs = a_coeffs + b_coeffs[ci]

    if best_coeffs is not None:
        # Local search to try to improve
        best_coeffs = _local_search(best_coeffs, W, n, deadline)
        return best_coeffs

    return _fallback(W, n)


def _fallback(W, n):
    """Greedy fallback ensuring nonzero."""
    coeffs = [0] * n
    curr = 0
    for i in range(n, 0, -1):
        w = W[i - 1]
        best_c = 0
        best_v = curr
        for c in (-1, 0, 1):
            v = curr + c * w
            if abs(v) < abs(best_v):
                best_v = v
                best_c = c
        coeffs[i - 1] = best_c
        curr = best_v
    if curr == 0:
        min_idx = min(range(n), key=lambda i: W[i])
        if coeffs[min_idx] == 1:
            coeffs[min_idx] = -1
        else:
            coeffs[min_idx] = 1
    return coeffs


def _local_search(coeffs, W, n, deadline):
    """Iterative improvement: try single and pair changes."""
    total = sum(c * W[i] for i, c in enumerate(coeffs))
    if total == 0:
        min_idx = min(range(n), key=lambda i: W[i])
        if coeffs[min_idx] == 1:
            coeffs[min_idx] = -1
        else:
            coeffs[min_idx] = 1
        total = sum(c * W[i] for i, c in enumerate(coeffs))

    improved = True
    while improved and time.time() < deadline:
        improved = False
        # Single flips
        for i in range(n):
            for c_new in (-1, 0, 1):
                if c_new == coeffs[i]:
                    continue
                new_total = total + (c_new - coeffs[i]) * W[i]
                if new_total != 0 and abs(new_total) < abs(total):
                    coeffs[i] = c_new
                    total = new_total
                    improved = True
                    break
            if improved:
                break
        if improved:
            continue
        # Pair changes
        for i in range(n):
            if time.time() > deadline:
                break
            for j in range(i + 1, n):
                for ci in (-1, 0, 1):
                    if ci == coeffs[i]:
                        continue
                    for cj in (-1, 0, 1):
                        if cj == coeffs[j]:
                            continue
                        new_total = total + (ci - coeffs[i]) * W[i] + (cj - coeffs[j]) * W[j]
                        if new_total != 0 and abs(new_total) < abs(total):
                            coeffs[i] = ci
                            coeffs[j] = cj
                            total = new_total
                            improved = True
                            break
                    if improved:
                        break
                if improved:
                    break
            if improved:
                break
    return coeffs
