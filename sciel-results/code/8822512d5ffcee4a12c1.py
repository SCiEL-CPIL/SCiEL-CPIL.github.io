import random
import time
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

    # Meet-in-the-middle: split into two halves of 15
    half = 15
    W1 = W[:half]  # weights for indices 0..14
    W2 = W[half:]  # weights for indices 15..29

    def enumerate_half(weights, offset=0):
        """Enumerate all 3^15 combinations, return list of (sum, coeffs_list)"""
        results = []
        m = len(weights)
        # Use iterative generation to avoid recursion depth issues
        coeffs = [0] * m
        for mask in range(3 ** m):
            tmp = mask
            s = 0
            for i in range(m):
                digit = tmp % 3
                tmp //= 3
                c = digit - 1  # -1, 0, 1
                coeffs[i] = c
                if c != 0:
                    s += c * weights[i]
            results.append((s, coeffs[:]))
        return results

    # Local search for refinement
    def local_search(coeffs, N):
        """Best-improvement local search with delta updates."""
        if N == 0:
            return None, None
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

    def perturb(coeffs, N, num_flips=None):
        """Apply random flips to escape local minima."""
        if num_flips is None:
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
                    new2 = random.choice([-1, 0, 1])
                    if new2 != old:
                        delta2 = (new2 - old) * W[idx]
                        new_N2 = N + delta2
                        if new_N2 != 0:
                            coeffs[idx] = new2
                            N = new_N2
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

    # Phase 1: Meet-in-the-middle to find near-optimal solution
    # Enumerate first half
    half1_results = enumerate_half(W1)
    # Sort by sum value for binary search
    half1_results.sort(key=lambda x: x[0])
    half1_sums = [r[0] for r in half1_results]

    # For each combination in second half, find closest match in first half
    # such that total is nonzero and minimized
    half2_results = enumerate_half(W2)

    # We want to minimize |s1 + s2| where s1 != -s2 (to keep total nonzero)
    # For each s2, find s1 closest to -s2 but not equal to -s2

    import bisect

    for s2, coeffs2 in half2_results:
        if time.time() > deadline - 5:
            break
        target = -s2
        # Find position where target would be inserted
        pos = bisect.bisect_left(half1_sums, target)

        # Check candidates around pos
        candidates = []
        for p in [pos - 2, pos - 1, pos, pos + 1, pos + 2]:
            if 0 <= p < len(half1_sums):
                candidates.append(p)

        best_local_N = None
        best_local_c1 = None
        best_local_c2 = None
        best_local_abs = float('inf')

        for p in candidates:
            s1 = half1_sums[p]
            total = s1 + s2
            if total == 0:
                continue
            abs_total = abs(total)
            if abs_total < best_local_abs:
                best_local_abs = abs_total
                best_local_N = total
                best_local_c1 = half1_results[p][1]
                best_local_c2 = coeffs2

        if best_local_N is not None and best_local_abs < best_abs_N:
            best_abs_N = best_local_abs
            best_N = best_local_N
            best_coeffs = best_local_c1 + best_local_c2

        # Early exit if we found 1
        if best_abs_N == 1:
            break

    # Phase 2: Local search refinement on the best found
    if best_coeffs is not None and time.time() < deadline:
        current_coeffs = best_coeffs[:]
        current_N = calc_N(current_coeffs)
        if current_N != 0:
            rc, rN = local_search(current_coeffs, current_N)
            if rN is not None and rN != 0 and abs(rN) < best_abs_N:
                best_abs_N = abs(rN)
                best_N = rN
                best_coeffs = rc[:]

    # Phase 3: Random restarts with local search if time remains
    while time.time() < deadline - 2:
        coeffs = [random.choice([-1, 0, 1]) for _ in range(n)]
        N = calc_N(coeffs)
        if N == 0:
            coeffs, N = perturb(coeffs, N)
        if N == 0:
            continue

        rc, rN = local_search(coeffs, N)
        if rN is not None and rN != 0:
            if abs(rN) < best_abs_N:
                best_abs_N = abs(rN)
                best_N = rN
                best_coeffs = rc[:]

        # Try perturbation + local search
        if time.time() < deadline - 2:
            pc, pN = perturb(best_coeffs, best_N)
            if pN != 0:
                pc2, pN2 = local_search(pc, pN)
                if pN2 is not None and pN2 != 0 and abs(pN2) < best_abs_N:
                    best_abs_N = abs(pN2)
                    best_N = pN2
                    best_coeffs = pc2[:]

        if best_abs_N == 1:
            break

    # Fallback
    if best_coeffs is None:
        fallback = [0] * 28 + [1, -1]
        return fallback

    return best_coeffs
