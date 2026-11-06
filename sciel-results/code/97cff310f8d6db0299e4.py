import time
import numpy as np
from math import gcd
from functools import reduce

def construct():
    start_time = time.time()
    deadline = start_time + 85.0

    n = 30
    def lcm(a, b):
        return a * b // gcd(a, b)

    L = reduce(lcm, range(1, n + 1))
    weights = np.array([L // (i + 1) for i in range(n)], dtype=np.int64)
    INF = np.iinfo(np.int64).max

    # ---- Method 1: structured local search (fast, good bound) ----
    def local_search():
        best_coeffs = None
        best_val = INF
        inits = []
        inits.append([1] * n)
        inits.append([-1] * n)
        rng = np.random.RandomState(42)
        for _ in range(30):
            inits.append(list(rng.choice([-1, 0, 1], size=n)))
        for init in inits:
            coeffs = list(init)
            s = 0
            for i, c in enumerate(coeffs):
                if c != 0:
                    s += c * weights[i]
            cur = abs(s)
            improved = True
            iters = 0
            while improved and iters < 30 and time.time() < deadline - 20:
                improved = False
                iters += 1
                for i in range(n):
                    old = coeffs[i]
                    old_contrib = old * weights[i]
                    best_new_c = old
                    best_new_s = s
                    best_new_val = cur
                    eff_best = best_new_val if best_new_val != 0 else INF
                    for new_c in (-1, 0, 1):
                        if new_c == old:
                            continue
                        new_contrib = new_c * weights[i]
                        new_s = s + (new_contrib - old_contrib)
                        new_val = abs(new_s)
                        eff = new_val if new_val != 0 else INF
                        if eff < eff_best:
                            best_new_c = new_c
                            best_new_s = new_s
                            best_new_val = new_val
                            eff_best = eff
                    if best_new_c != old:
                        coeffs[i] = best_new_c
                        s = best_new_s
                        cur = best_new_val
                        improved = True
            if cur != 0 and cur < best_val:
                best_val = cur
                best_coeffs = list(coeffs)
        return best_coeffs, best_val

    best_coeffs, best_val = local_search()

    # ---- Method 2: exhaustive meet-in-the-middle with corrected neighbor search ----
    mid = n // 2
    w1 = weights[:mid]
    w2 = weights[mid:]

    def generate_sums_and_codes(weights_part):
        sums = np.array([0], dtype=np.int64)
        codes = np.array([0], dtype=np.int64)
        for i, w in enumerate(weights_part):
            new_sums = np.empty(len(sums) * 3, dtype=np.int64)
            new_codes = np.empty(len(codes) * 3, dtype=np.int64)
            pow3_i = 3 ** i
            new_sums[0::3] = sums - w
            new_codes[0::3] = codes
            new_sums[1::3] = sums
            new_codes[1::3] = codes + pow3_i
            new_sums[2::3] = sums + w
            new_codes[2::3] = codes + 2 * pow3_i
            sums = new_sums
            codes = new_codes
            if time.time() > deadline - 15:
                return None, None
        return sums, codes

    if time.time() < deadline - 15:
        sums1, codes1 = generate_sums_and_codes(w1)
        if sums1 is not None:
            sums2, codes2 = generate_sums_and_codes(w2)
            if sums2 is not None:
                sort_idx = np.argsort(sums2, kind='stable')
                sums2_sorted = sums2[sort_idx]
                codes2_sorted = codes2[sort_idx]
                N2 = len(sums2_sorted)

                targets = -sums1
                left = np.searchsorted(sums2_sorted, targets, side='left')
                right = np.searchsorted(sums2_sorted, targets, side='right')

                ex_best_abs = INF
                ex_best_code1 = 0
                ex_best_code2 = 0

                for cand_idx in (left - 1, left, right):
                    idx_c = np.clip(cand_idx, 0, N2 - 1)
                    s2_c = sums2_sorted[idx_c]
                    total = sums1 + s2_c
                    abs_t = np.abs(total)
                    abs_t[total == 0] = INF
                    min_t = np.min(abs_t)
                    if min_t < ex_best_abs:
                        ex_best_abs = min_t
                        idx_best = np.argmin(abs_t)
                        ex_best_code1 = codes1[idx_best]
                        ex_best_code2 = codes2_sorted[idx_c[idx_best]]

                if ex_best_abs != INF and ex_best_abs < best_val:
                    best_val = ex_best_abs
                    def decode_code(code, n_coeffs):
                        coeffs = []
                        for i in range(n_coeffs):
                            digit = code % 3
                            if digit == 0:
                                coeffs.append(-1)
                            elif digit == 1:
                                coeffs.append(0)
                            else:
                                coeffs.append(1)
                            code //= 3
                        return coeffs
                    c1 = decode_code(ex_best_code1, mid)
                    c2 = decode_code(ex_best_code2, n - mid)
                    best_coeffs = c1 + c2

    # Fallback
    if best_coeffs is None or best_val == 0 or best_val == INF:
        best_coeffs = [1, -1] + [0] * 28

    # Verify non-zero
    s = 0
    for i, c in enumerate(best_coeffs):
        if c != 0:
            s += c * weights[i]
    if s == 0:
        # find a valid non-zero by flipping one coefficient
        for i in range(n):
            old = best_coeffs[i]
            for new_c in (-1, 0, 1):
                if new_c == old:
                    continue
                best_coeffs[i] = new_c
                s_new = s + (new_c - old) * weights[i]
                if s_new != 0:
                    break
            if s != 0:
                break
            s = s_new
        # If still zero, use a known valid
        s = 0
        for i, c in enumerate(best_coeffs):
            if c != 0:
                s += c * weights[i]
        if s == 0:
            best_coeffs = [1, -1] + [0] * 28

    return best_coeffs
