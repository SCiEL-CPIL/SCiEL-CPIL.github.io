"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import numpy as np
from math import gcd


def _evaluate(coeffs):
    num = 0
    den = 1
    for i, c in enumerate(coeffs):
        if c == 0:
            continue
        d = i + 1
        g = gcd(den, d)
        den = den // g * d
        num = num * (d // g) + c * (den // d)
    g = gcd(abs(num), den)
    if g:
        num //= g
        den //= g
    return num, den


def _abs_val(num, den):
    return abs(num) / den


def _lcm(a, b):
    return a * b // gcd(a, b)


def _decode(code, coeffs, offset):
    temp = int(code)
    for k in range(15):
        digit = temp % 3
        if digit == 2:
            coeffs[offset + k] = -1
        elif digit == 1:
            coeffs[offset + k] = 1
        else:
            coeffs[offset + k] = 0
        temp //= 3


def _meet_in_middle(weights):
    w1 = weights[:15]
    w2 = weights[15:]

    def gen(ws):
        sums = np.array([0], dtype=np.int64)
        codes = np.array([0], dtype=np.int64)
        for k, w in enumerate(ws):
            p3 = 3 ** k
            sums = np.concatenate([sums - w, sums, sums + w])
            codes = np.concatenate([codes - p3, codes, codes + p3])
        return sums, codes

    sums1, codes1 = gen(w1)
    sums2, codes2 = gen(w2)

    idx1 = np.argsort(sums1)
    sums1 = sums1[idx1]
    codes1 = codes1[idx1]
    idx2 = np.argsort(sums2)
    sums2 = sums2[idx2]
    codes2 = codes2[idx2]

    targets = -sums1
    idxs = np.searchsorted(sums2, targets)
    n2 = len(sums2)
    idxs_c = np.clip(idxs, 0, n2 - 1)
    idxs_m = np.clip(idxs - 1, 0, n2 - 1)

    sa = sums1 + sums2[idxs_c]
    sb = sums1 + sums2[idxs_m]
    ca = codes1
    cb = codes1
    da = codes2[idxs_c]
    db = codes2[idxs_m]

    all_sums = np.concatenate([sa, sb])
    all_c1 = np.concatenate([ca, cb])
    all_c2 = np.concatenate([da, db])

    mask = all_sums != 0
    all_sums = all_sums[mask]
    all_c1 = all_c1[mask]
    all_c2 = all_c2[mask]

    if len(all_sums) == 0:
        return None

    min_idx = int(np.argmin(np.abs(all_sums)))
    return int(all_c1[min_idx]), int(all_c2[min_idx])


def construct():
    deadline = time.time() + 75

    # Default safe fallback: 1/29 - 1/30 = 1/870
    best_coeffs = [0] * 28 + [1, -1]

    # Exact global optimum via meet-in-the-middle over all 3^30 subsums.
    try:
        L = 1
        for i in range(1, 31):
            L = _lcm(L, i)
        weights = [L // (i + 1) for i in range(30)]

        res = _meet_in_middle(weights)
        if res is not None and time.time() < deadline - 5:
            c1, c2 = res
            coeffs = [0] * 30
            _decode(c1, coeffs, 0)
            _decode(c2, coeffs, 15)

            num, den = _evaluate(coeffs)
            if num != 0:
                best_coeffs = coeffs
    except Exception:
        pass

    # Safety net: a quick local search in case the exact path failed.
    try:
        if time.time() < deadline - 8:
            best_val = _abs_val(*_evaluate(best_coeffs))
            random.seed(42)
            for restart in range(300):
                if time.time() > deadline - 6:
                    break
                coeffs = [0] * 30
                n = random.randint(3, 15)
                pos = random.sample(range(30), n)
                for p in pos:
                    coeffs[p] = random.choice([-1, 1])
                for _ in range(60):
                    if time.time() > deadline - 6:
                        break
                    num, den = _evaluate(coeffs)
                    if num == 0:
                        idx = random.randint(0, 29)
                        coeffs[idx] = random.choice([-1, 0, 1])
                        continue
                    cur = _abs_val(num, den)
                    if cur < best_val:
                        best_val = cur
                        best_coeffs = list(coeffs)
                    idx = random.randint(0, 29)
                    old = coeffs[idx]
                    new = random.choice([-1, 0, 1])
                    if new == old:
                        continue
                    coeffs[idx] = new
                    n2, d2 = _evaluate(coeffs)
                    if n2 == 0:
                        coeffs[idx] = old
                    else:
                        v2 = _abs_val(n2, d2)
                        if v2 >= cur:
                            coeffs[idx] = old
    except Exception:
        pass

    return best_coeffs
# EVOLVE-BLOCK-END
