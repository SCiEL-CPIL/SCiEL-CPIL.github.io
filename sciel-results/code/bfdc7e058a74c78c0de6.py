# EVOLVE-BLOCK-START
import time
import numpy as np

_N = 256

def _max_ap_discrepancy(s):
    """Max |sum| over all APs in [1..n]. s: np array of +1/-1."""
    n = len(s)
    max_abs = 0
    for d in range(1, n + 1):
        for i in range(min(d, n)):
            seq = s[i::d]
            if len(seq) == 0:
                continue
            cums = np.cumsum(seq)
            m = int(np.max(np.abs(cums)))
            if m > max_abs:
                max_abs = m
                if max_abs >= n:
                    return max_abs
    return max_abs

def _walsh_signs(n):
    signs = np.ones(n, dtype=np.int32)
    for k in range(n):
        c = bin(k).count('1')
        signs[k] = 1 if c % 2 == 0 else -1
    return signs

def _vdcorput_signs(n, base):
    signs = np.zeros(n, dtype=np.int32)
    for k in range(n):
        val = 0.0
        f = 1.0 / base
        m = k
        while m > 0:
            val += f * (m % base)
            m //= base
            f /= base
        signs[k] = 1 if val < 0.5 else -1
    return signs

def construct():
    t0 = time.time()
    deadline = t0 + 84.0
    n = _N
    best_signs = None
    best_disc = n + 1

    # Diverse initial candidates
    candidates = []
    candidates.append(_walsh_signs(n))
    candidates.append(_vdcorput_signs(n, 2))
    candidates.append(_vdcorput_signs(n, 3))
    candidates.append(_vdcorput_signs(n, 5))

    rng = np.random.RandomState(42)
    for _ in range(8):
        candidates.append(rng.choice([-1, 1], size=n).astype(np.int32))

    # Alternating
    alt = np.ones(n, dtype=np.int32)
    alt[1::2] = -1
    candidates.append(alt)

    # Block patterns
    for bs in [2, 4, 8, 16, 32]:
        pat = np.ones(n, dtype=np.int32)
        for i in range(n):
            if (i // bs) % 2 == 1:
                pat[i] = -1
        candidates.append(pat)

    # Evaluate
    for cand in candidates:
        if time.time() > deadline:
            break
        d = _max_ap_discrepancy(cand)
        if d < best_disc:
            best_disc = d
            best_signs = cand.copy()

    if best_signs is None:
        best_signs = candidates[0].copy()
        best_disc = _max_ap_discrepancy(best_signs)

    # Simulated annealing
    cur = best_signs.copy()
    cur_disc = best_disc
    temp = 3.0
    min_temp = 0.05
    cool = 0.9995
    rng_sa = np.random.RandomState(7)
    no_improve = 0

    while time.time() < deadline and temp > min_temp:
        idx = rng_sa.randint(0, n)
        cur[idx] = -cur[idx]
        nd = _max_ap_discrepancy(cur)
        delta = nd - cur_disc

        if delta <= 0 or (temp > 0 and rng_sa.rand() < np.exp(-delta / temp)):
            cur_disc = nd
            if nd < best_disc:
                best_disc = nd
                best_signs = cur.copy()
                no_improve = 0
            else:
                no_improve += 1
        else:
            cur[idx] = -cur[idx]
            no_improve += 1

        temp *= cool

        # Plateau escape: if stuck, do a random restart with perturbation
        if no_improve > 500:
            # Perturb current solution
            for _ in range(rng_sa.randint(5, 20)):
                j = rng_sa.randint(0, n)
                cur[j] = -cur[j]
            cur_disc = _max_ap_discrepancy(cur)
            if cur_disc < best_disc:
                best_disc = cur_disc
                best_signs = cur.copy()
            no_improve = 0
            temp = max(temp, 2.0)

    return [int(x) for x in best_signs]
# EVOLVE-BLOCK-END
