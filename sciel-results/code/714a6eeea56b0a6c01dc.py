import time
import numpy as np
from scipy.optimize import minimize

N = 50

def _autoconv(h):
    return np.convolve(h, h, mode='full')

def _score(h):
    ac = _autoconv(h)
    l1 = float(np.sum(h))
    if l1 <= 0:
        return 0.0
    l2sq = float(np.dot(h, h))
    sup = float(np.max(ac))
    return l2sq / (l1 * sup)

def _reweight(h, iters, T, step, peak_mode, deadline):
    h = np.array(h, dtype=float)
    n = len(h)
    best = h.copy()
    best_s = _score(h)
    ac = _autoconv(h)
    peak = int(np.argmax(ac))
    for it in range(iters):
        if time.time() > deadline:
            break
        # peak index: either current argmax or fixed from start (per finding)
        if peak_mode == 'fixed':
            pass  # keep initial peak
        else:
            peak = int(np.argmax(ac))
        ac = _autoconv(h)
        L1 = float(np.sum(h))
        # weight: local spectral density of the current support at the peak
        w = np.zeros(n)
        for i in range(n):
            w[i] = ac[peak + i]
        w = np.maximum(w, 1e-12)
        # multiplicative update: allocate inversely proportional to local spectral density
        # scale to keep L1 fixed (nonnegative heights, L1-invariant objective)
        target = L1
        h_new = h * (target / (L1 + 1e-30))  # keep L1
        # water-filling: move mass away from high-weight positions
        h_new = h / (1.0 + step * w / w.max())
        h_new = h_new / (h_new.sum() + 1e-30) * L1
        h_new = np.clip(h_new, 0.0, None)
        if h_new.sum() <= 0:
            break
        s = _score(h_new)
        if s > best_s + 1e-12:
            best = h_new.copy()
            best_s = s
            h = h_new
        else:
            # anneal: occasionally accept, otherwise dampen step
            step *= 0.95
            h = best.copy()
    return best, best_s

def _polish(h, deadline):
    h = np.array(h, dtype=float)
    def neg_obj(x):
        return -_score(x)
    res = minimize(neg_obj, h, method='Nelder-Mead',
                   options={'maxiter': 400, 'xatol': 1e-10, 'fatol': 1e-10,
                            'adaptive': True})
    if time.time() > deadline:
        return h
    if res.x is not None and res.x.sum() > 0:
        x = np.clip(res.x, 0.0, None)
        if _score(x) > _score(h):
            h = x
    return h

def construct():
    start = time.time()
    deadline = start + 110.0  # safety margin before 120s / 150s kill
    rng = np.random.default_rng(12345)

    # diverse starting points
    starts = []
    starts.append(np.ones(N))
    # sparse peaks (Sidon-like)
    for _ in range(6):
        idx = rng.choice(N, size=8, replace=False)
        v = np.zeros(N)
        v[idx] = 1.0
        starts.append(v)
    # random
    for _ in range(6):
        v = rng.random(N) + 0.05
        starts.append(v / v.sum() * N)
    # alternating
    v = np.ones(N); v[1::2] *= 0.2
    starts.append(v)

    best = np.ones(N)
    best_s = _score(best)

    for s0 in starts:
        if time.time() > deadline:
            break
        for peak_mode in ('fixed', 'track'):
            if time.time() > deadline:
                break
            for iters in (300, 600):
                if time.time() > deadline:
                    break
                h, s = _reweight(s0, iters, T=1.0, step=0.5,
                                 peak_mode=peak_mode, deadline=deadline)
                if s > best_s + 1e-12:
                    best_s = s
                    best = h

    # high-precision polish
    if time.time() < deadline - 5.0:
        best = _polish(best, deadline)
        best_s = _score(best)

    return [float(x) for x in best]
