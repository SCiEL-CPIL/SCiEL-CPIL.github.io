import numpy as np
import time
from scipy.optimize import minimize

N = 50

def compute_ratio(h):
    h = np.asarray(h, dtype=np.float64)
    l1 = np.sum(h)
    if l1 == 0:
        return 0.0
    l2_sq = np.sum(h * h)
    autocorr = np.convolve(h, h, mode='full')
    sup_norm = np.max(autocorr)
    if sup_norm == 0:
        return 0.0
    return l2_sq / (l1 * sup_norm)

def _sidon_set(n, max_size):
    """Greedy Sidon set in [0, n)."""
    if max_size <= 0:
        return []
    sidon = [0]
    used_sums = {0}
    for cand in range(1, n):
        ok = True
        for i in sidon:
            s = i + cand
            if s in used_sums:
                ok = False
                break
        if ok:
            for i in sidon:
                if i + cand in used_sums:
                    ok = False
                    break
        if ok:
            sidon.append(cand)
            used_sums.add(cand + cand)
            for i in sidon[:-1]:
                used_sums.add(i + cand)
        if len(sidon) >= max_size:
            break
    return sidon

def construct():
    start_time = time.time()
    DEADLINE = start_time + 140.0

    best_h = np.ones(N)
    best_score = compute_ratio(best_h)

    np.random.seed(42)

    # Build a diverse pool of initializations centered on Sidon-like supports
    inits = []

    # Sidon sets of various sizes
    for sz in [6, 8, 10, 12, 14]:
        s = _sidon_set(N, sz)
        h = np.zeros(N)
        for idx in s:
            h[idx] = 1.0
        inits.append(h)

    # Random sparse patterns
    for _ in range(30):
        k = np.random.randint(4, 16)
        idx = np.random.choice(N, size=k, replace=False)
        h = np.zeros(N)
        h[idx] = np.random.uniform(0.3, 1.5, size=k)
        inits.append(h)

    # Structured: uniform, alternating, blocks, gaussians
    inits.append(np.ones(N))
    h_alt = np.zeros(N); h_alt[::2] = 1.0; inits.append(h_alt)
    h_sp3 = np.zeros(N); h_sp3[::3] = 1.0; inits.append(h_sp3)
    t = np.linspace(-1, 1, N)
    inits.append(np.exp(-t**2 / 0.3))
    inits.append(np.maximum(1 - t**2, 0.05))
    inits.append(np.maximum(1.0 + 2.0*np.exp(-((t-0.6)**2)/0.1) + 2.0*np.exp(-((t+0.6)**2)/0.1), 0.05))
    for bs in [3, 5, 8, 12]:
        st = np.random.randint(0, N - bs + 1)
        hb = np.zeros(N); hb[st:st+bs] = 1.0; inits.append(hb)

    # Evaluate raw initializations
    for h in inits:
        h = np.maximum(h, 0)
        r = compute_ratio(h)
        if r > best_score:
            best_score = r
            best_h = h.copy()

    # Normalize helper: ratio is scale-invariant, so we can fix L1=1 for optimization
    def neg_ratio(x):
        x = np.maximum(x, 0)
        s = np.sum(x)
        if s < 1e-15:
            return 0.0
        return -compute_ratio(x / s)

    # Optimize each promising initialization with Nelder-Mead
    # Sort inits by raw score descending to spend more time on good ones
    scored_inits = []
    for h in inits:
        h = np.maximum(h, 0)
        s = np.sum(h)
        if s < 1e-15:
            continue
        hn = h / s
        scored_inits.append((compute_ratio(hn), hn))
    scored_inits.sort(key=lambda x: -x[0])

    # Limit number of optimizations to fit budget
    max_opts = 25
    for sc, h0 in scored_inits[:max_opts]:
        if time.time() > DEADLINE - 5:
            break
        try:
            res = minimize(neg_ratio, h0.copy(), method='Nelder-Mead',
                           options={'maxiter': 600, 'xatol': 1e-9, 'fatol': 1e-11, 'adaptive': True})
            hc = np.maximum(res.x, 0)
            s = np.sum(hc)
            if s > 1e-15:
                hc = hc / s
                sc2 = compute_ratio(hc)
                if sc2 > best_score:
                    best_score = sc2
                    best_h = hc.copy()
        except Exception:
            continue

    # Local refinement around the best found
    if time.time() < DEADLINE - 10:
        try:
            res = minimize(neg_ratio, best_h.copy(), method='Nelder-Mead',
                           options={'maxiter': 3000, 'xatol': 1e-11, 'fatol': 1e-13, 'adaptive': True})
            hc = np.maximum(res.x, 0)
            s = np.sum(hc)
            if s > 1e-15:
                hc = hc / s
                sc2 = compute_ratio(hc)
                if sc2 > best_score:
                    best_score = sc2
                    best_h = hc.copy()
        except Exception:
            pass

    # Second refinement pass if time remains
    if time.time() < DEADLINE - 8:
        try:
            res = minimize(neg_ratio, best_h.copy(), method='Nelder-Mead',
                           options={'maxiter': 2000, 'xatol': 1e-12, 'fatol': 1e-14, 'adaptive': True})
            hc = np.maximum(res.x, 0)
            s = np.sum(hc)
            if s > 1e-15:
                hc = hc / s
                sc2 = compute_ratio(hc)
                if sc2 > best_score:
                    best_score = sc2
                    best_h = hc.copy()
        except Exception:
            pass

    # Return normalized (L1=1) which is scale-invariant and safe
    l1 = np.sum(best_h)
    if l1 > 0:
        best_h = best_h / l1

    return list(best_h)
