import numpy as np
import time

N = 50
_start = time.time()
_DEADLINE = _start + 140.0

def _time_left():
    return _DEADLINE - time.time()

def compute_score(h):
    h = np.asarray(h, dtype=float)
    if h.shape != (N,):
        return 0.0
    if np.any(h < -1e-9):
        return 0.0
    h = np.maximum(h, 0.0)
    l1 = h.sum()
    if l1 <= 0:
        return 0.0
    l2_sq = float(h @ h)
    c = np.correlate(h, h, mode='full')
    sup_c = float(c.max())
    if sup_c <= 0:
        return 0.0
    return l2_sq / (l1 * sup_c)

def construct():
    best_h = np.zeros(N)
    best_s = 0.0

    def try_candidate(h):
        nonlocal best_h, best_s
        h = np.asarray(h, dtype=float)
        h = np.maximum(h, 0.0)
        s = compute_score(h)
        if s > best_s:
            best_s = s
            best_h = h.copy()
        return s

    # ---- 1. Explicit symmetric two-bump family (the key structure) ----
    # h[i] = a * exp(-((i-c)^2)/(2w^2)) for i in [0,c], mirrored on [N-1-c, N-1]
    # Zero in the middle. Optimizes over center c, width w, and normalizes.
    def two_bump(c, w):
        h = np.zeros(N)
        idx_l = np.arange(0, c + 1)
        idx_r = np.arange(N - 1 - c, N)
        h[idx_l] = np.exp(-((idx_l - c) ** 2) / (2.0 * w * w))
        h[idx_r] = np.exp(-((idx_r - (N - 1 - c)) ** 2) / (2.0 * w * w))
        s = h.sum()
        if s > 0:
            h /= s
        return h

    # Coarse grid over (c, w)
    for c in range(1, N // 2 + 1):
        if _time_left() < 5:
            break
        for w in np.arange(0.4, c + 1.0, 0.15):
            h = two_bump(c, w)
            s = compute_score(h)
            if s > best_s:
                best_s = s
                best_h = h.copy()

    # ---- 2. High-precision local optimization on the two-bump family ----
    # Parameter vector: [c, w] with small step search around the best.
    def _refine_two_bump(c0, w0):
        c = c0
        w = w0
        # local Nelder-Mead-like on (c, w)
        def obj(p):
            c_i = int(round(p[0]))
            w_i = max(0.2, p[1])
            return -compute_score(two_bump(c_i, w_i))
        # simple coordinate refinement
        for _ in range(60):
            if _time_left() < 5:
                break
            improved = False
            for dc in [0, 1, -1]:
                for dw in [-0.1, -0.05, 0, 0.05, 0.1]:
                    nc = int(round(c + dc))
                    nw = max(0.2, w + dw)
                    if 1 <= nc <= N // 2:
                        s = compute_score(two_bump(nc, nw))
                        if s > best_s + 1e-12:
                            c, w = nc, nw
                            improved = True
            if not improved:
                break
        return c, w

    # find best (c,w) from coarse grid, then refine
    bc, bw = 1, 1.0
    bs = 0.0
    for c in range(1, N // 2 + 1):
        for w in np.arange(0.4, c + 1.0, 0.15):
            s = compute_score(two_bump(c, w))
            if s > bs:
                bs, bc, bw = s, c, w
    bc, bw = _refine_two_bump(bc, bw)
    h = two_bump(bc, bw)
    try_candidate(h)

    # ---- 3. Multiplicative updates (Sinkhorn-style) on a smooth surrogate ----
    # Minimize f = (h@h) * (logsumexp(correlate(h,h)/T) * T) subject to sum=1, h>=0.
    # Use gradient-free multiplicative scaling driven by per-index "pressure".
    def multiplicative_updates(h0, iters=200):
        h = np.maximum(h0, 1e-12).copy()
        h /= h.sum()
        for _ in range(iters):
            if _time_left() < 5:
                break
            c = np.correlate(h, h, mode='full')
            l2_sq = float(h @ h)
            # pressure: indices that contribute to the peak correlation get reduced
            # peak location
            pk = int(np.argmax(c))
            # contribution of h[i] to the peak: h[i]*h[pk-i] (valid indices)
            # approximate gradient of sup_c w.r.t. h[i]
            j = pk - np.arange(N)
            contrib = np.zeros(N)
            valid = (j >= 0) & (j < N)
            contrib[valid] = h[j[valid]]
            # gradient of f = l2_sq * sup_c:
            # df/dh_i = 2 h_i * sup_c + l2_sq * (2 h_i * contrib_i)  [approx]
            grad = 2.0 * h * sup_c_val + l2_sq * 2.0 * h * contrib
            # multiplicative update: reduce where gradient is high
            scale = np.exp(-0.5 * grad / max(sup_c_val, 1e-12))
            h = h * scale
            h = np.maximum(h, 0.0)
            if h.sum() <= 0:
                break
            h /= h.sum()
        return h

    sup_c_val = 0.0  # placeholder, recomputed inside loop below properly
    # Redo multiplicative updates with correct sup_c tracking
    def multiplicative_updates2(h0, iters=300):
        h = np.maximum(h0, 1e-12).copy()
        h /= h.sum()
        for _ in range(iters):
            if _time_left() < 5:
                break
            c = np.correlate(h, h, mode='full')
            sup_c = float(c.max())
            l2_sq = float(h @ h)
            pk = int(np.argmax(c))
            j = pk - np.arange(N)
            contrib = np.zeros(N)
            valid = (j >= 0) & (j < N)
            contrib[valid] = h[j[valid]]
            grad = 2.0 * h * sup_c + l2_sq * 2.0 * h * contrib
            scale = np.exp(-0.3 * grad / max(sup_c, 1e-12))
            h = h * scale
            h = np.maximum(h, 0.0)
            if h.sum() <= 0:
                break
            h /= h.sum()
        return h

    h_mu = multiplicative_updates2(best_h, iters=200)
    try_candidate(h_mu)

    # ---- 4. Coordinate-descent polish on the best found ----
    def coordinate_descent(h, max_iter=100):
        h = h.copy()
        for _ in range(max_iter):
            if _time_left() < 5:
                break
            improved = False
            for i in range(N):
                old = h[i]
                best_local = old
                for factor in [0.3, 0.5, 0.7, 0.85, 1.0, 1.15, 1.3, 1.7, 2.5, 0.0]:
                    h[i] = max(0.0, old * factor)
                    s = compute_score(h)
                    if s > best_s + 1e-12:
                        best_local = h[i]
                        improved = True
                h[i] = best_local
            if not improved:
                break
        return h

    h_cd = coordinate_descent(best_h, max_iter=150)
    try_candidate(h_cd)

    # ---- 5. Final high-precision polish: tight line search on the two peaks ----
    # If best_h has a clear two-bump structure, do a fine 1D search on each peak height
    # and the gap, keeping the symmetry.
    # Re-run a very fine (c, w) grid around the best.
    if _time_left() > 10:
        for c in range(max(1, bc - 2), min(N // 2 + 1, bc + 3)):
            for w in np.arange(max(0.2, bw - 0.3), bw + 0.4, 0.03):
                h = two_bump(c, w)
                s = compute_score(h)
                if s > best_s:
                    best_s = s
                    best_h = h.copy()

    return list(best_h)
