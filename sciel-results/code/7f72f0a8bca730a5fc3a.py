import numpy as np
import time

N = 600
W = 1.0 / 1200.0
X = np.linspace(-0.25, 0.25, N, endpoint=False)
# precompute for fast ratio: use a fine grid for the convolution
FINE_N = 8000
X_FINE = np.linspace(-0.25, 0.25, FINE_N, endpoint=False)
DX = 0.5 / FINE_N
# map fine grid to step cells
CELL = np.clip(((X_FINE + 0.25) / W).astype(np.int64), 0, N - 1)
N_FFT = 1 << (2 * FINE_N - 1).bit_length()

def ratio_from_h(h):
    h = np.asarray(h, dtype=np.float64)
    if h.size != N:
        h = np.resize(h, N)
    h = np.clip(h, 0.0, None)
    s = h.sum()
    if s < 1e-15:
        return float('inf')
    f = h[CELL]
    F = np.fft.rfft(f, N_FFT)
    conv = np.fft.irfft(F * F, N_FFT)[:2 * FINE_N - 1] * DX
    l2 = np.sum(f * f) * DX
    if l2 < 1e-15:
        return float('inf')
    return float(np.max(conv) / l2)

def param_to_h(base, peak, width, skew):
    g = np.exp(-0.5 * ((X - skew * 0.05) / width) ** 2)
    return base + peak * g

def construct():
    deadline = time.time() + 138.0

    best_h = np.ones(N)
    best_val = ratio_from_h(best_h)

    # ---- Phase 1: grid search over parametric family ----
    bases = [0.0, 0.02, 0.05, 0.1, 0.15, 0.2, 0.3]
    peaks = [0.2, 0.5, 1.0, 2.0, 3.0, 5.0, 8.0, 12.0]
    widths = [0.005, 0.008, 0.012, 0.018, 0.025, 0.035, 0.05, 0.07, 0.1]
    skews = [-1.0, 0.0, 1.0]

    for base in bases:
        for peak in peaks:
            for width in widths:
                if time.time() > deadline:
                    break
                for skew in skews:
                    h = param_to_h(base, peak, width, skew)
                    v = ratio_from_h(h)
                    if v < best_val:
                        best_val = v
                        best_h = h.copy()

    # ---- Phase 2: local Nelder-Mead-ish refinement on parameters ----
    if time.time() < deadline - 30.0:
        # start from best param guess found by grid (re-derive nearest)
        # do a fine local search by small perturbations of a good param set
        # We'll just do random perturbations around the best param region.
        # Determine a good starting param by scanning a denser grid around center.
        cand_params = []
        for base in [0.0, 0.02, 0.05, 0.1, 0.15]:
            for peak in [2.0, 4.0, 6.0, 8.0, 10.0, 14.0]:
                for width in [0.006, 0.01, 0.015, 0.02, 0.03, 0.04]:
                    cand_params.append((base, peak, width, 0.0))
        # evaluate and keep best few
        scored = []
        for p in cand_params:
            if time.time() > deadline - 30.0:
                break
            h = param_to_h(*p)
            v = ratio_from_h(h)
            scored.append((v, p))
        scored.sort(key=lambda t: t[0])
        for v, p in scored[:6]:
            # local random walk on the 4 params
            cur = list(p)
            cur_v = v
            for it in range(200):
                if time.time() > deadline - 25.0:
                    break
                step = 0.05
                trial = list(cur)
                trial[0] = max(0.0, cur[0] + np.random.randn() * 0.02)
                trial[1] = max(0.1, cur[1] + np.random.randn() * cur[1] * 0.15)
                trial[2] = max(0.003, cur[2] + np.random.randn() * cur[2] * 0.2)
                trial[3] = cur[3] + np.random.randn() * 0.2
                h = param_to_h(*trial)
                vv = ratio_from_h(h)
                if vv < cur_v:
                    cur, cur_v = trial, vv
                    if cur_v < best_val:
                        best_val = cur_v
                        best_h = h.copy()

    # ---- Phase 3: direct coordinate / block refinement on best_h ----
    if time.time() < deadline - 5.0:
        rng = np.random.default_rng(12345)
        for it in range(4000):
            if time.time() > deadline - 5.0:
                break
            mode = rng.integers(0, 3)
            if mode == 0:
                # perturb a contiguous block
                c = int(rng.integers(0, N))
                w = int(rng.integers(3, 25))
                lo = max(0, c - w // 2)
                hi = min(N, lo + w)
                amp = float(rng.normal(0.0, 0.15))
                h_try = best_h.copy()
                h_try[lo:hi] += amp * np.linspace(1, 0.3, hi - lo)
                h_try = np.clip(h_try, 0.0, None)
            elif mode == 1:
                # scale a small region
                c = int(rng.integers(0, N))
                w = int(rng.integers(2, 12))
                lo = max(0, c - w // 2)
                hi = min(N, lo + w)
                f = float(rng.uniform(0.5, 1.8))
                h_try = best_h.copy()
                h_try[lo:hi] *= f
            else:
                # single coordinate
                c = int(rng.integers(0, N))
                h_try = best_h.copy()
                h_try[c] = max(0.0, best_h[c] + float(rng.normal(0.0, 0.1)))
            v = ratio_from_h(h_try)
            if v < best_val:
                best_val = v
                best_h = h_try.copy()

    return list(best_h)
