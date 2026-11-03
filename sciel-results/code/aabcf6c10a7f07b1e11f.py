"""Combined program: fast FFT ratio + parametric Gaussian search + local refinement."""
import numpy as np
import time

N = 600
W = 1.0 / 1200.0
X = np.linspace(-0.25, 0.25, N, endpoint=False)

# Precompute for fast convolution ratio
FINE_N = 10000
X_FINE = np.linspace(-0.25, 0.25, FINE_N, endpoint=False)
DX = 0.5 / FINE_N
CELL = np.clip(((X_FINE + 0.25) / W).astype(np.int64), 0, N - 1)
N_FFT = 1 << (2 * FINE_N - 1).bit_length()

def ratio_from_h(h):
    """Compute normalized max autoconvolution ratio using FFT."""
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
    """Generate step heights from a Gaussian bump with base offset."""
    g = np.exp(-0.5 * ((X - skew * 0.05) / width) ** 2)
    return base + peak * g

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s kill
    
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
    
    # ---- Phase 2: local refinement on parameters via random walk ----
    if time.time() < deadline - 30.0:
        # Generate candidate parameter sets and score them
        cand_params = []
        for base in [0.0, 0.02, 0.05, 0.1, 0.15]:
            for peak in [2.0, 4.0, 6.0, 8.0, 10.0, 14.0]:
                for width in [0.006, 0.01, 0.015, 0.02, 0.03, 0.04]:
                    cand_params.append((base, peak, width, 0.0))
        
        scored = []
        for p in cand_params:
            if time.time() > deadline - 30.0:
                break
            h = param_to_h(*p)
            v = ratio_from_h(h)
            scored.append((v, p))
        scored.sort(key=lambda t: t[0])
        
        # Refine the top candidates with random walk
        for v, p in scored[:6]:
            cur = list(p)
            cur_v = v
            for it in range(200):
                if time.time() > deadline - 25.0:
                    break
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
    
    # ---- Phase 3: diverse random candidates ----
    if time.time() < deadline - 20.0:
        x_norm = np.linspace(-1.0, 1.0, N)
        candidates = []
        # Linear ramps
        candidates.append(1.0 + x_norm)
        candidates.append(1.0 - x_norm)
        candidates.append(0.1 + 0.9 * x_norm)
        candidates.append(0.9 - 0.1 * x_norm)
        # Asymmetric quadratic
        candidates.append(1.5 + x_norm - x_norm * x_norm)
        candidates.append(1.5 - x_norm - x_norm * x_norm)
        # Shifted Gaussians
        for shift in [-0.4, -0.3, -0.2, -0.1, 0.1, 0.2, 0.3, 0.4]:
            for width in [0.3, 0.5, 0.7, 1.0]:
                candidates.append(np.exp(-((x_norm - shift) ** 2) / (2.0 * width ** 2)))
        # Asymmetric power
        for p in [0.5, 1.5, 2.0, 3.0]:
            candidates.append(np.clip(1.0 + x_norm, 0, None) ** p)
            candidates.append(np.clip(1.0 - x_norm, 0, None) ** p)
        # Sinusoidal
        candidates.append(1.0 + 0.5 * np.sin(np.pi * x_norm))
        candidates.append(1.0 + 0.5 * np.cos(np.pi * x_norm))
        
        for h in candidates:
            if time.time() > deadline - 20.0:
                break
            h = np.clip(h, 0, None)
            if h.sum() < 1e-12:
                continue
            val = ratio_from_h(h)
            if val < best_val:
                best_val = val
                best_h = h.copy()
        
        # Random search with diverse shapes
        rng = np.random.default_rng(42)
        n_random = 4000
        for i in range(n_random):
            if time.time() > deadline - 20.0:
                break
            r = rng.random()
            if r < 0.2:
                n_ctrl = int(rng.integers(8, 40))
                ctrl_y = rng.random(n_ctrl)
                h = np.interp(np.linspace(0, N - 1, N), np.linspace(0, n_ctrl - 1, n_ctrl), ctrl_y)
            elif r < 0.4:
                center = float(rng.uniform(0.15, 0.85)) * (N - 1)
                width = float(rng.uniform(0.08, 0.5)) * N
                idx = np.arange(N)
                h = np.exp(-((idx - center) ** 2) / (2.0 * width ** 2))
            elif r < 0.55:
                slope = float(rng.uniform(-0.8, 0.8))
                intercept = float(rng.uniform(0.2, 0.8))
                idx = np.linspace(-1, 1, N)
                h = intercept + slope * idx + rng.normal(0, 0.03, N)
            elif r < 0.7:
                n_knots = int(rng.integers(3, 12))
                knot_x = np.sort(rng.random(n_knots) * N)
                knot_y = rng.random(n_knots)
                h = np.interp(np.arange(N), knot_x, knot_y)
            elif r < 0.85:
                a = rng.normal(0, 1, 4)
                idx = np.linspace(-1, 1, N)
                h = a[0] + a[1] * idx + a[2] * idx ** 2 + a[3] * idx ** 3
            else:
                h = rng.random(N)
            
            h = np.clip(h, 0, None)
            if h.sum() < 1e-12:
                continue
            val = ratio_from_h(h)
            if val < best_val:
                best_val = val
                best_h = h.copy()
    
    # ---- Phase 4: direct coordinate / block refinement on best_h ----
    if time.time() < deadline - 5.0:
        rng = np.random.default_rng(12345)
        for it in range(4000):
            if time.time() > deadline - 5.0:
                break
            mode = int(rng.integers(0, 3))
            if mode == 0:
                c = int(rng.integers(0, N))
                w = int(rng.integers(3, 25))
                lo = max(0, c - w // 2)
                hi = min(N, lo + w)
                amp = float(rng.normal(0.0, 0.15))
                h_try = best_h.copy()
                h_try[lo:hi] += amp * np.linspace(1, 0.3, hi - lo)
                h_try = np.clip(h_try, 0.0, None)
            elif mode == 1:
                c = int(rng.integers(0, N))
                w = int(rng.integers(2, 12))
                lo = max(0, c - w // 2)
                hi = min(N, lo + w)
                f = float(rng.uniform(0.5, 1.8))
                h_try = best_h.copy()
                h_try[lo:hi] *= f
            else:
                c = int(rng.integers(0, N))
                h_try = best_h.copy()
                h_try[c] = max(0.0, best_h[c] + float(rng.normal(0.0, 0.1)))
            v = ratio_from_h(h_try)
            if v < best_val:
                best_val = v
                best_h = h_try.copy()
    
    return list(best_h)
# EVOLVE-BLOCK-END
