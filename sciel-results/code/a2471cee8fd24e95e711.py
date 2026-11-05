"""Structural change: randomised structured search + local refinement.

Instead of a fixed 2-parameter sinusoid family, we:
  1. Generate diverse random structured candidates (random frequency combs,
     random sign patterns, random smooth low-rank perturbations).
  2. Run a short local subgradient/coordinate refinement on the hard objective.
  3. Keep the best candidate found.
"""
N = 400
import time
import numpy as np
from scipy.optimize import minimize


def _autoconv(h):
    n = len(h)
    f = np.fft.rfft(h, n=2 * n)
    c = np.fft.irfft(f * f, n=2 * n)
    return c


def _hard(h):
    s = h.sum()
    if abs(s) < 1e-9:
        return 1e12
    conv = _autoconv(h)
    return abs(2.0 * N * np.max(conv) / (s * s))


def _local_refine(h0, max_iters=80, time_budget=5.0):
    """Subgradient-style local refinement on the hard objective."""
    t0 = time.time()
    h = h0.copy()
    s = h.sum()
    if abs(s) < 1e-9:
        return h
    h = h / s  # normalize sum to 1
    best_val = _hard(h)
    best_h = h.copy()
    n = N
    i_range = np.arange(n)
    
    for it in range(max_iters):
        if time.time() - t0 > time_budget:
            break
        conv = _autoconv(h)
        kmax = int(np.argmax(conv))
        lag = kmax - (n - 1)
        
        # Subgradient of max(conv) w.r.t. h
        sg = np.zeros(n)
        idx1 = lag - i_range
        mask1 = (idx1 >= 0) & (idx1 < n)
        sg[mask1] += h[idx1[mask1]]
        idx2 = i_range - lag
        mask2 = (idx2 >= 0) & (idx2 < n)
        sg[mask2] += h[idx2[mask2]]
        
        ss = h.sum()
        if abs(ss) < 1e-9:
            break
        sg_centered = sg - sg.mean()
        
        # Try several step sizes and directions
        improved = False
        for alpha in [1e-3, 5e-4, 1e-4, 2e-3, 5e-3]:
            for direction in [-1, 1]:
                cand = h + direction * alpha * sg_centered
                vv = _hard(cand)
                if vv < best_val - 1e-12:
                    best_val = vv
                    best_h = cand.copy()
                    h = cand.copy()
                    improved = True
                    break
            if improved:
                break
        
        if not improved:
            # Try coordinate descent: flip a few random coordinates
            rng_local = np.random.default_rng(hash((it, int(h.sum()))) % (2**31))
            for _ in range(20):
                if time.time() - t0 > time_budget:
                    break
                idx = rng_local.integers(0, n)
                for delta in [0.01, -0.01, 0.005, -0.005]:
                    cand = h.copy()
                    cand[idx] += delta
                    vv = _hard(cand)
                    if vv < best_val - 1e-12:
                        best_val = vv
                        best_h = cand.copy()
                        h = cand.copy()
                        break
    
    return best_h


def _gen_random_candidate(rng, mode):
    """Generate a diverse structured candidate."""
    i = np.arange(N)
    if mode == 0:
        # Random single frequency cosine
        freq = rng.uniform(1, N / 2)
        A = rng.uniform(0.5, 2.0)
        B = rng.uniform(-1.0, 1.0)
        h = A + B * np.cos(2 * np.pi * freq * i / N)
    elif mode == 1:
        # Random sign pattern (square wave)
        freq = rng.uniform(1, N / 2)
        phase = rng.uniform(0, 2 * np.pi)
        A = rng.uniform(0.5, 1.5)
        B = rng.uniform(0.3, 1.0)
        h = A + B * np.sign(np.cos(2 * np.pi * freq * i / N + phase))
    elif mode == 2:
        # Low-rank: sum of 2 random cosines
        f1 = rng.uniform(1, N / 3)
        f2 = rng.uniform(N / 3, N / 2)
        A = rng.uniform(0.5, 2.0)
        B1 = rng.uniform(-0.8, 0.8)
        B2 = rng.uniform(-0.8, 0.8)
        p1 = rng.uniform(0, 2 * np.pi)
        p2 = rng.uniform(0, 2 * np.pi)
        h = A + B1 * np.cos(2 * np.pi * f1 * i / N + p1) + B2 * np.cos(2 * np.pi * f2 * i / N + p2)
    elif mode == 3:
        # Random smooth noise + DC
        A = rng.uniform(0.5, 2.0)
        # Generate smooth noise via random Fourier series
        coeffs = rng.normal(0, 1, size=N // 2)
        noise = np.zeros(N)
        for k in range(1, min(10, N // 2)):
            noise += coeffs[k] * np.cos(2 * np.pi * k * i / N)
        h = A + 0.3 * noise
    else:
        # Random binary-like pattern
        p = rng.uniform(0.3, 0.7)
        h = rng.choice([-1.0, 1.0], size=N, p=[p, 1 - p])
        h += rng.uniform(0.5, 1.0)
    
    # Ensure sum is not zero
    s = h.sum()
    if abs(s) < 0.1:
        h += 1.0
    return h


def construct():
    t0 = time.time()
    DEADLINE = 135.0  # safety margin before 150s kill

    def time_left():
        return DEADLINE - (time.time() - t0)

    rng = np.random.default_rng(42)

    # Seed candidate
    best_h = np.ones(N, dtype=float)
    best_val = _hard(best_h)

    # Phase 1: diverse random structured candidates + local refinement
    n_candidates = 30
    for idx in range(n_candidates):
        if time_left() < 8.0:
            break
        mode = idx % 5
        h_cand = _gen_random_candidate(rng, mode)
        v = _hard(h_cand)
        if v < best_val:
            best_val = v
            best_h = h_cand.copy()
        
        # Local refine
        if time_left() < 5.0:
            break
        h_ref = _local_refine(h_cand, max_iters=60, time_budget=min(4.0, time_left() - 2.0))
        v_ref = _hard(h_ref)
        if v_ref < best_val:
            best_val = v_ref
            best_h = h_ref.copy()

    # Phase 2: L-BFGS-B polish on best with soft-max surrogate
    if time_left() > 15.0:
        def _soft_max_conv(h, beta):
            n = len(h)
            conv = _autoconv(h)
            shifted = beta * conv
            shifted -= np.max(shifted)
            shifted = np.clip(shifted, -500, 500)
            ez = np.exp(shifted)
            z = ez.sum()
            sm = np.max(conv) + np.log(z) / beta
            w = ez / z
            
            size = 4 * n
            Wf = np.fft.rfft(w, n=size)
            Hf = np.fft.rfft(h, n=size)
            cc = np.fft.irfft(Wf * np.conj(Hf), n=size)
            
            grad = np.zeros(n)
            i_range = np.arange(n)
            for i in i_range:
                if i == 0:
                    grad[i] = 2.0 * cc[0]
                else:
                    grad[i] = cc[i] + cc[size - i]
            return sm, grad

        def _obj_and_grad(h, beta):
            s = h.sum()
            if abs(s) < 1e-9:
                return 1e12, np.zeros_like(h)
            sm, g_sm = _soft_max_conv(h, beta)
            val = 2.0 * N * sm / (s * s)
            g = 2.0 * N * (g_sm * s - 2.0 * sm) / (s * s * s)
            return val, g

        for beta in [32.0, 64.0, 128.0]:
            if time_left() < 10.0:
                break
            try:
                def obj(x):
                    val, _ = _obj_and_grad(x, beta)
                    return val
                def gr(x):
                    _, g = _obj_and_grad(x, beta)
                    return g
                h_init = best_h.copy()
                s = h_init.sum()
                if abs(s) > 1e-9:
                    h_init = h_init / s
                res = minimize(obj, h_init, jac=gr, method='L-BFGS-B',
                               options={'maxiter': 300, 'ftol': 1e-14, 'gtol': 1e-12,
                                        'maxfun': 4000})
                h_new = res.x
                v = _hard(h_new)
                if v < best_val:
                    best_val = v
                    best_h = h_new.copy()
            except Exception:
                pass

    # Phase 3: final local refinement
    if time_left() > 3.0:
        h_ref = _local_refine(best_h, max_iters=100, time_budget=min(8.0, time_left() - 1.0))
        v_ref = _hard(h_ref)
        if v_ref < best_val:
            best_val = v_ref
            best_h = h_ref.copy()

    # Sanity
    if not np.all(np.isfinite(best_h)):
        best_h = np.ones(N)
        best_val = _hard(best_h)
    if abs(best_h.sum()) < 1e-10:
        best_h = np.ones(N)
        best_val = _hard(best_h)

    return list(best_h)
