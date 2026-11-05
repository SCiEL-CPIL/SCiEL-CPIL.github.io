"""Rewritten program: minimize value = abs(2*N*max(convolve(h,h))/sum(h)^2)."""
import time
import numpy as np

N = 400
_DEADLINE = 130.0  # safety margin under 150s wall-clock

def _value(h):
    s = np.sum(h)
    if s == 0 or not np.all(np.isfinite(h)):
        return np.inf
    conv = np.convolve(h, h)
    return abs(2.0 * N * np.max(conv) / (s * s))

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import differential_evolution, minimize

N = 400

def _value_from_h(h):
    h = np.asarray(h, dtype=float)
    s = np.sum(h)
    if s == 0.0 or not np.all(np.isfinite(h)):
        return np.inf
    conv = np.convolve(h, h)
    m = np.max(conv)
    return abs(2.0 * N * m / (s * s))

def _smooth_max_value(h, beta=50.0):
    """Smooth surrogate for max(convolve(h,h)) using log-sum-exp."""
    h = np.asarray(h, dtype=float)
    s = np.sum(h)
    if s == 0.0 or not np.all(np.isfinite(h)):
        return np.inf
    conv = np.convolve(h, h)
    # Smooth max: (1/beta) * log(sum(exp(beta*x)))
    max_c = np.max(conv)
    shifted = beta * (conv - max_c)
    log_sum = max_c + (1.0/beta) * np.log(np.sum(np.exp(shifted)))
    return abs(2.0 * N * log_sum / (s * s))

def _chirp(p):
    # p = [phase, chirp_rate, amp_cos, amp_sin, base]
    phase, rate, ac, as_, base = p
    i = np.arange(N, dtype=float)
    t = i / N
    freq = rate * t
    ang = phase + 2.0 * np.pi * np.cumsum(freq)  # chirp
    h = base + ac * np.cos(ang) + as_ * np.sin(ang)
    return h

def _zadoff_chu_h(p):
    """Zadoff-Chu inspired sequence: h[i] = Re(exp(2*pi*j*(u*i + i^2/2)/N))
    p = [phase, u, base] where u is integer-like root index"""
    phase, u, base = p
    i = np.arange(N, dtype=float)
    # Zadoff-Chu: x[i] = exp(-j*pi*u*i*(i+1)/N) for even N
    # Use real part with added DC offset
    ang = phase - np.pi * u * i * (i + 1) / N
    h = base + np.cos(ang)
    return h

def _two_level(p):
    # p = [a, b, frac, offset]
    a, b, frac, offset = p
    k = int(round(frac * N))
    h = np.empty(N)
    h[:k] = a
    h[k:] = b
    # add a small linear ramp to break symmetry
    h = h + offset * (np.arange(N, dtype=float) / N)
    return h

def _dft_basis_h(p):
    """Represent h as sum of low-frequency DFT basis functions.
    p = [dc, a1, b1, a2, b2, a3, b3, a4, b4] (10 params)"""
    i = np.arange(N, dtype=float)
    h = np.zeros(N)
    h += p[0]  # DC
    for k in range(1, 5):
        h += p[2*k-1] * np.cos(2*np.pi*k*i/N)
        h += p[2*k] * np.sin(2*np.pi*k*i/N)
    return h

def _random_h(rng):
    # mix of structured patterns
    kind = rng.integers(0, 6)
    i = np.arange(N, dtype=float)
    if kind == 0:
        f = rng.uniform(1, 20)
        h = np.cos(2 * np.pi * f * i / N + rng.uniform(0, 2*np.pi))
        h = h + rng.uniform(-0.5, 0.5)
    elif kind == 1:
        # two-level
        a = rng.uniform(-2, 2); b = rng.uniform(-2, 2)
        k = rng.integers(1, N)
        h = np.where(np.arange(N) < k, a, b)
    elif kind == 2:
        # chirp
        rate = rng.uniform(0, 50)
        ang = 2 * np.pi * np.cumsum(rate * i / N) + rng.uniform(0, 2*np.pi)
        h = np.cos(ang) + rng.uniform(-0.3, 0.3)
    elif kind == 3:
        # Zadoff-Chu inspired
        u = rng.integers(1, N)
        phase = rng.uniform(0, 2*np.pi)
        ang = phase - np.pi * u * i * (i + 1) / N
        h = np.cos(ang) + rng.uniform(-0.2, 0.2)
    elif kind == 4:
        # random with smoothing
        h = rng.normal(0, 1, N)
        k = max(1, int(N // 50))
        ker = np.ones(k) / k
        h = np.convolve(h, ker, mode='same')
    else:
        # DFT basis with random coefficients
        p = rng.uniform(-1, 1, 10)
        h = _dft_basis_h(p)
    # ensure nonzero sum
    s = np.sum(h)
    if abs(s) < 1e-6:
        h = h + 0.1
    return h

def construct():
    start = time.time()
    best_h = np.ones(N)
    best_v = _value_from_h(best_h)

    rng = np.random.default_rng(12345)

    def deadline_ok():
        return (time.time() - start) < _DEADLINE

    # Phase 1: random structured search
    iters = 0
    while deadline_ok() and iters < 3000:
        iters += 1
        h = _random_h(rng)
        v = _value_from_h(h)
        if v < best_v:
            best_v = v
            best_h = h.copy()

    # Phase 2: Differential evolution on chirp params
    def chirp_obj(p):
        h = _chirp(p)
        return _value_from_h(h)

    bounds = [(-np.pi, np.pi), (-100, 100), (-2, 2), (-2, 2), (-2, 2)]
    try:
        if deadline_ok():
            res = differential_evolution(
                chirp_obj, bounds,
                maxiter=25, popsize=10, seed=7, tol=1e-4,
                polish=False,
            )
            if res.x is not None:
                h = _chirp(res.x)
                v = _value_from_h(h)
                if v < best_v:
                    best_v = v
                    best_h = h.copy()
    except Exception:
        pass

    # Phase 3: Differential evolution on Zadoff-Chu params
    def zadoff_obj(p):
        h = _zadoff_chu_h(p)
        return _value_from_h(h)

    bounds_z = [(-np.pi, np.pi), (1, 400), (-1, 1)]
    try:
        if deadline_ok():
            res_z = differential_evolution(
                zadoff_obj, bounds_z,
                maxiter=30, popsize=12, seed=17, tol=1e-4,
                polish=False,
            )
            if res_z.x is not None:
                h = _zadoff_chu_h(res_z.x)
                v = _value_from_h(h)
                if v < best_v:
                    best_v = v
                    best_h = h.copy()
    except Exception:
        pass

    # Phase 4: L-BFGS-B refinement using smooth-max surrogate on DFT basis
    def dft_obj_smooth(p, beta=50.0):
        h = _dft_basis_h(p)
        return _smooth_max_value(h, beta)

    # Initialize from best_h's DFT projection
    i = np.arange(N, dtype=float)
    dc_init = np.mean(best_h)
    a1 = 2.0/N * np.sum(best_h * np.cos(2*np.pi*1*i/N))
    b1 = 2.0/N * np.sum(best_h * np.sin(2*np.pi*1*i/N))
    a2 = 2.0/N * np.sum(best_h * np.cos(2*np.pi*2*i/N))
    b2 = 2.0/N * np.sum(best_h * np.sin(2*np.pi*2*i/N))
    a3 = 2.0/N * np.sum(best_h * np.cos(2*np.pi*3*i/N))
    b3 = 2.0/N * np.sum(best_h * np.sin(2*np.pi*3*i/N))
    a4 = 2.0/N * np.sum(best_h * np.cos(2*np.pi*4*i/N))
    b4 = 2.0/N * np.sum(best_h * np.sin(2*np.pi*4*i/N))
    p_init = np.array([dc_init, a1, b1, a2, b2, a3, b3, a4, b4])

    bounds_dft = [(-3, 3)] + [(-2, 2)] * 8
    try:
        if deadline_ok():
            for beta in [20.0, 50.0, 100.0]:
                if not deadline_ok():
                    break
                res_dft = minimize(
                    lambda p: _smooth_max_value(_dft_basis_h(p), beta),
                    p_init,
                    method='L-BFGS-B',
                    bounds=bounds_dft,
                    options={'maxiter': 200, 'ftol': 1e-10, 'gtol': 1e-8}
                )
                if res_dft.x is not None:
                    h = _dft_basis_h(res_dft.x)
                    v = _value_from_h(h)
                    if v < best_v:
                        best_v = v
                        best_h = h.copy()
    except Exception:
        pass

    # Phase 5: Local Nelder-Mead on 2-level parameterization
    def two_obj(p):
        h = _two_level(p)
        return _value_from_h(h)

    bounds2 = [(-3, 3), (-3, 3), (0.01, 0.99), (-1, 1)]
    try:
        if deadline_ok():
            res2 = differential_evolution(
                two_obj, bounds2,
                maxiter=25, popsize=10, seed=13, tol=1e-4, polish=False,
            )
            if res2.x is not None:
                h = _two_level(res2.x)
                v = _value_from_h(h)
                if v < best_v:
                    best_v = v
                    best_h = h.copy()
    except Exception:
        pass

    # Phase 6: Final Nelder-Mead polish on chirp params from best
    def chirp_obj_nm(p):
        h = _chirp(p)
        return _value_from_h(h)

    # Try to extract chirp-like params from best_h
    # Just try a few random inits near reasonable values
    try:
        if deadline_ok():
            for seed_try in [42, 99, 123]:
                if not deadline_ok():
                    break
                p0 = np.array([
                    rng.uniform(-np.pi, np.pi),
                    rng.uniform(-50, 50),
                    rng.uniform(-1, 1),
                    rng.uniform(-1, 1),
                    rng.uniform(-0.5, 0.5)
                ])
                res_nm = minimize(
                    chirp_obj_nm, p0,
                    method='Nelder-Mead',
                    options={'maxiter': 500, 'xatol': 1e-6, 'fatol': 1e-8}
                )
                if res_nm.x is not None:
                    h = _chirp(res_nm.x)
                    v = _value_from_h(h)
                    if v < best_v:
                        best_v = v
                        best_h = h.copy()
    except Exception:
        pass

    # Final: ensure validity and return
    s = np.sum(best_h)
    if s == 0 or not np.all(np.isfinite(best_h)):
        return np.ones(N)
    return best_h
# EVOLVE-BLOCK-END
