import time
import numpy as np
from scipy.optimize import minimize

N = 50
w = 0.5 / N  # step width = 0.01
_START = time.time()
DEADLINE = 145.0  # safety margin before 150s kill


def _elapsed():
    return time.time() - _START


def _autoconv_lags(h):
    """Compute all lag values of autoconvolution (without the w factor).
    c[m] = sum_j h[j]*h[j+m] for m=0..N-1
    """
    c = np.zeros(N)
    c[0] = np.dot(h, h)
    for m in range(1, N):
        c[m] = np.dot(h[:N - m], h[m:])
    return c


def _ratio(h):
    """Compute ||h||_2^2 / (||h||_1 * ||h*h||_inf)."""
    h = np.maximum(h, 0.0)
    s2 = np.dot(h, h)
    s1 = np.sum(h)
    if s1 <= 0 or s2 <= 0:
        return 0.0
    c = _autoconv_lags(h)
    acmax = w * c.max()
    if acmax <= 0:
        return 0.0
    # ||h||_2^2 = w * s2, ||h||_1 = w * s1, ||h*h||_inf = acmax
    # ratio = (w*s2) / (w*s1 * acmax) = s2 / (s1 * acmax)
    return s2 / (s1 * acmax)


def _ratio_and_grad(h):
    """Compute ratio and analytic gradient (with respect to h, assuming h >= 0)."""
    h = np.maximum(h, 0.0)
    s2 = np.dot(h, h)
    s1 = np.sum(h)
    if s1 <= 0 or s2 <= 0:
        return 0.0, np.zeros_like(h)

    c = _autoconv_lags(h)
    m_max = int(np.argmax(c))
    cmax = c[m_max]
    if cmax <= 0:
        return 0.0, np.zeros_like(h)

    acmax = w * cmax
    ratio = s2 / (s1 * acmax)

    dF = 2.0 * h  # d(s2)/dh_i

    dcmax = np.zeros_like(h)
    if m_max == 0:
        dcmax[:] = 2.0 * h
    else:
        dcmax[:N - m_max] += h[m_max:]
        dcmax[m_max:] += h[:N - m_max]

    dG = cmax + s1 * dcmax
    G = s1 * cmax
    grad = (dF * G - s2 * dG) / (w * G * G)

    return ratio, grad


def _objective_neg_scalar(x):
    """Scalar negative ratio."""
    x = np.maximum(x, 0.0)
    mx = np.max(x)
    if mx <= 0:
        return 0.0
    h = x / mx
    r, _ = _ratio_and_grad(h)
    return -r


def _objective_neg(x):
    """Negative ratio for optimizers."""
    x = np.maximum(x, 0.0)
    mx = np.max(x)
    if mx <= 0:
        return 0.0
    h = x / mx
    return -_ratio(h)


def _initializations():
    """Generate diverse parametric initializations."""
    step_centers = -0.25 + (np.arange(N) + 0.5) * w
    shapes = []

    # Gaussian with various widths
    for sig in [0.005, 0.01, 0.015, 0.02, 0.03, 0.04, 0.05, 0.07, 0.1, 0.15, 0.2, 0.25, 0.3]:
        s = np.exp(-step_centers**2 / (2 * sig * sig))
        shapes.append(s)

    # Lorentzian / Cauchy
    for sig in [0.01, 0.02, 0.03, 0.05, 0.07, 0.1, 0.15, 0.2]:
        s = 1.0 / (1.0 + (step_centers / sig)**2)
        shapes.append(s)

    # Triangle
    for width in [0.01, 0.02, 0.03, 0.05, 0.07, 0.1, 0.15, 0.2, 0.25]:
        s = np.clip(1.0 - np.abs(step_centers) / width, 0, None)
        if np.max(s) > 0:
            shapes.append(s)

    # Box / plateau
    for width in [0.01, 0.02, 0.03, 0.05, 0.07, 0.1, 0.15]:
        s = (np.abs(step_centers) < width).astype(float)
        shapes.append(s)

    # Single step
    s = np.zeros(N); s[N // 2] = 1.0
    shapes.append(s)

    # Two symmetric steps
    for d in [1, 2, 3, 4, 5, 7, 10]:
        s = np.zeros(N)
        c = N // 2
        if c - d >= 0 and c + d < N:
            s[c - d] = 1
            s[c + d] = 1
        shapes.append(s)

    # Three symmetric steps
    for d in [2, 3, 5]:
        s = np.zeros(N)
        c = N // 2
        if c - d >= 0 and c + d < N:
            s[c - d] = 1
            s[c] = 1
            s[c + d] = 1
        shapes.append(s)

    # Power-law decay
    for p in [1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0]:
        s = np.exp(-p * np.abs(step_centers))
        shapes.append(s)

    # Sine bump
    for freq in [1, 2, 3, 4, 5, 6, 8]:
        s = np.sin(freq * np.pi * (step_centers + 0.25) / 0.5)
        s = np.maximum(s, 0)
        if np.max(s) > 0:
            shapes.append(s)

    # Cosine bump (centered)
    for freq in [1, 2, 3, 4]:
        s = np.cos(freq * np.pi * step_centers / 0.25)
        s = np.maximum(s, 0)
        if np.max(s) > 0:
            shapes.append(s)

    # Smoothstep
    for width in [0.05, 0.1, 0.15, 0.2, 0.25]:
        t = np.clip((width - np.abs(step_centers)) / (2 * width), 0, 1)
        s = t * t * (3 - 2 * t)
        if np.max(s) > 0:
            shapes.append(s)

    # Bump function (compact support, smooth)
    for width in [0.05, 0.08, 0.1, 0.15, 0.2]:
        t = np.abs(step_centers) / width
        s = np.where(t < 1, np.exp(-1.0 / (1.0 - t**2)), 0.0)
        if np.max(s) > 0:
            shapes.append(s)

    # Polynomial (Chebyshev-based)
    t = 2 * step_centers  # map to [-0.5, 0.5]
    for deg in [2, 3, 4, 5, 6, 8, 10, 12]:
        c = np.polynomial.chebyshev.chebval(t, [0] * deg + [1])
        c = np.maximum(c, 0)
        if np.max(c) > 0:
            shapes.append(c / np.max(c))

    # Random smooth shapes
    rng = np.random.default_rng(42)
    for _ in range(15):
        freqs = rng.integers(1, 10, size=10)
        amps = rng.random(10)
        phases = rng.random(10) * 2 * np.pi
        s = np.zeros(N)
        for f, a, ph in zip(freqs, amps, phases):
            s += a * np.cos(f * 2 * np.pi * step_centers + ph)
        s = np.maximum(s, 0)
        if np.max(s) > 0:
            shapes.append(s / np.max(s))

    # Random with smoothing
    for _ in range(20):
        base = rng.random(N)
        kernel = np.exp(-np.arange(-8, 9)**2 / 8.0)
        kernel /= kernel.sum()
        base = np.convolve(base, kernel, mode='same')
        base = np.maximum(base, 0)
        if np.max(base) > 0:
            shapes.append(base / np.max(base))

    # Pure random
    for _ in range(30):
        shapes.append(rng.random(N))

    # One-sided concentrated shapes
    for alpha in [0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 8.0, 10.0]:
        s = np.exp(-alpha * (step_centers + 0.25))
        shapes.append(s)
        s2 = np.exp(-alpha * (0.25 - step_centers))
        shapes.append(s2)

    # Power-law one-sided
    for p in [1.0, 2.0, 3.0, 5.0, 8.0]:
        s = np.maximum(0.25 - step_centers, 0.01) ** (-p)
        s = s / np.max(s)
        shapes.append(s)
        s2 = np.maximum(step_centers + 0.25, 0.01) ** (-p)
        s2 = s2 / np.max(s2)
        shapes.append(s2)

    # Tapered plateau (raised cosine window)
    for width in [0.05, 0.1, 0.15, 0.2, 0.25]:
        s = np.where(np.abs(step_centers) < width,
                     0.5 * (1 + np.cos(np.pi * step_centers / width)), 0.0)
        if np.max(s) > 0:
            shapes.append(s / np.max(s))

    # Sidon-set inspired sparse initializations
    # Choose subsets of indices that form "Sidon-like" sets (distinct pairwise sums)
    # This reduces autocorrelation peak relative to L2 norm
    sidon_sets = [
        [0, 1, 4, 10, 12, 17, 23, 25, 28, 31, 35, 40, 44, 47],
        [2, 3, 6, 11, 13, 18, 24, 26, 29, 32, 36, 41, 45, 48],
        [0, 2, 5, 11, 13, 18, 24, 26, 29, 34, 38, 42, 46, 49],
        [1, 4, 8, 13, 15, 20, 26, 28, 31, 36, 40, 43, 47, 49],
        [0, 3, 7, 12, 14, 19, 25, 27, 30, 35, 39, 42, 46, 49],
    ]
    for sset in sidon_sets:
        s = np.zeros(N)
        for idx in sset:
            if 0 <= idx < N:
                s[idx] = 1.0
        shapes.append(s)
        # With varying amplitudes
        for amp_scale in [0.5, 1.0, 2.0]:
            s_var = np.zeros(N)
            for i, idx in enumerate(sset):
                if 0 <= idx < N:
                    s_var[idx] = amp_scale * (1 + 0.3 * np.sin(i * 1.7))
            if np.max(s_var) > 0:
                shapes.append(s_var / np.max(s_var))

    # Sparse random with smoothing
    for _ in range(15):
        s = np.zeros(N)
        k = rng.integers(8, 20)
        idxs = rng.choice(N, size=k, replace=False)
        s[idxs] = rng.uniform(0.5, 1.0, size=k)
        # Light smoothing
        kernel = np.array([0.2, 0.6, 0.2])
        s = np.convolve(s, kernel, mode='same')
        s = np.maximum(s, 0)
        if np.max(s) > 0:
            shapes.append(s / np.max(s))

    return shapes


def _local_optimize(h0, max_iter=2000, tol=1e-12):
    """Run trust-region reflective optimization from h0."""
    if _elapsed() > DEADLINE - 5:
        return h0, _ratio(h0)

    h0 = np.array(h0, dtype=float)
    mx = np.max(h0)
    if mx <= 0:
        return h0, 0.0
    h0 = h0 / mx

    bounds = [(0.0, None)] * N

    try:
        res = minimize(_objective_neg_scalar, h0, method='trust-constr',
                       bounds=bounds,
                       options={'maxiter': max_iter, 'gtol': tol,
                                'xtol': 1e-14, 'barrier_tol': 1e-12,
                                'verbose': 0})
        x = np.maximum(res.x, 0.0)
        mx2 = np.max(x)
        if mx2 <= 0:
            return h0, _ratio(h0)
        h2 = x / mx2
        f2 = _ratio(h2)
        f1 = _ratio(h0)
        if f2 > f1:
            return h2, f2
        return h0, f1
    except Exception:
        pass

    # Fallback: L-BFGS-B
    try:
        res = minimize(_objective_neg_scalar, h0, method='L-BFGS-B',
                       bounds=bounds,
                       options={'maxiter': max_iter, 'ftol': tol, 'gtol': 1e-10})
        x = np.maximum(res.x, 0.0)
        mx2 = np.max(x)
        if mx2 <= 0:
            return h0, _ratio(h0)
        h2 = x / mx2
        f2 = _ratio(h2)
        f1 = _ratio(h0)
        if f2 > f1:
            return h2, f2
        return h0, f1
    except Exception:
        pass

    return h0, _ratio(h0)


def _polish(h):
    """High-precision final polish with tighter tolerances."""
    if _elapsed() > DEADLINE - 3:
        return h, _ratio(h)

    h = np.array(h, dtype=float)
    mx = np.max(h)
    if mx <= 0:
        return h, 0.0
    h = h / mx

    bounds = [(0.0, None)] * N

    # Try SLSQP with tight tolerance
    try:
        res = minimize(_objective_neg_scalar, h, method='SLSQP',
                       bounds=bounds,
                       options={'maxiter': 1000, 'ftol': 1e-15, 'eps': 1e-10})
        x = np.maximum(res.x, 0.0)
        mx2 = np.max(x)
        if mx2 <= 0:
            return h, _ratio(h)
        h2 = x / mx2
        f2 = _ratio(h2)
        f1 = _ratio(h)
        if f2 > f1:
            return h2, f2
    except Exception:
        pass

    # Try L-BFGS-B with very tight tolerance
    try:
        res = minimize(_objective_neg_scalar, h, method='L-BFGS-B',
                       bounds=bounds,
                       options={'maxiter': 2000, 'ftol': 1e-16, 'gtol': 1e-14, 'maxls': 500})
        x = np.maximum(res.x, 0.0)
        mx2 = np.max(x)
        if mx2 <= 0:
            return h, _ratio(h)
        h2 = x / mx2
        f2 = _ratio(h2)
        f1 = _ratio(h)
        if f2 > f1:
            return h2, f2
    except Exception:
        pass

    return h, _ratio(h)


def _coordinate_refine(h, max_rounds=20):
    """Coordinate descent refinement."""
    h = np.array(h, dtype=float)
    best_f = _ratio(h)

    for rnd in range(max_rounds):
        if _elapsed() > DEADLINE - 8:
            break
        improved = False
        for i in range(N):
            if _elapsed() > DEADLINE - 8:
                break
            orig = h[i]
            for delta in [0.001, -0.001, 0.005, -0.005, 0.01, -0.01,
                          0.02, -0.02, 0.05, -0.05, 0.1, -0.1]:
                h[i] = max(0.0, orig + delta)
                f = _ratio(h)
                if f > best_f:
                    best_f = f
                    improved = True
                    orig = h[i]
                else:
                    h[i] = orig
        if not improved:
            break

    return h, best_f


def _spectral_optimize(h0, max_iter=500):
    """
    Structural change: parameterize heights via DFT magnitude squared.
    h_i = |DFT(z)_i|^2 where z is a complex vector.
    This ensures non-negativity and smooth manifold.
    We optimize over the complex parameters z (2N real parameters).
    """
    if _elapsed() > DEADLINE - 5:
        return h0, _ratio(h0)

    h0 = np.array(h0, dtype=float)
    mx = np.max(h0)
    if mx <= 0:
        return h0, 0.0
    h0 = h0 / mx

    sqrt_h = np.sqrt(h0)
    z_init = np.fft.ifft(sqrt_h)

    def _h_from_z(z):
        H = np.fft.fft(z)
        h = np.abs(H)**2
        mx = np.max(h)
        if mx <= 0:
            return np.zeros(N)
        return h / mx

    def _objective_z(z):
        h = _h_from_z(z)
        if np.max(h) <= 0:
            return 0.0
        r = _ratio(h)
        return -r

    z0_real = np.concatenate([z_init.real, z_init.imag])

    def _obj_flat(x):
        z = x[:N] + 1j * x[N:]
        return _objective_z(z)

    try:
        res = minimize(_obj_flat, z0_real, method='L-BFGS-B',
                       options={'maxiter': max_iter, 'ftol': 1e-14, 'gtol': 1e-10, 'maxls': 100})
        x = res.x
        z = x[:N] + 1j * x[N:]
        h2 = _h_from_z(z)
        f2 = _ratio(h2)
        f1 = _ratio(h0)
        if f2 > f1:
            return h2, f2
    except Exception:
        pass

    return h0, _ratio(h0)


def _genetic_search(h_best, n_pop=20, n_gen=10):
    """
    Genetic algorithm with mutation and crossover.
    """
    if _elapsed() > DEADLINE - 10:
        return h_best, _ratio(h_best)

    rng = np.random.default_rng(12345)
    h_best = np.array(h_best, dtype=float)
    f_best = _ratio(h_best)

    pop = [h_best.copy()]
    for _ in range(n_pop - 1):
        if rng.random() < 0.5:
            s = rng.random(N)
        else:
            s = h_best + rng.normal(0, 0.1, N)
            s = np.maximum(s, 0)
        mx = np.max(s)
        if mx > 0:
            pop.append(s / mx)
        else:
            pop.append(rng.random(N))

    for gen in range(n_gen):
        if _elapsed() > DEADLINE - 10:
            break

        fitness = [_ratio(p) for p in pop]

        idx = np.argsort(fitness)[::-1]
        top = [pop[i].copy() for i in idx[:n_pop // 2]]

        new_pop = [top[0].copy()]
        while len(new_pop) < n_pop:
            if _elapsed() > DEADLINE - 10:
                break
            a = top[rng.integers(0, len(top))]
            b = top[rng.integers(0, len(top))]
            child = 0.5 * (a + b)
            if rng.random() < 0.3:
                child = child + rng.normal(0, 0.05, N)
            child = np.maximum(child, 0)
            mx = np.max(child)
            if mx > 0:
                child = child / mx
            new_pop.append(child)

        pop = new_pop

    for p in pop:
        f = _ratio(p)
        if f > f_best:
            f_best = f
            h_best = p.copy()

    return h_best, f_best


def _sparse_optimize(h0, max_iter=1000):
    """
    Structural change: optimize a sparse representation.
    Select a subset of indices to be nonzero, then optimize their amplitudes.
    This directly exploits the Sidon-set insight: sparse structures minimize
    autocorrelation overlap by ensuring distinct pairwise sums.
    """
    if _elapsed() > DEADLINE - 5:
        return h0, _ratio(h0)

    h0 = np.array(h0, dtype=float)
    mx = np.max(h0)
    if mx <= 0:
        return h0, 0.0
    h0 = h0 / mx

    # Determine which indices to keep based on current h0
    # Keep top-k indices by amplitude
    for k in [10, 12, 14, 16, 18]:
        if _elapsed() > DEADLINE - 8:
            break
        idx = np.argsort(h0)[::-1][:k]
        mask = np.zeros(N, dtype=bool)
        mask[idx] = True

        # Initialize amplitudes from h0
        h_sparse = np.zeros(N)
        h_sparse[idx] = h0[idx]

        def _obj_sparse(x):
            h = np.zeros(N)
            h[idx] = np.maximum(x, 0.0)
            mx = np.max(h)
            if mx <= 0:
                return 0.0
            h = h / mx
            return -_ratio(h)

        try:
            res = minimize(_obj_sparse, h_sparse[idx].copy(), method='L-BFGS-B',
                           options={'maxiter': max_iter, 'ftol': 1e-12, 'gtol': 1e-10, 'maxls': 100})
            x_opt = np.maximum(res.x, 0.0)
            h_opt = np.zeros(N)
            h_opt[idx] = x_opt
            mx2 = np.max(h_opt)
            if mx2 > 0:
                h_opt = h_opt / mx2
                f_opt = _ratio(h_opt)
                f_cur = _ratio(h0)
                if f_opt > f_cur:
                    return h_opt, f_opt
        except Exception:
            pass

    return h0, _ratio(h0)


def _sidon_construct():
    """
    Construct Sidon-like sets directly and optimize their amplitudes.
    A Sidon set has all pairwise sums distinct, minimizing autocorrelation peaks.
    """
    if _elapsed() > DEADLINE - 5:
        return None, 0.0

    # Generate candidate Sidon-like sets using greedy construction
    # We want subsets of {0,...,49} where all pairwise sums (i+j for i<=j) are distinct
    best_h = None
    best_f = 0.0

    # Precomputed good sparse sets
    candidate_sets = [
        [0, 1, 4, 10, 12, 17, 23, 25, 28, 31, 35, 40, 44, 47],
        [2, 3, 6, 11, 13, 18, 24, 26, 29, 32, 36, 41, 45, 48],
        [0, 2, 5, 11, 13, 18, 24, 26, 29, 34, 38, 42, 46, 49],
        [1, 4, 8, 13, 15, 20, 26, 28, 31, 36, 40, 43, 47, 49],
        [0, 3, 7, 12, 14, 19, 25, 27, 30, 35, 39, 42, 46, 49],
        [0, 1, 5, 11, 14, 19, 25, 27, 30, 35, 41, 43, 46, 49],
        [2, 5, 9, 14, 16, 21, 27, 29, 32, 37, 41, 44, 47, 49],
        [0, 4, 8, 13, 15, 20, 26, 28, 33, 37, 41, 44, 47, 49],
    ]

    # Also generate some via greedy algorithm
    rng = np.random.default_rng(777)
    for _ in range(10):
        used_sums = set()
        sset = []
        for i in range(N):
            if len(sset) >= 14:
                break
            # Check if adding i creates duplicate sums
            valid = True
            for j in sset:
                s = i + j
                if s in used_sums:
                    valid = False
                    break
            if valid:
                for j in sset:
                    used_sums.add(i + j)
                used_sums.add(2 * i)
                sset.append(i)
        if len(sset) >= 10:
            candidate_sets.append(sset)

    for sset in candidate_sets:
        if _elapsed() > DEADLINE - 8:
            break
        k = len(sset)
        h_init = np.zeros(N)
        h_init[sset] = 1.0

        def _obj(x):
            h = np.zeros(N)
            h[sset] = np.maximum(x, 0.0)
            mx = np.max(h)
            if mx <= 0:
                return 0.0
            h = h / mx
            return -_ratio(h)

        try:
            res = minimize(_obj, np.ones(k), method='L-BFGS-B',
                           options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-10, 'maxls': 50})
            x_opt = np.maximum(res.x, 0.0)
            h_opt = np.zeros(N)
            h_opt[sset] = x_opt
            mx = np.max(h_opt)
            if mx > 0:
                h_opt = h_opt / mx
                f_opt = _ratio(h_opt)
                if f_opt > best_f:
                    best_f = f_opt
                    best_h = h_opt.copy()
        except Exception:
            pass

    if best_h is None:
        return None, 0.0
    return best_h, best_f


def construct():
    t0 = time.time()
    best_h = np.ones(N)
    best_f = _ratio(best_h)

    starts = _initializations()

    # Normalize all starts
    norm_starts = []
    for s in starts:
        s = np.array(s, dtype=float)
        mx = np.max(s)
        if mx > 0:
            norm_starts.append(s / mx)

    # Phase 0: Sidon-set construction (structural)
    if _elapsed() < DEADLINE - 60:
        h_sidon, f_sidon = _sidon_construct()
        if h_sidon is not None and f_sidon > best_f:
            best_f = f_sidon
            best_h = h_sidon.copy()

    # Phase 1: Quick screening with fewer iterations
    for idx, s in enumerate(norm_starts):
        if _elapsed() > DEADLINE - 40:
            break
        h, f = _local_optimize(s, max_iter=500, tol=1e-10)
        if f > best_f:
            best_f = f
            best_h = h.copy()

    # Phase 2: Refine the best with more iterations
    if _elapsed() < DEADLINE - 20:
        best_h, best_f = _local_optimize(best_h, max_iter=2000, tol=1e-13)

    # Phase 3: Spectral optimization (structural change)
    if _elapsed() < DEADLINE - 15:
        h_spec, f_spec = _spectral_optimize(best_h, max_iter=500)
        if f_spec > best_f:
            best_f = f_spec
            best_h = h_spec.copy()

    # Phase 3b: Sparse optimization (structural change)
    if _elapsed() < DEADLINE - 15:
        h_sp, f_sp = _sparse_optimize(best_h, max_iter=500)
        if f_sp > best_f:
            best_f = f_sp
            best_h = h_sp.copy()

    # Phase 4: Genetic search (structural change)
    if _elapsed() < DEADLINE - 12:
        h_gen, f_gen = _genetic_search(best_h, n_pop=15, n_gen=8)
        if f_gen > best_f:
            best_f = f_gen
            best_h = h_gen.copy()

    # Phase 5: Coordinate descent refinement
    if _elapsed() < DEADLINE - 8:
        best_h, best_f = _coordinate_refine(best_h, max_rounds=15)

    # Phase 6: High-precision polish
    if _elapsed() < DEADLINE - 4:
        best_h, best_f = _polish(best_h)

    # Phase 7: A few random restarts from perturbed best
    if _elapsed() < DEADLINE - 10:
        rng = np.random.default_rng(999)
        for _ in range(8):
            if _elapsed() > DEADLINE - 6:
                break
            h_pert = best_h + rng.normal(0, 0.05, N)
            h_pert = np.maximum(h_pert, 0)
            mx = np.max(h_pert)
            if mx <= 0:
                continue
            h_pert = h_pert / mx
            h, f = _local_optimize(h_pert, max_iter=800, tol=1e-12)
            if f > best_f:
                best_f = f
                best_h = h.copy()

    # Final polish if time remains
    if _elapsed() < DEADLINE - 3:
        best_h, best_f = _polish(best_h)

    best_h = np.maximum(best_h, 0.0)
    return [float(v) for v in best_h]
