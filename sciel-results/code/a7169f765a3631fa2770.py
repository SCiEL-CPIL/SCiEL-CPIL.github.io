import time
import numpy as np
from scipy.optimize import minimize

N = 50
w = 0.5 / N  # step width = 0.01
_START = time.time()
DEADLINE = 148.0  # safety margin before 150s kill


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


def _neg_ratio(h):
    """Negative ratio for minimization."""
    return -_ratio(h)


def _neg_ratio_grad(x):
    """Negative ratio and gradient."""
    r, g = _ratio_and_grad(x)
    return -r, -g


def _find_sidon_set(n, rng):
    """
    Find a Sidon-like set of size n from {0, ..., N-1}.
    A Sidon set has all pairwise sums (i+j for i<=j) distinct.
    We use a greedy approach with randomization.
    """
    best_set = None
    best_score = -1

    for attempt in range(50):
        used_sums = set()
        sset = []
        candidates = list(range(N))
        rng.shuffle(candidates)

        for i in candidates:
            if len(sset) >= n:
                break
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

        if len(sset) >= n:
            sset = sset[:n]
            # Score: prefer sets with more spread out elements
            score = np.var(sset)
            if score > best_score:
                best_score = score
                best_set = sset

    if best_set is None:
        # Fallback: evenly spaced
        best_set = list(range(0, N, N // n))[:n]

    return best_set


def _sparse_poly_optimize(exponents, rng, max_iter=2000):
    """
    Optimize the complex amplitudes of a sparse polynomial P(z) = sum a_k * z^{e_k}
    where exponents form a Sidon set. The heights are h = |FFT(a)|^2 normalized.
    """
    K = len(exponents)
    if K < 2:
        return None, 0.0

    # Initialize with random complex amplitudes
    a_init = rng.standard_normal(K) + 1j * rng.standard_normal(K)

    def _h_from_a(a):
        """Compute heights from complex amplitudes with given exponents."""
        # P(z) = sum_k a_k * z^{e_k}
        # h[j] = |sum_k a_k * exp(2*pi*i*j*e_k/N)|^2
        # = |FFT of a placed at exponents|^2
        coeffs = np.zeros(N, dtype=complex)
        for k, e in enumerate(exponents):
            if 0 <= e < N:
                coeffs[e] = a[k]
        H = np.fft.fft(coeffs)
        h = np.abs(H) ** 2
        mx = np.max(h)
        if mx <= 0:
            return np.zeros(N)
        return h / mx

    def _objective(x):
        """x = [a_0_real, a_0_imag, a_1_real, a_1_imag, ...]"""
        a = x[0::2] + 1j * x[1::2]
        h = _h_from_a(a)
        if np.max(h) <= 0:
            return 0.0
        return -_ratio(h)

    x0 = np.concatenate([a_init.real, a_init.imag])

    try:
        res = minimize(_objective, x0, method='L-BFGS-B',
                       options={'maxiter': max_iter, 'ftol': 1e-14,
                                'gtol': 1e-12, 'maxls': 100})
        x_opt = res.x
        a_opt = x_opt[0::2] + 1j * x_opt[1::2]
        h_opt = _h_from_a(a_opt)
        f_opt = _ratio(h_opt)
        return h_opt, f_opt
    except Exception:
        pass

    return None, 0.0


def _sparse_poly_optimize_multi(exponents_list, rng, max_iter=1500):
    """
    Try multiple random initializations for sparse polynomial optimization.
    """
    best_h = None
    best_f = 0.0

    for exponents in exponents_list:
        if _elapsed() > DEADLINE - 10:
            break

        # Try several random initializations
        for trial in range(5):
            if _elapsed() > DEADLINE - 10:
                break
            h, f = _sparse_poly_optimize(exponents, rng, max_iter=max_iter)
            if h is not None and f > best_f:
                best_f = f
                best_h = h.copy()

    return best_h, best_f


def _dense_optimize(h0, max_iter=2000):
    """
    Standard dense optimization using L-BFGS-B with analytic gradient.
    """
    if _elapsed() > DEADLINE - 5:
        return h0, _ratio(h0)

    h0 = np.array(h0, dtype=float)
    mx = np.max(h0)
    if mx <= 0:
        return h0, 0.0
    h0 = h0 / mx

    bounds = [(0.0, None)] * N

    try:
        res = minimize(_neg_ratio, h0, jac=_neg_ratio_grad, method='L-BFGS-B',
                       bounds=bounds,
                       options={'maxiter': max_iter, 'ftol': 1e-14,
                                'gtol': 1e-12, 'maxls': 100})
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


def _coordinate_refine(h, max_rounds=30):
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
            # Adaptive step sizes
            for delta in [0.001, -0.001, 0.005, -0.005, 0.01, -0.01,
                          0.02, -0.02, 0.05, -0.05, 0.1, -0.1,
                          0.2, -0.2]:
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


def _polish(h):
    """High-precision final polish."""
    if _elapsed() > DEADLINE - 3:
        return h, _ratio(h)

    h = np.array(h, dtype=float)
    mx = np.max(h)
    if mx <= 0:
        return h, 0.0
    h = h / mx

    bounds = [(0.0, None)] * N

    try:
        res = minimize(_neg_ratio, h, jac=_neg_ratio_grad, method='L-BFGS-B',
                       bounds=bounds,
                       options={'maxiter': 3000, 'ftol': 1e-16, 'gtol': 1e-14,
                                'maxls': 200})
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


def _initializations(rng):
    """Generate diverse initializations."""
    step_centers = -0.25 + (np.arange(N) + 0.5) * w
    shapes = []

    # Uniform
    shapes.append(np.ones(N))

    # Gaussian with various widths
    for sig in [0.01, 0.02, 0.03, 0.05, 0.07, 0.1, 0.15, 0.2, 0.25, 0.3]:
        s = np.exp(-step_centers**2 / (2 * sig * sig))
        shapes.append(s)

    # Lorentzian
    for sig in [0.01, 0.02, 0.03, 0.05, 0.07, 0.1, 0.15, 0.2]:
        s = 1.0 / (1.0 + (step_centers / sig)**2)
        shapes.append(s)

    # Triangle
    for width in [0.02, 0.05, 0.1, 0.15, 0.2, 0.25]:
        s = np.clip(1.0 - np.abs(step_centers) / width, 0, None)
        if np.max(s) > 0:
            shapes.append(s)

    # Box
    for width in [0.02, 0.05, 0.1, 0.15, 0.2]:
        s = (np.abs(step_centers) < width).astype(float)
        shapes.append(s)

    # Single step
    s = np.zeros(N); s[N // 2] = 1.0
    shapes.append(s)

    # Two symmetric steps
    for d in [1, 2, 3, 4, 5, 7, 10, 15]:
        s = np.zeros(N)
        c = N // 2
        if c - d >= 0 and c + d < N:
            s[c - d] = 1
            s[c + d] = 1
        shapes.append(s)

    # Three symmetric steps
    for d in [2, 3, 5, 8]:
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

    # Cosine bump
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

    # Bump function
    for width in [0.05, 0.08, 0.1, 0.15, 0.2]:
        t = np.abs(step_centers) / width
        s = np.where(t < 1, np.exp(-1.0 / (1.0 - t**2)), 0.0)
        if np.max(s) > 0:
            shapes.append(s)

    # Random smooth shapes
    for _ in range(20):
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

    # One-sided concentrated
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

    # Tapered plateau
    for width in [0.05, 0.1, 0.15, 0.2, 0.25]:
        s = np.where(np.abs(step_centers) < width,
                     0.5 * (1 + np.cos(np.pi * step_centers / width)), 0.0)
        if np.max(s) > 0:
            shapes.append(s / np.max(s))

    # Sidon-set inspired sparse initializations
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

    # Sparse random
    for _ in range(20):
        s = np.zeros(N)
        k = rng.integers(8, 20)
        idxs = rng.choice(N, size=k, replace=False)
        s[idxs] = rng.uniform(0.5, 1.0, size=k)
        kernel = np.array([0.2, 0.6, 0.2])
        s = np.convolve(s, kernel, mode='same')
        s = np.maximum(s, 0)
        if np.max(s) > 0:
            shapes.append(s / np.max(s))

    return shapes


def construct():
    t0 = time.time()
    rng = np.random.default_rng(42)

    best_h = np.ones(N)
    best_f = _ratio(best_h)

    # Phase 1: Sparse polynomial optimization with Sidon sets (structural approach)
    # This is the key structural change: represent heights as |FFT of sparse poly|^2
    if _elapsed() < DEADLINE - 80:
        # Try various Sidon set sizes
        for K in [5, 6, 7, 8, 9, 10, 12]:
            if _elapsed() > DEADLINE - 20:
                break
            # Find multiple Sidon sets of size K
            sidon_exponents_list = []
            for _ in range(5):
                if _elapsed() > DEADLINE - 20:
                    break
                sset = _find_sidon_set(K, rng)
                if sset is not None and len(sset) == K:
                    # Check if we already have a similar set
                    is_new = True
                    for existing in sidon_exponents_list:
                        if set(sset) == set(existing):
                            is_new = False
                            break
                    if is_new:
                        sidon_exponents_list.append(sset)

            if sidon_exponents_list:
                h_sp, f_sp = _sparse_poly_optimize_multi(sidon_exponents_list, rng, max_iter=1000)
                if h_sp is not None and f_sp > best_f:
                    best_f = f_sp
                    best_h = h_sp.copy()

    # Phase 2: Dense optimization from diverse initializations
    starts = _initializations(rng)
    norm_starts = []
    for s in starts:
        s = np.array(s, dtype=float)
        mx = np.max(s)
        if mx > 0:
            norm_starts.append(s / mx)

    # Quick screening
    for idx, s in enumerate(norm_starts):
        if _elapsed() > DEADLINE - 40:
            break
        h, f = _dense_optimize(s, max_iter=500)
        if f > best_f:
            best_f = f
            best_h = h.copy()

    # Phase 3: Refine the best
    if _elapsed() < DEADLINE - 20:
        best_h, best_f = _dense_optimize(best_h, max_iter=2000)

    # Phase 4: Coordinate descent refinement
    if _elapsed() < DEADLINE - 10:
        best_h, best_f = _coordinate_refine(best_h, max_rounds=20)

    # Phase 5: High-precision polish
    if _elapsed() < DEADLINE - 5:
        best_h, best_f = _polish(best_h)

    # Phase 6: A few random restarts from perturbed best
    if _elapsed() < DEADLINE - 15:
        for _ in range(10):
            if _elapsed() > DEADLINE - 5:
                break
            h_pert = best_h + rng.normal(0, 0.05, N)
            h_pert = np.maximum(h_pert, 0)
            mx = np.max(h_pert)
            if mx <= 0:
                continue
            h_pert = h_pert / mx
            h, f = _dense_optimize(h_pert, max_iter=800)
            if f > best_f:
                best_f = f
                best_h = h.copy()

    # Final polish
    if _elapsed() < DEADLINE - 3:
        best_h, best_f = _polish(best_h)

    best_h = np.maximum(best_h, 0.0)
    return [float(v) for v in best_h]
