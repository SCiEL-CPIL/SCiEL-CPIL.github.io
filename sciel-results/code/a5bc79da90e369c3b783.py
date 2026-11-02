import numpy as np
import time
from scipy.optimize import differential_evolution

N = 50
DELTA = 1.0 / N
DEADLINE = 145.0


def _autoconv_sup(h):
    """Sup norm of autoconvolution of the step function with heights h.

    For a step function f(x)=h[i] on [i*DELTA,(i+1)*DELTA), the autoconvolution
    (f*f)(t) is piecewise-linear with breakpoints at multiples of DELTA, so its
    maximum occurs at one of these breakpoints. We evaluate the continuous
    convolution at t = k*DELTA, k=0..2N-2, by zero-padding to length 2N-1 and
    taking the DFT (circular convolution on the padded grid equals the linear
    autoconvolution sampled at the breakpoints).
    """
    h = np.asarray(h, dtype=float)
    n = len(h)
    pad = 2 * n - 1
    a = np.zeros(pad)
    a[:n] = h
    c = np.fft.ifft(np.fft.fft(a) ** 2).real
    return float(np.max(c)) * DELTA


def _ratio(h):
    h = np.maximum(h, 0.0)
    s1 = float(np.sum(h))
    if s1 < 1e-15:
        return 0.0
    s2 = float(np.sum(h * h))
    sup = _autoconv_sup(h)
    if sup < 1e-15:
        return 0.0
    return s2 / (s1 * sup)


def _spectral_start(rng, n_bins=4):
    """Nonnegative starting point = |IFFT of a sparse random spectral mask|."""
    mask = np.zeros(N, dtype=complex)
    idx = rng.choice(N, size=n_bins, replace=False)
    mask[idx] = (rng.uniform(0.5, 2.0, n_bins) +
                 1j * rng.uniform(-1.0, 1.0, n_bins))
    mask[0] += float(np.sum(mask.real)) * 0.3  # push toward nonnegativity
    h = np.abs(np.fft.ifft(mask))
    if h.sum() < 1e-15:
        h = np.ones(N)
    return h / h.sum()


def construct():
    t0 = time.time()
    rng = np.random.default_rng(12345)

    # ---- seed solutions (diverse, cheap) ----
    seeds = []
    # 1) sparse Sidon-like spike patterns (known to score ~1.17)
    def sidon_set(max_size):
        s = [0]
        used = {0}
        for c in range(1, N):
            ok = all((i + c) not in used for i in s)
            if ok:
                s.append(c)
                used.add(c + c)
                for i in s[:-1]:
                    used.add(i + c)
            if len(s) >= max_size:
                break
        return s

    for sz in (8, 10, 12, 14):
        h = np.zeros(N)
        for i in sidon_set(sz):
            h[i] = 1.0
        seeds.append(h / h.sum())
    # 2) sparse random spike patterns
    for _ in range(20):
        k = int(rng.integers(5, 15))
        h = np.zeros(N)
        h[rng.choice(N, size=k, replace=False)] = rng.uniform(0.5, 1.5, k)
        seeds.append(h / h.sum())
    # 3) spectral-mask starts (new representation family)
    for nb in (2, 3, 4, 5, 6, 8):
        for _ in range(3):
            seeds.append(_spectral_start(rng, n_bins=nb))
    # 4) classical shapes
    t = np.linspace(-1, 1, N)
    for base in (np.ones(N),
                 np.maximum(1 - t ** 2, 0.01),
                 np.maximum(t ** 2 + 0.1, 0.01),
                 np.exp(-t ** 2 / 0.3),
                 np.maximum(1 + 2 * np.exp(-((t - 0.6) ** 2) / 0.1) +
                            2 * np.exp(-((t + 0.6) ** 2) / 0.1), 0.05)):
        seeds.append(base / base.sum())
    alt = np.zeros(N); alt[::2] = 1.0
    seeds.append(alt / alt.sum())
    sp3 = np.zeros(N); sp3[::3] = 1.0
    seeds.append(sp3 / sp3.sum())

    best_h = seeds[0].copy()
    best_score = _ratio(best_h)
    for s in seeds[1:]:
        sc = _ratio(s)
        if sc > best_score:
            best_score = sc
            best_h = s.copy()

    # ---- global search: differential evolution (handles non-smooth max) ----
    def neg_ratio(x):
        x = np.maximum(x, 0.0)
        s = float(np.sum(x))
        if s < 1e-15:
            return 0.0
        return -_ratio(x / s)

    def elapsed_ok(margin):
        return (time.time() - t0) < DEADLINE - margin

    bounds = [(0.0, 1.0)] * N
    n_iter = 90
    pop = 40
    if elapsed_ok(20):
        try:
            res = differential_evolution(
                neg_ratio, bounds, seed=7, maxiter=n_iter, popsize=pop,
                tol=1e-9, init='sobol', polish=True,
                updating='immediate', mutation=(0.4, 1.0), recombination=0.7)
            h = np.maximum(res.x, 0.0)
            if h.sum() > 1e-15:
                h = h / h.sum()
                sc = _ratio(h)
                if sc > best_score:
                    best_score, best_h = sc, h.copy()
        except Exception:
            pass

    # ---- second DE round from a fresh spectral population, if time remains ----
    if elapsed_ok(20):
        try:
            def neg_ratio2(x):
                x = np.maximum(x, 0.0)
                s = float(np.sum(x))
                if s < 1e-15:
                    return 0.0
                return -_ratio(x / s)
            res2 = differential_evolution(
                neg_ratio2, bounds, seed=99, maxiter=60, popsize=30,
                tol=1e-9, init='sobol', polish=True, updating='immediate',
                mutation=(0.5, 1.0), recombination=0.8)
            h = np.maximum(res2.x, 0.0)
            if h.sum() > 1e-15:
                h = h / h.sum()
                sc = _ratio(h)
                if sc > best_score:
                    best_score, best_h = sc, h.copy()
        except Exception:
            pass

    # ---- local refinement around the best solution ----
    if elapsed_ok(15):
        try:
            from scipy.optimize import minimize

            def neg_ratio3(x):
                x = np.maximum(x, 0.0)
                s = float(np.sum(x))
                if s < 1e-15:
                    return 0.0
                return -_ratio(x / s)

            res3 = minimize(neg_ratio3, best_h.copy(), method='Nelder-Mead',
                            options={'maxiter': 3000, 'xatol': 1e-10,
                                     'fatol': 1e-12, 'adaptive': True})
            h = np.maximum(res3.x, 0.0)
            if h.sum() > 1e-15:
                h = h / h.sum()
                sc = _ratio(h)
                if sc > best_score:
                    best_score, best_h = sc, h.copy()
        except Exception:
            pass

    # ---- coordinate-descent polish on individual heights ----
    if elapsed_ok(10):
        try:
            cur = best_h.copy()
            improved = True
            while improved and elapsed_ok(10):
                improved = False
                cur_base = _ratio(cur)
                for i in range(N):
                    for step in (0.5, 1.0, 2.0, 4.0):
                        for sign in (1.0, -1.0):
                            trial = cur.copy()
                            trial[i] = max(0.0, cur[i] + sign * step * cur[i] + sign * 0.05)
                            sc = _ratio(trial)
                            if sc > cur_base + 1e-12:
                                cur = trial
                                cur_base = sc
                                improved = True
                                if sc > best_score:
                                    best_score, best_h = sc, cur.copy()
        except Exception:
            pass

    return best_h.tolist()
