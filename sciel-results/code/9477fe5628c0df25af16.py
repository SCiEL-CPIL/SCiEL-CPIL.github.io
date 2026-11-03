"""Combined: spectral subspace projection + FFT local search + epigraph polish for signed step heights."""

import numpy as np
import time
from scipy.optimize import minimize

N = 400
_FSIZE = 1
while _FSIZE < 2 * N - 1:
    _FSIZE *= 2

# EVOLVE-BLOCK-START

def _autoconv(h):
    """Full linear autoconvolution via FFT."""
    H = np.fft.rfft(h, _FSIZE)
    ac = np.fft.irfft(H * H, _FSIZE)
    return ac[:2 * N - 1]

def _raw_value(h):
    """max|autoconv| / (sum h)^2."""
    s = np.sum(h)
    if abs(s) < 1e-12:
        return 1e10
    ac = _autoconv(h)
    return float(np.max(np.abs(ac))) / (s * s)

def _soft_max_abs(ac, beta=50.0):
    """Smooth approximation of max|ac| via log-sum-exp. Returns (value, weights)."""
    a = np.abs(ac)
    m = np.max(a)
    if m < 1e-15:
        return 0.0, None
    scaled = beta * (a - m)
    scaled = np.clip(scaled, -60, 60)
    e = np.exp(scaled)
    log_sum = np.log(np.sum(e))
    weights = e / np.sum(e)
    return m + log_sum / beta, weights

def _smooth_objective(h, beta=50.0):
    """Smooth objective: soft_max_abs(ac) / (sum h)^2."""
    s = np.sum(h)
    if abs(s) < 1e-12:
        return 1e10
    ac = _autoconv(h)
    sm, _ = _soft_max_abs(ac, beta)
    return sm / (s * s)

def _smooth_gradient(h, beta=50.0):
    """
    Gradient of soft_max_abs(ac) / (sum h)^2 with respect to h.
    Uses FFT-based cross-correlation for the autoconvolution gradient.
    """
    s = np.sum(h)
    if abs(s) < 1e-12:
        return np.zeros_like(h)

    ac = _autoconv(h)
    sm, weights = _soft_max_abs(ac, beta)

    if weights is None:
        return np.zeros_like(h)

    # g_ac[m] = weights[m] * sign(ac[m])
    g_ac = weights * np.sign(ac)

    # Gradient of S = soft_max_abs(ac) w.r.t. h:
    # dS/dh[j] = sum_m g_ac[m] * (h[m-j] + h[j-m])
    # By symmetry of autoconv and g_ac, this equals 2 * cross_corr(g_ac, h)[j]
    # where cross_corr(a, b)[j] = sum_m a[m+j] * b[m]

    # Compute cross-correlation via FFT
    fsize_corr = 1
    while fsize_corr < 3 * N:
        fsize_corr *= 2

    V = np.fft.rfft(g_ac, fsize_corr)
    H = np.fft.rfft(h, fsize_corr)
    C = np.fft.irfft(V * np.conj(H), fsize_corr)

    grad_S = 2.0 * C[:N]

    # f = S / s^2
    # df/dh[j] = (grad_S[j] * s - 2 * S) / s^3
    df = (grad_S * s - 2.0 * sm) / (s ** 3)

    return df

def _lbfgs_optimize(h, beta=50.0, max_iter=300, deadline=None):
    """L-BFGS optimization of the smooth surrogate."""
    h0 = h.copy()

    def fun(x):
        if deadline is not None and time.time() > deadline:
            return 1e10
        return _smooth_objective(x, beta)

    def jac(x):
        if deadline is not None and time.time() > deadline:
            return np.zeros_like(x)
        return _smooth_gradient(x, beta)

    try:
        res = minimize(fun, h0, method='L-BFGS-B', jac=jac,
                       options={'maxiter': max_iter, 'gtol': 1e-7, 'maxfun': 1000})
        return res.x, res.fun
    except Exception:
        return h0, _smooth_objective(h0, beta)

def _block_sign_flip_local_search(h, block_sizes=None, max_passes=5, deadline=None):
    """
    Local search: try flipping signs of blocks of consecutive elements.
    O(N log N) per evaluation via FFT.
    """
    if block_sizes is None:
        block_sizes = [1, 2, 4, 8, 16, 32, 64, 128]

    h = h.copy()
    best_val = _raw_value(h)

    for _ in range(max_passes):
        if deadline is not None and time.time() > deadline:
            break

        improved = False
        for bs in block_sizes:
            if deadline is not None and time.time() > deadline:
                break

            for start in range(0, N, bs):
                if deadline is not None and time.time() > deadline:
                    break

                end = min(start + bs, N)
                h[start:end] *= -1
                new_val = _raw_value(h)
                if new_val < best_val:
                    best_val = new_val
                    improved = True
                else:
                    h[start:end] *= -1  # revert

        if not improved:
            break

    return h, best_val

def _single_flip_local_search(h, max_passes=10, deadline=None):
    """
    Local search: try flipping each individual element's sign.
    Uses incremental autoconvolution update for speed (O(N) per flip).
    """
    h = h.copy()
    s = np.sum(h)
    ac = _autoconv(h)
    best_val = float(np.max(np.abs(ac))) / (s * s)

    for _ in range(max_passes):
        if deadline is not None and time.time() > deadline:
            break

        improved = False
        for i in range(N):
            if deadline is not None and time.time() > deadline:
                break

            hi = h[i]
            s_new = s - 2.0 * hi
            if abs(s_new) < 1e-12:
                continue

            # delta_ac[k] = -2*hi*(h[k-i] + h[i-k]) for valid indices
            delta_ac = np.zeros(2 * N - 1)
            k1_start = i
            k1_end = min(i + N, 2 * N - 1)
            if k1_start < k1_end:
                delta_ac[k1_start:k1_end] -= 2.0 * hi * h[0:k1_end - k1_start]
            k2_start = max(0, i - N + 1)
            k2_end = i + 1
            if k2_start < k2_end:
                idx = i - np.arange(k2_start, k2_end)
                delta_ac[k2_start:k2_end] -= 2.0 * hi * h[idx]

            ac_new = ac + delta_ac
            new_val = float(np.max(np.abs(ac_new))) / (s_new * s_new)

            if new_val < best_val:
                h[i] = -hi
                s = s_new
                ac = ac_new
                best_val = new_val
                improved = True

        if not improved:
            break

    return h, best_val

def _spectral_subspace_optimize(h_init, dim=8, deadline=None):
    """
    Project h onto a low-dimensional subspace, then optimize coefficients.
    """
    if deadline is not None and time.time() > deadline:
        return h_init, _raw_value(h_init)

    rng = np.random.default_rng(12345)

    basis = np.zeros((dim, N))
    for d in range(dim):
        freq = (d + 1) * np.pi / N
        basis[d] = np.cos(freq * np.arange(N))

    Q, _ = np.linalg.qr(basis.T)  # Q is N x dim
    Q = Q.T  # dim x N

    coeffs = Q @ h_init  # dim

    def obj(coeffs):
        if deadline is not None and time.time() > deadline:
            return 1e10
        h = coeffs @ Q
        s = np.sum(h)
        if abs(s) < 1e-12:
            return 1e10
        ac = _autoconv(h)
        return float(np.max(np.abs(ac))) / (s * s)

    def grad_coeffs(coeffs):
        if deadline is not None and time.time() > deadline:
            return np.zeros_like(coeffs)
        h = coeffs @ Q
        g = _smooth_gradient(h, beta=50.0)
        return Q @ g

    try:
        res = minimize(obj, coeffs, method='L-BFGS-B', jac=grad_coeffs,
                       options={'maxiter': 200, 'gtol': 1e-6, 'maxfun': 500})
        best_coeffs = res.x
    except Exception:
        best_coeffs = coeffs

    h_opt = best_coeffs @ Q
    val = _raw_value(h_opt)

    if val < _raw_value(h_init):
        return h_opt, val
    else:
        return h_init, _raw_value(h_init)

def _epigraph_polish(h, deadline=None, max_iters=80, mu0=1e-3, mu_min=1e-6):
    """
    High-precision polish via epigraph formulation with auxiliary variable t.

    Minimize t subject to |ac[k]| <= t * s^2  for all k.
    Use a smooth log-barrier: f(h, t) = t + (1/mu) * sum_k log(t*s^2 - |ac[k]| + eps).
    Optimize jointly over h and t with L-BFGS.
    """
    if deadline is not None and time.time() > deadline:
        return h, _raw_value(h)

    h0 = h.copy()
    best_val = _raw_value(h0)
    best_h = h0.copy()

    def _epi_fun(x, mu):
        hh = x[:N]
        t = x[N]
        s = np.sum(hh)
        if abs(s) < 1e-12:
            return 1e10
        ac = _autoconv(hh)
        a = np.abs(ac)
        margin = t * s * s - a
        if np.any(margin <= 0):
            # infeasible: heavy penalty
            viol = np.sum(np.maximum(0, -margin))
            return t + 1e6 * viol + 1e8 * np.sum(np.log1p(np.maximum(0, -margin) * 100))
        return t + (1.0 / mu) * np.sum(np.log(margin))

    def _epi_jac(x, mu):
        hh = x[:N]
        t = x[N]
        s = np.sum(hh)
        if abs(s) < 1e-12:
            return np.zeros_like(x)
        ac = _autoconv(hh)
        a = np.abs(ac)
        margin = t * s * s - a
        if np.any(margin <= 1e-12):
            return np.zeros_like(x)
        w = (1.0 / mu) / margin  # d/d margin
        # d(margin)/d t = s^2
        # d(margin)/d hh = -sign(ac)*2*cross... and -2*t*s (from t*s^2 derivative w.r.t. hh)
        # margin = t*s^2 - |ac|
        # d(margin)/d hh[j] = t*2*s - sign(ac)*d|ac|/d hh[j]
        # d|ac|/d hh[j] = 2 * cross_corr(sign(ac), hh)[j]  (same structure as before)
        g_ac = np.sign(ac)
        fsize_corr = 1
        while fsize_corr < 3 * N:
            fsize_corr *= 2
        V = np.fft.rfft(g_ac, fsize_corr)
        H = np.fft.rfft(hh, fsize_corr)
        C = np.fft.irfft(V * np.conj(H), fsize_corr)
        dabs = 2.0 * C[:N]
        dmargin_hh = 2.0 * t * s - dabs
        dmargin_t = s * s
        grad_hh = np.sum(w * dmargin_hh)
        grad_t = 1.0 + np.sum(w * dmargin_t)
        g = np.zeros_like(x)
        g[:N] = grad_hh
        g[N] = grad_t
        return g

    # anneal mu from mu0 down to mu_min
    n_mu = 6
    mus = np.geomspace(mu0, mu_min, n_mu)
    x = np.concatenate([h0, [best_val]])
    for mu in mus:
        if deadline is not None and time.time() > deadline - 3:
            break
        try:
            res = minimize(_epi_fun, x, method='L-BFGS-B',
                           jac=lambda z: _epi_jac(z, mu),
                           options={'maxiter': max_iters, 'gtol': 1e-8, 'maxfun': 300})
            x = res.x
            val = _raw_value(x[:N])
            if val < best_val:
                best_val = val
                best_h = x[:N].copy()
        except Exception:
            pass

    return best_h, best_val

def _gen_chirp(n):
    i = np.arange(n)
    phase = np.pi * i**2 / n
    return np.cos(phase)

def _gen_random(n, seed):
    rng = np.random.default_rng(seed)
    return rng.standard_normal(n)

def _gen_window(n, window_type='gaussian'):
    x = np.arange(n)
    if window_type == 'gaussian':
        mu = n / 2
        sigma = n / 4
        return np.exp(-(x - mu)**2 / (2 * sigma**2))
    elif window_type == 'hamming':
        return 0.54 - 0.46 * np.cos(2 * np.pi * x / (n - 1))
    elif window_type == 'triangle':
        mu = n / 2
        return np.maximum(0, 1 - np.abs(x - mu) / mu)
    elif window_type == 'blackman':
        a0, a1, a2 = 0.42, 0.5, 0.08
        return a0 - a1 * np.cos(2 * np.pi * x / (n - 1)) + a2 * np.cos(4 * np.pi * x / (n - 1))
    return np.ones(n)

def _gen_smoothed_random(n, seed, kernel_size=15):
    rng = np.random.default_rng(seed)
    noise = rng.standard_normal(n)
    kernel = np.ones(kernel_size) / kernel_size
    smooth = np.convolve(noise, kernel, mode='same')
    return smooth

def _normalize_sum(h, target=1.0):
    s = np.sum(h)
    if abs(s) < 1e-12:
        return h
    return h * (target / s)

def construct():
    t_start = time.time()
    deadline = t_start + 135.0  # Safety margin before 150s kill

    best_score = np.inf
    best_h = np.ones(N) / N

    base_score = _raw_value(best_h)
    best_score = base_score

    # Generate diverse initializations
    inits = []

    inits.append(np.ones(N) / N)

    h_chirp = _gen_chirp(N)
    inits.append(_normalize_sum(h_chirp))

    for seed in [1, 7, 13, 42, 99]:
        h_rand = _gen_random(N, seed)
        inits.append(_normalize_sum(h_rand))

    for wtype in ['gaussian', 'hamming', 'triangle', 'blackman']:
        h_win = _gen_window(N, wtype)
        inits.append(_normalize_sum(h_win))

    for seed in [5, 11, 23, 37, 53]:
        h_smooth = _gen_smoothed_random(N, seed)
        inits.append(_normalize_sum(h_smooth))

    for seed in [3, 17]:
        rng = np.random.default_rng(seed)
        env = np.exp(-((np.arange(N) - N / 2)**2) / (2 * (N / 4)**2))
        signs = rng.choice([-1, 1], size=N)
        h_alt = signs * env
        inits.append(_normalize_sum(h_alt))

    for freq_combo in [(1, 2), (1, 3), (2, 5), (3, 7)]:
        f1, f2 = freq_combo
        i = np.arange(N)
        h_multi = np.cos(2 * np.pi * f1 * i / N) + 0.5 * np.cos(2 * np.pi * f2 * i / N)
        inits.append(_normalize_sum(h_multi))

    # Optimization pipeline for each init
    for h_init in inits:
        if time.time() > deadline - 25:
            break

        h_work = h_init.copy()

        # Stage 1: Spectral subspace optimization
        remaining = deadline - time.time()
        if remaining > 20:
            h_spec, val_spec = _spectral_subspace_optimize(h_work, dim=8,
                                                           deadline=deadline - 15)
            h_work = h_spec

        # Stage 2: L-BFGS smooth optimization
        remaining = deadline - time.time()
        if remaining > 15:
            h_opt, val_opt = _lbfgs_optimize(h_work, beta=50.0, max_iter=200,
                                             deadline=deadline - 10)
            h_work = h_opt

        # Stage 3: Block sign-flip local search
        remaining = deadline - time.time()
        if remaining > 10:
            h_block, val_block = _block_sign_flip_local_search(h_work,
                                                               block_sizes=[1, 2, 4, 8, 16, 32],
                                                               max_passes=4,
                                                               deadline=deadline - 5)
            h_work = h_block

        # Stage 4: Single flip local search (incremental, fast)
        remaining = deadline - time.time()
        if remaining > 5:
            h_flip, val_flip = _single_flip_local_search(h_work, max_passes=12,
                                                         deadline=deadline - 3)
            h_work = h_flip

        val = _raw_value(h_work)
        if val < best_score:
            best_score = val
            best_h = h_work.copy()

    # Final refinement on the best solution
    remaining = deadline - time.time()
    if remaining > 20:
        h_final, val_final = _block_sign_flip_local_search(best_h,
                                                           block_sizes=[1, 2, 4, 8, 16, 32, 64, 128],
                                                           max_passes=6,
                                                           deadline=deadline - 15)
        if val_final < best_score:
            best_score = val_final
            best_h = h_final

    remaining = deadline - time.time()
    if remaining > 12:
        h_final2, val_final2 = _single_flip_local_search(best_h, max_passes=20,
                                                         deadline=deadline - 8)
        if val_final2 < best_score:
            best_score = val_final2
            best_h = h_final2

    # Epigraph polish (high-precision final pass)
    remaining = deadline - time.time()
    if remaining > 10:
        h_epi, val_epi = _epigraph_polish(best_h, deadline=deadline - 2)
        if val_epi < best_score:
            best_score = val_epi
            best_h = h_epi

    # Ensure we return something valid
    if not np.all(np.isfinite(best_h)):
        best_h = np.ones(N) / N
        best_score = _raw_value(best_h)

    return best_h

# EVOLVE-BLOCK-END
