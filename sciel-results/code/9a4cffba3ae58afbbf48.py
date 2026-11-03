"""Combined: analytic-gradient continuous optimization + O(N) incremental
sign-flip ground-state search (CA-style energy minimization) + block flips."""
import numpy as np
import time
from scipy.optimize import minimize

N = 400

# EVOLVE-BLOCK-START

_FSIZE = 1
while _FSIZE < 2 * N - 1:
    _FSIZE *= 2
_FSIZE3 = 1
while _FSIZE3 < 3 * N:
    _FSIZE3 *= 2


def _autoconv(h):
    H = np.fft.rfft(h, _FSIZE)
    ac = np.fft.irfft(H * H, _FSIZE)
    return ac[:2 * N - 1]


def _raw_value(h):
    s = np.sum(h)
    if abs(s) < 1e-12:
        return 1e10
    ac = _autoconv(h)
    return float(np.max(np.abs(ac))) / (s * s)


def _soft_max_abs(ac, beta=50.0):
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
    s = np.sum(h)
    if abs(s) < 1e-12:
        return 1e10
    ac = _autoconv(h)
    sm, _ = _soft_max_abs(ac, beta)
    return sm / (s * s)


def _smooth_gradient(h, beta=50.0):
    s = np.sum(h)
    if abs(s) < 1e-12:
        return np.zeros_like(h)
    ac = _autoconv(h)
    sm, weights = _soft_max_abs(ac, beta)
    if weights is None:
        return np.zeros_like(h)
    g_ac = weights * np.sign(ac)
    V = np.fft.rfft(g_ac, _FSIZE3)
    H = np.fft.rfft(h, _FSIZE3)
    C = np.fft.irfft(V * np.conj(H), _FSIZE3)
    grad_S = 2.0 * C[:N]
    df = (grad_S * s - 2.0 * sm) / (s ** 3)
    return df


def _lbfgs(h0, beta=50.0, max_iter=200, deadline=None):
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
                       options={'maxiter': max_iter, 'gtol': 1e-7, 'maxfun': 600})
        return res.x, float(res.fun)
    except Exception:
        return h0, _smooth_objective(h0, beta)


def _single_flip_search(h, max_passes=1200, deadline=None):
    """Ground-state refinement map: greedily flip signs using O(N)
    incremental autoconvolution updates (cellular-automaton style)."""
    h = h.copy()
    s = float(np.sum(h))
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
            # delta ac[k] = -2*hi*(h[k-i] + h[i-k])
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


def _block_flip_search(h, block_sizes=None, max_passes=4, deadline=None):
    if block_sizes is None:
        block_sizes = [64, 32, 16, 8, 4, 2, 1]
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
                nv = _raw_value(h)
                if nv < best_val:
                    best_val = nv
                    improved = True
                else:
                    h[start:end] *= -1
        if not improved:
            break
    return h, best_val


def _gen_gaussian(n, sigma_ratio=0.25):
    x = np.arange(n)
    mu = n / 2
    sigma = n * sigma_ratio
    return np.exp(-0.5 * ((x - mu) / sigma) ** 2)


def _gen_chirp(n):
    i = np.arange(n)
    return np.cos(np.pi * i**2 / n)


def _gen_barker_like(n):
    h = np.ones(n)
    for i in range(1, n):
        h[i] = 1.0 if ((i * 7 + 3) % 11) < 5 else -1.0
    return h


def _gen_random_smoothed(n, seed=42, kernel_size=10):
    rng = np.random.default_rng(seed)
    noise = rng.standard_normal(n)
    kernel = np.ones(kernel_size) / kernel_size
    return np.convolve(noise, kernel, mode='same')


def _gen_sparse_support(n, k, seed=42, spacing='uniform'):
    rng = np.random.default_rng(seed)
    if spacing == 'uniform':
        base = np.linspace(0, n - 1, k, endpoint=False).astype(int)
        jitter = rng.integers(-n // k // 2, n // k // 2 + 1, size=k)
        positions = np.sort((base + jitter) % n)
    elif spacing == 'clustered':
        n_clusters = max(2, k // 10)
        centers = rng.choice(n, size=n_clusters, replace=False)
        positions = np.sort(rng.choice(n, size=k, replace=False))
        for c in centers:
            idx = np.argmin(np.abs(positions - c))
            positions[idx] = (c + rng.integers(-5, 5)) % n
        positions = np.sort(positions % n)
    else:
        positions = np.sort(rng.choice(n, size=k, replace=False))
    return positions


def _optimize_sparse_amplitudes(support, init_amplitudes, beta=100.0, deadline=None):
    h = np.zeros(N)
    h[support] = init_amplitudes
    s0 = np.sum(h)
    if abs(s0) < 1e-9:
        return h, 1e10
    h = h / s0

    def obj(x):
        h_x = np.zeros(N)
        h_x[support] = x
        sx = np.sum(h_x)
        if abs(sx) < 1e-9:
            return 1e10
        return _smooth_objective(h_x / sx, beta)

    x0 = h[support].copy()
    try:
        res = minimize(obj, x0, method='L-BFGS-B',
                       options={'maxiter': 80, 'disp': False, 'gtol': 1e-5,
                                'maxfun': 400})
        h_opt = np.zeros(N)
        h_opt[support] = res.x
        s_opt = np.sum(h_opt)
        if abs(s_opt) > 1e-9:
            h_opt = h_opt / s_opt
            return h_opt, _raw_value(h_opt)
    except Exception:
        pass
    return h, _raw_value(h)


def construct():
    t_start = time.time()
    deadline = t_start + 144.0

    best_score = np.inf
    best_h = np.ones(N) / N
    best_score = _raw_value(best_h)

    # ---- Phase 1: sparse combinatorial + amplitude optimization ----
    phase1_deadline = t_start + 55.0
    sparsity_configs = [
        (N // 4, 'uniform', 42),
        (N // 4, 'uniform', 123),
        (N // 4, 'clustered', 42),
        (N // 3, 'uniform', 77),
        (N // 3, 'clustered', 156),
        (N // 5, 'uniform', 42),
        (N // 5, 'random', 42),
        (N // 2, 'uniform', 42),
    ]
    for k, spacing, seed in sparsity_configs:
        if time.time() > phase1_deadline - 6:
            break
        support = _gen_sparse_support(N, k, seed=seed, spacing=spacing)
        rng = np.random.default_rng(seed + 1000)
        amplitudes = rng.choice([-1.0, 1.0], size=k)
        h_opt, val_opt = _optimize_sparse_amplitudes(
            support, amplitudes, beta=100.0, deadline=phase1_deadline - 3)
        if val_opt < best_score:
            best_score = val_opt
            best_h = h_opt.copy()

    # ---- Phase 2: dense continuous optimization with analytic gradient ----
    phase2_deadline = t_start + 95.0
    x = np.arange(N)
    inits = [
        np.ones(N),
        _gen_gaussian(N),
        _gen_chirp(N),
        _gen_barker_like(N),
        _gen_random_smoothed(N, seed=1),
        _gen_random_smoothed(N, seed=2),
        np.cos(np.pi * x / N),
        _gen_gaussian(N, sigma_ratio=0.3) * np.where(x % 2 == 0, 1.0, -1.0),
        _gen_gaussian(N, sigma_ratio=0.15) * np.where(x % 3 == 0, 1.0, -1.0),
    ]
    for h_init in inits:
        if time.time() > phase2_deadline - 8:
            break
        s0 = np.sum(h_init)
        if abs(s0) < 1e-6:
            continue
        h_norm = h_init / s0
        try:
            current_h = h_norm.copy()
            for beta in [40, 80, 150]:
                if time.time() > phase2_deadline - 6:
                    break
                h_opt, _ = _lbfgs(current_h, beta=beta, max_iter=150,
                                  deadline=phase2_deadline - 3)
                if abs(np.sum(h_opt)) < 1e-6:
                    continue
                val = _raw_value(h_opt)
                if val < best_score:
                    best_score = val
                    best_h = h_opt.copy()
                current_h = h_opt.copy()
        except Exception:
            pass

    # ---- Phase 3: block-flip structural refinement ----
    if time.time() < deadline - 30:
        h_ref, val_ref = _block_flip_search(best_h, max_passes=3,
                                            deadline=deadline - 28)
        if val_ref < best_score:
            best_score = val_ref
            best_h = h_ref.copy()

    # ---- Phase 4: ground-state single-flip refinement (CA energy map) ----
    if time.time() < deadline - 8:
        h_ref, val_ref = _single_flip_search(best_h, max_passes=1500,
                                             deadline=deadline - 6)
        if val_ref < best_score:
            best_score = val_ref
            best_h = h_ref.copy()

    # ---- Phase 5: final polish ----
    if time.time() < deadline - 2:
        h_fin, val_fin = _single_flip_search(best_h, max_passes=800,
                                             deadline=deadline - 1)
        if val_fin < best_score:
            best_score = val_fin
            best_h = h_fin.copy()

    if not np.all(np.isfinite(best_h)):
        return np.ones(N) / N
    return best_h

# EVOLVE-BLOCK-END
