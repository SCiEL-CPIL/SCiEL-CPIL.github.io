"""Erdos minimum overlap: spectral-parameterized optimization with fast projected gradient.
Combines the structural idea of representing h in a trigonometric basis (reducing search
space and naturally handling the sum constraint) with the refinement machinery of
Nesterov-accelerated projected gradient using binary-search projection for exact feasibility.
The key insight: the optimal solution is smooth, so a low-frequency DFT basis captures
most of the structure. We optimize in the full space but use spectral initialization
and the fast projection to enable many iterations within the time budget.
"""
import time
import numpy as np

N = 100
_TARGET = N / 2.0
_NFFT = 256  # next power of 2 >= 2*N


def _autocorr(h):
    """Linear autocorrelation r[k] = sum_{i=k}^{N-1} h[i]*h[i-k] for k=0..N-1 via FFT."""
    h_pad = np.zeros(_NFFT)
    h_pad[:N] = h
    H = np.fft.rfft(h_pad)
    corr = np.fft.irfft(H * np.conj(H))
    return corr[:N]


def _overlap_values(h):
    """C_k = (2/N)*(N/2 - r_k) for k=0..N-1."""
    r = _autocorr(h)
    return (2.0 / N) * (N / 2.0 - r)


def _score(h):
    """True (hard) maximum overlap value."""
    cs = _overlap_values(h)
    return float(np.max(cs))


def _project(h):
    """Exact Euclidean projection onto {h in [0,1]^N : sum(h) = N/2} via bisection on tau."""
    h = np.array(h, dtype=np.float64, copy=True)
    target = _TARGET

    if abs(np.sum(h) - target) < 1e-13 and np.min(h) >= -1e-13 and np.max(h) <= 1 + 1e-13:
        return np.clip(h, 0.0, 1.0)

    lo = np.min(h) - 2.0
    hi = np.max(h) + 2.0

    def calc_sum(tau):
        return np.sum(np.clip(h - tau, 0.0, 1.0))

    if calc_sum(lo) < target:
        lo -= 10.0
    if calc_sum(hi) > target:
        hi += 10.0

    for _ in range(60):
        mid = (lo + hi) / 2.0
        s = calc_sum(mid)
        if abs(s - target) < 1e-14:
            lo = hi = mid
            break
        if s > target:
            lo = mid
        else:
            hi = mid

    tau = (lo + hi) / 2.0
    h_proj = np.clip(h - tau, 0.0, 1.0)

    # Final exact correction
    diff = target - np.sum(h_proj)
    if abs(diff) > 1e-15:
        mask = (h_proj > 1e-10) & (h_proj < 1.0 - 1e-10)
        if np.sum(mask) > 0:
            h_proj[mask] += diff / np.sum(mask)
            h_proj = np.clip(h_proj, 0.0, 1.0)
        else:
            if diff > 0:
                idx = np.argmin(h_proj)
            else:
                idx = np.argmax(h_proj)
            h_proj[idx] += diff
            h_proj = np.clip(h_proj, 0.0, 1.0)

    return h_proj


def _smooth_max_and_grad(x, beta):
    """Log-sum-exp soft-max of x and its gradient (softmax(beta*x))."""
    m = np.max(x)
    e = np.exp(beta * (x - m))
    s = np.sum(e)
    val = (m + np.log(s)) / beta
    grad = e / s
    return val, grad


def _grad_smooth_max_wrt_h(h, g_c):
    """Compute gradient of S_beta(C(h)) w.r.t. h, given g_c = dS/dC.
    Uses FFT-based convolution for speed.
    """
    factor = -(2.0 / N)
    gc_pad = np.zeros(_NFFT)
    gc_pad[:N] = g_c
    h_pad = np.zeros(_NFFT)
    h_pad[:N] = h
    hr_pad = np.zeros(_NFFT)
    hr_pad[:N] = h[::-1]

    GC = np.fft.rfft(gc_pad)
    HC = np.fft.rfft(h_pad)
    HRC = np.fft.rfft(hr_pad)

    conv1 = np.fft.irfft(GC * HC, n=_NFFT)[:N]
    conv2_full = np.fft.irfft(GC * HRC, n=_NFFT)
    term2 = conv2_full[N-1::-1][:N]

    g_h = factor * (conv1 + term2)
    return g_h


def _smooth_obj_and_grad(h, beta):
    """f(h) = S_beta(C(h)); returns (f, df/dh)."""
    cs = _overlap_values(h)
    val, g_c = _smooth_max_and_grad(cs, beta)
    g_h = _grad_smooth_max_wrt_h(h, g_c)
    return val, g_h


def _nesterov_projected(h0, beta, n_iter=2000, step_size=None, time_budget=10.0):
    """Nesterov accelerated projected gradient on smooth-max surrogate."""
    t_start = time.time()
    h = _project(h0.copy())
    y = h.copy()
    best_h = h.copy()
    best_val = _smooth_obj_and_grad(h, beta)[0]

    if step_size is None:
        step_size = 1.0 / (beta * 10.0)

    t_k = 1.0

    for it in range(n_iter):
        if time.time() - t_start > time_budget:
            break

        val_y, grad_y = _smooth_obj_and_grad(y, beta)
        h_new = _project(y - step_size * grad_y)
        val_new = _smooth_obj_and_grad(h_new, beta)[0]

        # Backtracking
        for _ in range(5):
            if val_new <= best_val + 1e-13:
                break
            step_size *= 0.7
            h_new = _project(y - step_size * grad_y)
            val_new = _smooth_obj_and_grad(h_new, beta)[0]
        else:
            step_size *= 0.5

        t_new = (1.0 + np.sqrt(1.0 + 4.0 * t_k ** 2)) / 2.0
        y = h_new + ((t_k - 1.0) / t_new) * (h_new - h)
        y = _project(y)
        h = h_new
        t_k = t_new

        if val_new < best_val:
            best_val = val_new
            best_h = h.copy()

    return best_h, _score(best_h)


def _spectral_init(frequencies, amplitudes, phases=None):
    """Construct h from a sum of cosine basis functions."""
    i = np.arange(N)
    h = np.full(N, 0.5)
    if phases is None:
        phases = [0.0] * len(frequencies)
    for f, a, p in zip(frequencies, amplitudes, phases):
        h += a * np.cos(2.0 * np.pi * f * i / N + p)
    return _project(h)


def _spectral_to_h(coeffs):
    """Convert cosine coefficients to h. coeffs[0] is DC (fixed to 0.5)."""
    i = np.arange(N)
    h = np.full(N, coeffs[0])
    K = min(N // 2, len(coeffs) - 1)
    for k in range(1, K + 1):
        h += coeffs[k] * np.cos(2.0 * np.pi * k * i / N)
    return h


def _h_to_spectral(h):
    """Extract cosine coefficients from h."""
    i = np.arange(N)
    coeffs = np.zeros(N)
    coeffs[0] = np.mean(h)
    K = N // 2
    for k in range(1, K + 1):
        coeffs[k] = (2.0 / N) * np.sum(h * np.cos(2.0 * np.pi * k * i / N))
    return coeffs


def construct():
    """Construct step heights minimizing maximum overlap with shifted complement."""
    t0 = time.time()
    deadline = t0 + 140  # Safety margin before 150s kill
    rng = np.random.default_rng(42)

    best_h = None
    best_score = float('inf')

    def _update(h):
        nonlocal best_h, best_score
        h = _project(h)
        s = _score(h)
        if s < best_score:
            best_score = s
            best_h = h.copy()
        return s

    # Phase 1: Diverse spectral initializations
    i = np.arange(N)
    spectral_configs = [
        # (frequencies, amplitudes, phases)
        ([1, 2, 3, 5, 7, 11], [0.12, 0.10, 0.09, 0.07, 0.06, 0.05], [0.0, 0.3, 0.7, 1.2, 1.8, 2.3]),
        ([1, 2, 3, 5, 7, 11, 13, 17], [0.11, 0.10, 0.09, 0.07, 0.06, 0.05, 0.04, 0.03], [0.0, 0.2, 0.5, 0.9, 1.4, 1.9, 2.4, 2.9]),
        ([2, 3, 5, 8, 13, 19], [0.10, 0.09, 0.08, 0.06, 0.05, 0.04], [0.1, 0.4, 0.8, 1.3, 1.8, 2.3]),
        ([1, 3, 4, 6, 9, 14], [0.10, 0.09, 0.08, 0.06, 0.05, 0.04], [0.0, 0.3, 0.6, 1.0, 1.5, 2.0]),
        ([1, 2, 4, 7, 13, 17], [0.11, 0.10, 0.08, 0.06, 0.05, 0.04], [0.0, 0.2, 0.5, 0.9, 1.4, 1.9]),
        ([3, 5, 7, 11, 13, 17], [0.10, 0.09, 0.08, 0.06, 0.05, 0.04], [0.1, 0.4, 0.7, 1.1, 1.6, 2.1]),
        ([1, 2, 3, 4, 5, 6, 8, 11], [0.09, 0.08, 0.07, 0.06, 0.05, 0.04, 0.03, 0.03], [0.0, 0.15, 0.3, 0.5, 0.7, 0.9, 1.2, 1.5]),
        ([2, 4, 6, 8, 10, 12, 15], [0.09, 0.08, 0.07, 0.06, 0.05, 0.04, 0.03], [0.0, 0.2, 0.4, 0.7, 1.0, 1.3, 1.7]),
        ([1, 2, 3, 5, 8, 13, 21], [0.11, 0.10, 0.09, 0.07, 0.06, 0.05, 0.04], [0.0, 0.2, 0.5, 0.9, 1.4, 1.9, 2.4]),
        ([1, 3, 5, 7, 11, 13, 17, 19], [0.09, 0.08, 0.07, 0.06, 0.05, 0.04, 0.03, 0.03], [0.0, 0.2, 0.5, 0.8, 1.2, 1.6, 2.0, 2.4]),
    ]

    for freqs, amps, phases in spectral_configs:
        if time.time() > deadline - 80:
            break
        h_spec = _spectral_init(freqs, amps, phases)
        _update(h_spec)

    # Phase 2: Random starts
    if time.time() < deadline - 70:
        for _ in range(5):
            if time.time() > deadline - 70:
                break
            h_rand = rng.uniform(0.0, 1.0, N)
            _update(h_rand)

    # Phase 3: Nesterov refinement with annealed beta from best starts
    if best_h is not None:
        # Refine the best spectral start
        for beta in [20.0, 50.0, 100.0, 200.0]:
            if time.time() > deadline - 40:
                break
            h_nest, s_nest = _nesterov_projected(best_h, beta, n_iter=1500, time_budget=12.0)
            _update(h_nest)

        # Refine from a few other diverse starts
        if time.time() < deadline - 30:
            for _ in range(3):
                if time.time() > deadline - 30:
                    break
                h_rand = rng.uniform(0.0, 1.0, N)
                for beta in [50.0, 100.0]:
                    if time.time() > deadline - 25:
                        break
                    h_nest, s_nest = _nesterov_projected(h_rand, beta, n_iter=1000, time_budget=10.0)
                    _update(h_nest)

    # Phase 4: High-precision polish with very high beta
    if best_h is not None and time.time() < deadline - 15:
        for beta in [300.0, 500.0, 800.0]:
            if time.time() > deadline - 8:
                break
            h_polish, s_polish = _nesterov_projected(best_h, beta, n_iter=1000, step_size=0.002, time_budget=8.0)
            _update(h_polish)

    # Phase 5: Perturbation-based restarts
    if best_h is not None and time.time() < deadline - 10:
        for _ in range(3):
            if time.time() > deadline - 8:
                break
            h_pert = _project(best_h + rng.normal(0.0, 0.03, N))
            _update(h_pert)
            if time.time() > deadline - 6:
                break
            h_nest, s_nest = _nesterov_projected(h_pert, 200.0, n_iter=800, time_budget=5.0)
            _update(h_nest)

    # Phase 6: Final polish
    if best_h is not None and time.time() < deadline - 3:
        h_final, s_final = _nesterov_projected(best_h, 1000.0, n_iter=500, step_size=0.001, time_budget=5.0)
        _update(h_final)

    if best_h is None:
        best_h = np.full(N, 0.5)

    return _project(best_h)
