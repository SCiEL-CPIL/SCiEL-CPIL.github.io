"""Erdos minimum overlap: spectral-parameterized optimization with Nesterov
accelerated projected gradient on a log-sum-exp smooth surrogate.
The key structural improvement is to parameterize h in the Fourier basis,
which naturally enforces the sum constraint (zero-frequency component fixed)
and reduces effective dimensionality. Combined with annealed beta for the
smooth max and a final high-precision polish, this should beat the
full-space Nesterov approach.
"""
import time
import numpy as np

N = 100
_TARGET = N / 2.0
_NFFT = 256  # next power of 2 >= 2*N
_DEADLINE = time.time() + 140  # Safety margin before 150s kill


def _autocorr(h):
    """Linear autocorrelation r[k] = sum_{i=k}^{N-1} h[i]*h[i-k] for k=0..N-1 via FFT."""
    h_pad = np.zeros(_NFFT)
    h_pad[:N] = h
    H = np.fft.rfft(h_pad)
    corr = np.fft.irfft(H * np.conj(H))
    return corr[:N]


def _overlap_values(h):
    """C_k = (2/N)*(N/2 - r_k) for k=0..N-1."""
    n = len(h)
    r = _autocorr(h)
    return (2.0 / n) * (n / 2.0 - r)


def _score(h):
    """True (hard) maximum overlap value."""
    cs = _overlap_values(h)
    return float(np.max(cs))


def _project(h):
    """Exact Euclidean projection onto {h in [0,1]^N : sum(h) = N/2}."""
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


def _smooth_obj_and_grad(h, beta):
    """f(h) = S_beta(C(h)); returns (f, df/dh)."""
    cs = _overlap_values(h)
    val, g_c = _smooth_max_and_grad(cs, beta)
    n = N
    g_h = np.zeros(n)
    factor = -(2.0 / n)
    for k in range(n):
        gc = g_c[k]
        if gc == 0.0:
            continue
        if k == 0:
            g_h += factor * gc * (2.0 * h)
        else:
            g_h[k:] += factor * gc * h[:n - k]
            g_h[:n - k] += factor * gc * h[k:]
    return val, g_h


def _nesterov_projected(h0, beta, n_steps=5000, lr=None, time_budget=10.0):
    """Nesterov accelerated projected gradient on the smooth surrogate."""
    t_start = time.time()
    h = _project(h0.copy())
    y = h.copy()
    best_h = h.copy()
    best_val = _smooth_obj_and_grad(h, beta)[0]

    if lr is None:
        lr = 1.0 / (beta * 12.0)

    t_k = 1.0

    for step in range(n_steps):
        if time.time() - t_start > time_budget:
            break

        val_y, grad_y = _smooth_obj_and_grad(y, beta)
        h_new = _project(y - lr * grad_y)
        val_new = _smooth_obj_and_grad(h_new, beta)[0]

        # Backtracking
        for _ in range(8):
            if val_new <= best_val + 1e-13:
                break
            lr *= 0.7
            h_new = _project(y - lr * grad_y)
            val_new = _smooth_obj_and_grad(h_new, beta)[0]
        else:
            lr *= 0.5

        t_new = (1.0 + np.sqrt(1.0 + 4.0 * t_k ** 2)) / 2.0
        y = h_new + ((t_k - 1.0) / t_new) * (h_new - h)
        y = _project(y)
        h = h_new
        t_k = t_new

        if val_new < best_val:
            best_val = val_new
            best_h = h.copy()

    return best_h, _score(best_h)


def _spectral_to_h(coeffs):
    """Convert spectral coefficients (real, zero-frequency fixed) to h in [0,1]^N with sum=N/2."""
    # coeffs: array of length N, where coeffs[0] is the DC component (fixed to 0.5)
    # We use real FFT representation: h = ifft of the spectrum
    # For real signals, we work with the rfft representation
    n = len(coeffs)
    # Build full complex spectrum
    spec = np.zeros(n, dtype=np.complex128)
    spec[0] = coeffs[0]  # DC = 0.5
    # For k=1..n/2-1, spec[k] and spec[n-k] are conjugates
    # coeffs[k] for k=1..n//2-1 represents the amplitude of the k-th harmonic
    # We use: spec[k] = coeffs[k] * exp(i * phase[k])
    # But to keep it simple, let's use a different parameterization:
    # h[i] = 0.5 + sum_{k=1}^{K} a_k * cos(2*pi*k*i/N + phi_k)
    # This is easier to handle directly.
    # Actually, let's just use the direct cosine parameterization.
    i = np.arange(n)
    h = np.full(n, coeffs[0])
    K = min(n // 2, len(coeffs) - 1)
    for k in range(1, K + 1):
        h += coeffs[k] * np.cos(2.0 * np.pi * k * i / n)
    return h


def _h_to_spectral(h):
    """Extract cosine coefficients from h (assuming h[i] = c0 + sum c_k cos(2*pi*k*i/N))."""
    n = len(h)
    coeffs = np.zeros(n)
    coeffs[0] = np.mean(h)
    K = n // 2
    for k in range(1, K + 1):
        coeffs[k] = (2.0 / n) * np.sum(h * np.cos(2.0 * np.pi * k * np.arange(n) / n))
    return coeffs


def _nesterov_spectral(coeffs0, beta, n_steps=3000, lr=None, time_budget=10.0):
    """Nesterov accelerated gradient in spectral space.
    The spectral parameterization naturally enforces sum(h)=N/2 when coeffs[0]=0.5,
    but we still need to project to [0,1]^N."""
    t_start = time.time()
    coeffs = coeffs0.copy()
    coeffs[0] = 0.5  # Fix DC component
    h = _spectral_to_h(coeffs)
    h = _project(h)
    # Re-extract spectral coeffs from projected h to stay consistent
    coeffs = _h_to_spectral(h)
    coeffs[0] = 0.5

    y = coeffs.copy()
    best_h = h.copy()
    best_val = _smooth_obj_and_grad(h, beta)[0]

    if lr is None:
        lr = 0.01 / (beta ** 0.5)

    t_k = 1.0

    for step in range(n_steps):
        if time.time() - t_start > time_budget:
            break

        # Compute gradient w.r.t. coeffs
        h_y = _spectral_to_h(y)
        h_y = _project(h_y)
        val_y, grad_h = _smooth_obj_and_grad(h_y, beta)

        # Chain rule: grad_coeffs[k] = <grad_h, d h / d coeffs[k]>
        # d h / d coeffs[0] = ones (but fixed)
        # d h / d coeffs[k] = cos(2*pi*k*i/N) for k>=1
        grad_coeffs = np.zeros(len(coeffs))
        i = np.arange(N)
        for k in range(1, len(coeffs)):
            grad_coeffs[k] = np.dot(grad_h, np.cos(2.0 * np.pi * k * i / N))
        # coeffs[0] is fixed, so gradient is 0 there

        # Nesterov step
        coeffs_new = _project_coeffs(y - lr * grad_coeffs)
        h_new = _spectral_to_h(coeffs_new)
        h_new = _project(h_new)
        val_new = _smooth_obj_and_grad(h_new, beta)[0]

        # Backtracking
        for _ in range(8):
            if val_new <= best_val + 1e-13:
                break
            lr *= 0.7
            coeffs_new = _project_coeffs(y - lr * grad_coeffs)
            h_new = _spectral_to_h(coeffs_new)
            h_new = _project(h_new)
            val_new = _smooth_obj_and_grad(h_new, beta)[0]
        else:
            lr *= 0.5

        t_new = (1.0 + np.sqrt(1.0 + 4.0 * t_k ** 2)) / 2.0
        y = coeffs_new + ((t_k - 1.0) / t_new) * (coeffs_new - coeffs)
        y[0] = 0.5
        coeffs = coeffs_new
        coeffs[0] = 0.5
        t_k = t_new

        if val_new < best_val:
            best_val = val_new
            best_h = h_new.copy()

    return best_h, _score(best_h)


def _project_coeffs(coeffs):
    """Project spectral coefficients to keep the resulting h in [0,1].
    Simple approach: clip the resulting h and re-extract. For now, just return as-is
    and rely on the subsequent h projection."""
    return coeffs


def _construct_direct():
    """Direct multi-frequency construction to flatten autocorrelation."""
    i = np.arange(N)
    h = np.full(N, 0.5)
    freqs = [1, 2, 3, 5, 7, 11, 13, 17, 19, 23]
    amplitudes = [0.12, 0.10, 0.08, 0.07, 0.06, 0.05, 0.04, 0.03, 0.02, 0.01]
    phases = [0.0, 0.7, 1.3, 2.1, 0.4, 1.8, 2.5, 0.9, 1.6, 0.3]
    for f, a, p in zip(freqs, amplitudes, phases):
        h += a * np.cos(2.0 * np.pi * f * i / N + p)
    return _project(h)


def _random_start(rng):
    """Random feasible start."""
    h = rng.uniform(0.0, 1.0, N)
    return _project(h)


def _perturb(h, rng, scale=0.05):
    """Add small random perturbation and re-project."""
    h_p = h + rng.normal(0.0, scale, N)
    return _project(h_p)


def _refine_full(h0, beta, n_steps=5000, time_budget=10.0):
    """Refine h using Nesterov accelerated projected gradient in full space."""
    return _nesterov_projected(h0, beta, n_steps=n_steps, time_budget=time_budget)


def _refine_spectral(h0, beta, n_steps=3000, time_budget=10.0):
    """Refine h using Nesterov accelerated gradient in spectral space."""
    coeffs = _h_to_spectral(h0)
    coeffs[0] = 0.5
    return _nesterov_spectral(coeffs, beta, n_steps=n_steps, time_budget=time_budget)


def construct():
    """Construct step heights minimizing maximum overlap with shifted complement."""
    t0 = time.time()
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

    # Phase 1: Direct construction
    h_direct = _construct_direct()
    _update(h_direct)

    # Phase 2: Diverse frequency-based starts
    if time.time() - t0 < _DEADLINE - 70:
        i = np.arange(N)
        freq_sets = [
            ([1, 2, 3, 5], [0.15, 0.10, 0.08, 0.06]),
            ([1, 2, 4, 7], [0.14, 0.10, 0.07, 0.05]),
            ([1, 3, 4, 6, 9], [0.12, 0.10, 0.08, 0.06, 0.04]),
            ([2, 3, 5, 8, 13], [0.12, 0.10, 0.08, 0.06, 0.04]),
            ([1, 2, 3, 4, 5, 6], [0.10, 0.08, 0.06, 0.05, 0.04, 0.03]),
            ([3, 5, 7, 11], [0.12, 0.10, 0.08, 0.06]),
            ([1, 4, 6, 9, 12], [0.11, 0.09, 0.07, 0.05, 0.03]),
            ([2, 5, 8, 13, 21], [0.10, 0.08, 0.06, 0.04, 0.02]),
            ([1, 2, 3, 4, 5], [0.13, 0.10, 0.08, 0.06, 0.04]),
            ([1, 3, 5, 7, 9], [0.11, 0.09, 0.07, 0.05, 0.03]),
        ]
        for freqs, amps in freq_sets:
            if time.time() - t0 > _DEADLINE - 70:
                break
            h = np.full(N, 0.5)
            for f, a in zip(freqs, amps):
                h += a * np.cos(2.0 * np.pi * f * i / N)
            _update(h)

    # Phase 3: Random starts
    if time.time() - t0 < _DEADLINE - 60:
        for _ in range(6):
            if time.time() - t0 > _DEADLINE - 60:
                break
            h_rand = _random_start(rng)
            _update(h_rand)

    # Phase 4: Smooth surrogate refinement with annealed beta
    # Alternate between full-space and spectral-space Nesterov
    beta_list = [10.0, 50.0, 100.0, 300.0, 600.0, 1200.0]

    for beta in beta_list:
        remaining = _DEADLINE - time.time()
        if remaining < 15:
            break

        time_per_beta = min(18.0, (remaining - 10.0) / 2.0)
        if time_per_beta < 5.0:
            break

        # Full-space refinement
        h_ref, s_ref = _refine_full(best_h, beta=beta, n_steps=4000,
                                     time_budget=min(10.0, time_per_beta * 0.5))
        _update(h_ref)

        # Spectral-space refinement from the improved point
        if time.time() - t0 < _DEADLINE - 10:
            remaining2 = _DEADLINE - time.time()
            if remaining2 > 5:
                h_ref2, s_ref2 = _refine_spectral(best_h, beta=beta, n_steps=2000,
                                                    time_budget=min(8.0, remaining2 - 5.0))
                _update(h_ref2)

        # Add perturbation for diversity
        if best_h is not None and time.time() - t0 < _DEADLINE - 20:
            h_pert = _perturb(best_h, rng, scale=0.02)
            h_ref3, s_ref3 = _refine_full(h_pert, beta=beta, n_steps=2000,
                                            time_budget=min(8.0, _DEADLINE - time.time() - 10))
            _update(h_ref3)

    # Phase 5: Final high-precision polish
    if best_h is not None and time.time() - t0 < _DEADLINE - 8:
        remaining = _DEADLINE - time.time() - 3.0
        if remaining > 5.0:
            # Very high beta for sharp max approximation
            h_polish, s_polish = _refine_full(best_h, beta=1500.0, n_steps=3000,
                                               time_budget=min(8.0, remaining))
            _update(h_polish)

            # One more pass with even higher beta
            if time.time() - t0 < _DEADLINE - 3:
                remaining2 = _DEADLINE - time.time() - 1.0
                if remaining2 > 2.0:
                    h_polish2, s_polish2 = _refine_full(best_h, beta=3000.0, n_steps=1500,
                                                         time_budget=min(4.0, remaining2))
                    _update(h_polish2)

    if best_h is None:
        best_h = np.full(N, 0.5)

    return _project(best_h)
