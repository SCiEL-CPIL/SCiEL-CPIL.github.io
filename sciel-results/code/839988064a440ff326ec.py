"""Combined: FFT-based smooth objective + diverse cosine/spike initialization + multi-start L-BFGS-B + coordinate descent polish."""
import numpy as np
import time
from scipy.optimize import minimize

N = 400
DEADLINE = 145.0


def _evaluate(h):
    """Return (value, valid) for the exact task objective."""
    if not np.all(np.isfinite(h)):
        return np.inf, False
    s = float(np.sum(h))
    if abs(s) < 1e-12:
        return np.inf, False
    conv = np.convolve(h, h)
    max_conv = float(np.max(conv))
    val = abs(2.0 * N * max_conv / (s * s))
    if not np.isfinite(val):
        return np.inf, False
    return val, True


def _smooth_objective(x, beta):
    """Smooth (soft-max) surrogate using FFT for convolution."""
    s = float(np.sum(x))
    if abs(s) < 1e-10:
        return 1e10
    nfft = 2 * N - 1
    X = np.fft.rfft(x, n=nfft)
    conv = np.fft.irfft(X * X, n=nfft)[:2 * N - 1]
    m = float(np.max(conv))
    shifted = conv - m
    shifted = np.clip(shifted, -50.0, 0.0)
    log_sum_exp = m + np.log(np.sum(np.exp(beta * shifted))) / beta
    val = 2.0 * N * log_sum_exp / (s * s)
    if not np.isfinite(val):
        return 1e10
    return float(val)


def _smooth_grad(x, beta):
    """FFT-vectorized gradient of the soft-max surrogate objective."""
    s = float(np.sum(x))
    if abs(s) < 1e-10:
        return np.zeros_like(x)
    nfft = 2 * N - 1
    X = np.fft.rfft(x, n=nfft)
    conv = np.fft.irfft(X * X, n=nfft)[:2 * N - 1]
    m = float(np.max(conv))
    w = np.exp(beta * (conv - m))
    w_sum = float(np.sum(w))
    w /= w_sum
    W = np.fft.rfft(w, n=nfft)
    grad_lse = 2.0 * np.fft.irfft(W * np.conj(X), n=nfft)[:N]
    lse = m + np.log(w_sum) / beta
    f = 2.0 * N * lse / (s * s)
    df = 2.0 * N * grad_lse / (s * s) - 4.0 * N * lse / (s * s * s)
    return df


def _diverse_init(max_time=12.0):
    """Generate diverse initial candidates: spikes, cosine modes, random."""
    t0 = time.time()
    candidates = []
    x = np.arange(N)
    rng = np.random.default_rng(123)

    # Spike candidates (high quality starts)
    for a in (0.3, 0.4, 0.5, 0.6, 0.7):
        h = np.zeros(N)
        h[0] = a
        h[1] = 1.0 - a
        h[2:100] = -0.001
        candidates.append(h)

    # Two-spike with tail
    h = np.zeros(N)
    h[0] = 0.5
    h[1] = 0.5
    h[2:200] = -0.001
    candidates.append(h)

    # Cosine modes (from inspiration)
    for k in [1, 2, 3, 5, 7, 10, 15, 20, 30, 50, 75, 100]:
        for amp in [0.5, 1.0, 1.5, 2.0, 3.0]:
            h = np.ones(N) + amp * np.cos(2 * np.pi * k * x / N)
            s = np.sum(h)
            if abs(s) > 1e-12:
                candidates.append(h / s)

    # Random Gaussian
    for trial in range(15):
        if time.time() - t0 > max_time * 0.5:
            break
        h = rng.normal(0, 1, N)
        s = np.sum(h)
        if abs(s) > 1e-12:
            candidates.append(h / s)

    # Block structure
    for nblocks in [4, 8, 16, 32]:
        block_size = N // nblocks
        for seed in range(3):
            if time.time() - t0 > max_time * 0.7:
                break
            rng_block = np.random.default_rng(seed + 1000)
            h = np.zeros(N)
            for b in range(nblocks):
                val = rng_block.normal(0, 1)
                h[b * block_size:(b + 1) * block_size] = val
            s = np.sum(h)
            if abs(s) > 1e-12:
                candidates.append(h / s)

    # Multi-frequency cosine
    for n_freq in range(2, 5):
        if time.time() - t0 > max_time * 0.85:
            break
        for trial in range(5):
            freqs = sorted(rng.uniform(1, 40, n_freq))
            phases = rng.uniform(0, 2 * np.pi, n_freq)
            amps = rng.uniform(0.2, 1.0, n_freq)
            dc = rng.uniform(-0.1, 0.1)
            h = np.ones(N) * dc
            for f, p, a in zip(freqs, phases, amps):
                h += a * np.cos(2 * np.pi * f * x / N + p)
            s = np.sum(h)
            if abs(s) > 1e-12:
                candidates.append(h / s)

    # Evaluate all candidates
    best_h = None
    best_val = np.inf
    for h in candidates:
        if not np.all(np.isfinite(h)):
            continue
        val, valid = _evaluate(h)
        if valid and val < best_val:
            best_val = val
            best_h = h.copy()

    if best_h is None:
        best_h = np.ones(N)
        best_val, _ = _evaluate(best_h)

    return best_h, best_val


def _local_refine(h0, max_time=20.0):
    """Local refinement using smooth objective with increasing beta."""
    t0 = time.time()
    best_h = h0.copy()
    best_val, _ = _evaluate(h0)

    for beta in [40.0, 80.0, 160.0, 320.0, 640.0]:
        if time.time() - t0 > max_time * 0.8:
            break
        try:
            res = minimize(
                lambda x, b=beta: _smooth_objective(x, b),
                best_h.copy(),
                jac=lambda x, b=beta: _smooth_grad(x, b),
                method='L-BFGS-B',
                options={'maxiter': 2000, 'maxfun': 6000, 'gtol': 1e-14},
            )
            if res.x is not None and np.all(np.isfinite(res.x)):
                val, valid = _evaluate(res.x)
                if valid and val < best_val:
                    best_val = val
                    best_h = res.x.copy()
        except Exception:
            pass

    return best_h, best_val


def _multi_start(h_best, n_starts=8, max_time=25.0):
    """Multi-start local refinement with perturbations."""
    t0 = time.time()
    best_h = h_best.copy()
    best_val, _ = _evaluate(h_best)
    rng = np.random.default_rng(42)

    for trial in range(n_starts):
        if time.time() - t0 > max_time * 0.85:
            break

        scale = 0.05 + 0.03 * trial
        perturbation = rng.normal(0, scale, N)
        perturbation -= np.mean(perturbation)

        h_init = best_h + perturbation
        s = np.sum(h_init)
        if abs(s) < 1e-12:
            continue
        h_init = h_init / s

        try:
            res = minimize(
                lambda x: _smooth_objective(x, 100.0),
                h_init,
                jac=lambda x: _smooth_grad(x, 100.0),
                method='L-BFGS-B',
                options={'maxiter': 1000, 'maxfun': 3000, 'ftol': 1e-14, 'gtol': 1e-12},
            )
            if res.x is not None and np.all(np.isfinite(res.x)):
                val, valid = _evaluate(res.x)
                if valid and val < best_val:
                    best_val = val
                    best_h = res.x.copy()
        except Exception:
            pass

    return best_h, best_val


def _exact_local_search(h, n_rounds=80, step=0.01, deadline=140.0):
    """Coordinate-descent local search on the true nonsmooth objective."""
    best_h = h.copy()
    best_val, _ = _evaluate(best_h)
    rng = np.random.RandomState(42)

    for _ in range(n_rounds):
        if time.time() > deadline:
            break
        improved = False
        for idx in range(N):
            if time.time() > deadline:
                break
            for delta in (step, -step, step * 0.5, -step * 0.5, step * 0.1, -step * 0.1):
                trial = best_h.copy()
                trial[idx] += delta
                val, valid = _evaluate(trial)
                if valid and val < best_val:
                    best_h = trial
                    best_val = val
                    improved = True
                    break
        if not improved:
            step *= 0.6
            if step < 1e-7:
                break
    return best_h, best_val


def construct():
    start = time.time()

    def time_left():
        return DEADLINE - (time.time() - start)

    # Phase 1: Diverse initialization
    best_h, best_val = _diverse_init(max_time=12.0)

    # Phase 2: Local refinement with increasing beta
    if time_left() > 35:
        best_h, best_val = _local_refine(best_h, max_time=min(25.0, time_left() - 25.0))

    # Phase 3: Multi-start refinement
    if time_left() > 25:
        best_h, best_val = _multi_start(best_h, n_starts=8, max_time=min(25.0, time_left() - 15.0))

    # Phase 4: Final high-precision polish
    if time_left() > 12:
        try:
            res = minimize(
                lambda x: _smooth_objective(x, 2048.0),
                best_h.copy(),
                jac=lambda x: _smooth_grad(x, 2048.0),
                method='L-BFGS-B',
                options={'maxiter': 3000, 'maxfun': 8000, 'gtol': 1e-15},
            )
            if res.x is not None and np.all(np.isfinite(res.x)):
                val, valid = _evaluate(res.x)
                if valid and val < best_val:
                    best_val = val
                    best_h = res.x.copy()
        except Exception:
            pass

    # Phase 5: Exact coordinate-descent polish
    if time_left() > 5:
        best_h, best_val = _exact_local_search(best_h, n_rounds=60, step=0.005, deadline=DEADLINE - 3)

    # Final validity guard
    val, valid = _evaluate(best_h)
    if not valid or not np.isfinite(val):
        best_h = np.zeros(N)
        best_h[0] = 0.5
        best_h[1] = 0.5
        best_h[2:100] = -0.001
        val, valid = _evaluate(best_h)
    if not valid or not np.isfinite(val):
        best_h = np.ones(N)

    return best_h
