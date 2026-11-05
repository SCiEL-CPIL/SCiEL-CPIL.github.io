"""Model-and-optimize: exploit scale invariance + nonnegativity to drive the
objective to its positive-sum regime minimum (a two-coordinate spike), then
polish with a small negative tail to shave the value slightly below the bound.
Uses FFT-vectorized gradient for fast L-BFGS-B, plus coordinate-descent refinement."""
import numpy as np
import time
from scipy.optimize import minimize

N = 400
DEADLINE = 148  # wall-clock seconds; evaluator kills at 150s, keep small margin


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


def _spike_candidate():
    """Deterministic high-quality candidate: two-spike + small negative tail.

    h[0]=0.5, h[1]=0.5, h[2:K] = -eps.  sum = 1 - (K-2)*eps (slightly < 1),
    max_conv = a^2+(1-a)^2 ~ 0.5 (diagonal dominates, off-diag = 0.25).
    value ~= 2*N*0.5 / (1-(K-2)eps)^2, which is < 2*N*0.5 = 400 for eps>0.
    """
    a = 0.5
    K = 200
    eps = 0.001
    h = np.zeros(N)
    h[0] = a
    h[1] = 1.0 - a
    h[2:K] = -eps
    return h


def _spectral_candidate(seed, n_sines=5, freq_range=(1, 20), amp_range=(0.01, 0.5)):
    """Generate a candidate as a sum of sine waves with random frequencies/amplitudes,
    then enforce zero-mean (subtract mean) so the denominator is decoupled from peak.
    This creates oscillatory patterns with negative lobes that can reduce max autoconv."""
    rng = np.random.RandomState(seed)
    t = np.linspace(0, 2 * np.pi, N, endpoint=False)
    h = np.zeros(N)
    for _ in range(n_sines):
        freq = rng.uniform(*freq_range)
        amp = rng.uniform(*amp_range)
        phase = rng.uniform(0, 2 * np.pi)
        h += amp * np.sin(freq * t + phase)
    # Enforce zero-sum to decouple denominator from peak magnitude
    h -= np.mean(h)
    # Scale to have unit sum for stable optimization
    s = np.sum(h)
    if abs(s) > 1e-12:
        h /= s
    return h


def _smooth_objective(x, beta):
    """Smooth (soft-max) surrogate for 2*N*max(conv)/sum^2, sum>0 regime.
    Uses FFT for the convolution."""
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
    # Full auto-convolution via FFT: conv = ifft(fft(x, 2N-1) * fft(x, 2N-1))
    nfft = 2 * N - 1
    X = np.fft.rfft(x, n=nfft)
    conv = np.fft.irfft(X * X, n=nfft)[:2 * N - 1]
    m = float(np.max(conv))
    w = np.exp(beta * (conv - m))
    w_sum = float(np.sum(w))
    w /= w_sum
    # grad_lse[i] = 2 * sum_j w[i+j] * x[j] for j in [0, N-1], i in [0, N-1]
    # This is a cross-correlation: grad_lse = 2 * correlate(w, x)
    # correlate(w, x)[i] = sum_j w[i+j] * x[j]
    # Use FFT: correlate(w, x) = ifft(fft(w, 2N-1) * conj(fft(x, 2N-1)))
    W = np.fft.rfft(w, n=nfft)
    grad_lse = 2.0 * np.fft.irfft(W * np.conj(X), n=nfft)[:N]
    lse = m + np.log(w_sum) / beta
    f = 2.0 * N * lse / (s * s)
    df = 2.0 * N * grad_lse / (s * s) - 4.0 * N * lse / (s * s * s)
    return df


def construct():
    start = time.time()

    # Candidate pool: deterministic spike variants + a few structured starts.
    candidates = [_spike_candidate()]

    # alternate-spike (h[0]=1, rest 0): value exactly 2*N = 800, valid fallback
    alt = np.zeros(N); alt[0] = 1.0
    candidates.append(alt)

    # single spike with tail of varying strength (explore the shave direction)
    for K in (50, 100, 200, 300, 399):
        for eps in (0.0005, 0.001, 0.002, 0.005):
            if K - 2 <= 0:
                continue
            h = np.zeros(N)
            h[0] = 0.5
            h[1] = 0.5
            h[2:K] = -eps
            candidates.append(h)

    # two-spike with unequal split (a, 1-a) - should land back near 0.5
    for a in (0.3, 0.4, 0.5, 0.6, 0.7):
        h = np.zeros(N)
        h[0] = a
        h[1] = 1.0 - a
        h[2:120] = -0.001
        candidates.append(h)

    # structured: positive spike on a short block, small negative elsewhere
    rng = np.random.RandomState(0)
    for _ in range(12):
        h = np.zeros(N)
        h[0] = 0.5
        h[1] = 0.5
        neg_mask = np.zeros(N, dtype=bool)
        neg_mask[2:] = True
        idx = np.where(neg_mask)[0]
        pick = rng.choice(idx, size=rng.randint(20, 300), replace=False)
        h[pick] = -rng.uniform(0.0003, 0.004)
        candidates.append(h)

    # spectral (sine-wave) candidates: oscillatory patterns with negative lobes
    for seed in range(20):
        for n_sines in (3, 5, 8):
            candidates.append(_spectral_candidate(seed, n_sines=n_sines))
    # Chebyshev polynomial based candidates (alternating signs)
    for degree in (5, 8, 12):
        x = np.linspace(-1, 1, N)
        h = np.polynomial.chebyshev.chebval(x, np.array([1.0] + [0.0] * degree))
        h -= np.mean(h)
        s = np.sum(h)
        if abs(s) > 1e-12:
            h /= s
        candidates.append(h)

    # Evaluate all candidates, keep the best.
    best_h = None
    best_value = np.inf
    for h in candidates:
        val, valid = _evaluate(h)
        if valid and val < best_value:
            best_value = val
            best_h = h.copy()

    # Guard: ensure we always have a valid finite candidate.
    if best_h is None or not np.isfinite(best_value):
        best_h = _spike_candidate()
        best_value, _ = _evaluate(best_h)

    # L-BFGS-B polish from the best candidate to squeeze a bit more.
    # With FFT-vectorized gradient, we can afford more iterations and more beta stages.
    if time.time() - start < DEADLINE - 8:
        for beta in (40.0, 80.0, 160.0, 320.0, 640.0, 1280.0):
            if time.time() - start > DEADLINE - 4:
                break
            try:
                res = minimize(
                    lambda x, b=beta: _smooth_objective(x, b),
                    best_h.copy(),
                    jac=lambda x, b=beta: _smooth_grad(x, b),
                    method='L-BFGS-B',
                    options={'maxiter': 800, 'maxfun': 2000, 'gtol': 1e-11},
                )
                val, valid = _evaluate(res.x)
                if valid and val < best_value:
                    best_value = val
                    best_h = res.x.copy()
            except Exception:
                pass

    # Coordinate-descent-style local refinement: perturb individual coordinates
    # to reduce the max auto-convolution peak.
    if time.time() - start < DEADLINE - 3:
        rng2 = np.random.RandomState(1)
        # First pass: random perturbations
        for _ in range(150):
            if time.time() - start > DEADLINE - 3:
                break
            h = best_h + rng2.normal(0.0, 0.005, N)
            val, valid = _evaluate(h)
            if valid and val < best_value:
                best_value = val
                best_h = h.copy()
        # Second pass: targeted coordinate adjustments on the spike region
        for _ in range(200):
            if time.time() - start > DEADLINE - 3:
                break
            h = best_h.copy()
            # Perturb coordinates in the first 50 positions more aggressively
            idx = rng2.randint(0, 50)
            h[idx] += rng2.uniform(-0.02, 0.02)
            # Also perturb some negative tail coordinates
            if idx < 50:
                idx2 = rng2.randint(50, N)
                h[idx2] += rng2.uniform(-0.005, 0.005)
            val, valid = _evaluate(h)
            if valid and val < best_value:
                best_value = val
                best_h = h.copy()

    # Final high-precision polish: very tight tolerance, more iterations
    if time.time() - start < DEADLINE - 1:
        try:
            res = minimize(
                lambda x: _smooth_objective(x, 2000.0),
                best_h.copy(),
                jac=lambda x: _smooth_grad(x, 2000.0),
                method='L-BFGS-B',
                options={'maxiter': 2000, 'maxfun': 5000, 'gtol': 1e-13},
            )
            val, valid = _evaluate(res.x)
            if valid and val < best_value:
                best_value = val
                best_h = res.x.copy()
        except Exception:
            pass

    # Final validity guard: never return an invalid vector.
    val, valid = _evaluate(best_h)
    if not valid or not np.isfinite(val):
        best_h = _spike_candidate()
        val, valid = _evaluate(best_h)
    if not valid or not np.isfinite(val):
        best_h = np.zeros(N)
        best_h[0] = 1.0  # value = 2*N = 800, always valid

    return best_h
