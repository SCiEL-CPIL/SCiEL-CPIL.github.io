"""Model-and-optimize: exploit scale invariance + nonnegativity to drive the
objective to its positive-sum regime minimum (a two-coordinate spike), then
polish with a small negative tail to shave the value slightly below the bound.
Adds spectral (sine-wave) initializations and a final high-precision polish."""
import numpy as np
import time
from scipy.optimize import minimize
from scipy.signal import fftconvolve

N = 400
DEADLINE = 145  # wall-clock seconds; evaluator kills at 150s, keep margin

# Precompute FFT size for fast convolution
_FFT_SIZE = 2 * N - 1
_FFT_NP = 1 << (int(np.log2(_FFT_SIZE)) + 1)  # next power of 2


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
    """Deterministic high-quality candidate: two-spike + small negative tail."""
    a = 0.5
    K = 200
    eps = 0.001
    h = np.zeros(N)
    h[0] = a
    h[1] = 1.0 - a
    h[2:K] = -eps
    return h


def _spectral_candidate(seed, n_sines=5, freq_range=(1, 20), amp_range=(0.01, 0.5)):
    """Generate a candidate as a sum of sine waves with random frequencies/amplitudes."""
    rng = np.random.RandomState(seed)
    t = np.linspace(0, 2 * np.pi, N, endpoint=False)
    h = np.zeros(N)
    for _ in range(n_sines):
        freq = rng.uniform(*freq_range)
        amp = rng.uniform(*amp_range)
        phase = rng.uniform(0, 2 * np.pi)
        h += amp * np.sin(freq * t + phase)
    h -= np.mean(h)
    s = np.sum(h)
    if abs(s) > 1e-12:
        h /= s
    return h


def _alternating_candidate(period, amplitude_ratio=0.5):
    """Create an alternating-sign pattern with a large positive DC offset.
    
    The idea: if h has a large positive mean but small variance, then
    sum(h) is large, but the autoconv peak (which includes sum^2 at lag 0
    plus cross terms) can be kept moderate. However, we want the MAX of
    autoconv to be as small as possible relative to sum^2.
    
    For a constant h = c, autoconv[k] = (N-|k|)*c^2, max = N*c^2, sum = N*c,
    so value = 2*N*N*c^2/(N*c)^2 = 2. That's the seed value.
    
    To beat this, we need max(autoconv) < sum^2/N. Since sum^2 = (sum h_i)^2
    and max autoconv >= h_i^2 for all i, we need all |h_i| small relative to sum.
    This suggests making h_i all positive and nearly equal, but that gives value=2.
    
    Alternative: make some h_i negative so that cross terms in autoconv cancel.
    """
    h = np.zeros(N)
    # Alternating pattern: +a, +b, -a, -b, ...
    for i in range(N):
        if i % period == 0 or (period > 2 and i % period == 1):
            h[i] = 1.0
        else:
            h[i] = -amplitude_ratio
    return h


def _smooth_objective(x, beta):
    """Smooth (soft-max) surrogate for 2*N*max(conv)/sum^2, sum>0 regime."""
    s = float(np.sum(x))
    if abs(s) < 1e-10:
        return 1e10
    conv = fftconvolve(x, x, mode='full')
    m = float(np.max(conv))
    shifted = conv - m
    log_sum_exp = m + np.log(np.sum(np.exp(beta * shifted))) / beta
    val = 2.0 * N * log_sum_exp / (s * s)
    if not np.isfinite(val):
        return 1e10
    return float(val)


def _smooth_grad(x, beta):
    """Gradient of the smooth objective using FFT-vectorized computation."""
    s = float(np.sum(x))
    if abs(s) < 1e-10:
        return np.zeros_like(x)
    conv = fftconvolve(x, x, mode='full')
    m = float(np.max(conv))
    w = np.exp(beta * (conv - m))
    w_sum = np.sum(w)
    w /= w_sum
    # Gradient of log-sum-exp w.r.t. x[i]:
    # d/dx[i] log(sum_k exp(beta*conv[k])) = beta * sum_k w[k] * d conv[k]/dx[i]
    # conv[k] = sum_j x[j]*x[k-j], so d conv[k]/dx[i] = x[k-i] + x[i-k] (with bounds)
    # Using FFT: the gradient can be computed as beta * (w * x) convolved appropriately
    # More directly: grad_lse[i] = 2 * sum_j w[i+j] * x[j] for the relevant range
    # Let's use the full correlation: grad[i] = 2 * (w ⊛ x)[i] where ⊛ is cross-correlation
    # Actually, d conv[k]/dx[i] = x[k-i] + x[i-k] when both indices are valid
    # So grad_lse = 2 * (w * x) in the appropriate sense. Let me compute it directly:
    # grad_lse[i] = 2 * sum_{j: 0<=j<N, 0<=i+j<2N-1} w[i+j] * x[j]
    # This is a cross-correlation of w and x, but w has length 2N-1.
    # Use FFT for speed:
    W = np.fft.rfft(w, n=_FFT_NP)
    X = np.fft.rfft(x, n=_FFT_NP)
    # cross-correlation: ifft(conj(W) * X) gives sum_j w[j+i] * x[j] (circular)
    # We need linear, so pad appropriately. w has len 2N-1, x has len N.
    # Let's just do it with numpy correlate for correctness, it's fast enough for N=400
    # grad_lse[i] = 2 * sum_j w[i+j] * x[j] for j in [0, N-1], i+j in [0, 2N-2]
    # This is equivalent to: for each i, take w[i:i+N] dot x, where w is zero-padded
    w_padded = np.zeros(2 * N - 1)
    w_padded[:len(w)] = w
    # Use stride tricks or just loop (N=400 is small)
    grad_lse = np.zeros(N)
    for i in range(N):
        grad_lse[i] = 2.0 * float(np.dot(w_padded[i:i + N], x))
    lse = m + np.log(w_sum) / beta
    f = 2.0 * N * lse / (s * s)
    df = 2.0 * N * grad_lse / (s * s) - 4.0 * N * lse / (s * s * s)
    return df


def _coordinate_descent(h, max_iters=5000, tol=1e-10, deadline=None):
    """Coordinate descent on the exact objective.
    
    For each coordinate i, find the optimal h[i] given all others fixed.
    The objective is: abs(2*N*max(conv)/sum^2)
    
    This is tricky because of the max and abs. Instead, we'll use the smooth
    surrogate and do gradient-based coordinate updates.
    """
    best_h = h.copy()
    best_val, _ = _evaluate(best_h)
    
    for it in range(max_iters):
        if deadline is not None and time.time() > deadline:
            break
        improved = False
        for i in range(N):
            if deadline is not None and time.time() > deadline:
                break
            # Try a small perturbation in coordinate i
            for delta in (-0.01, -0.001, 0.001, 0.01):
                h_trial = best_h.copy()
                h_trial[i] += delta
                val, valid = _evaluate(h_trial)
                if valid and val < best_val - 1e-12:
                    best_h = h_trial
                    best_val = val
                    improved = True
                    break
        if not improved:
            break
    return best_h, best_val


def _targeted_search(h_init, n_samples=1000, deadline=None):
    """Targeted search: perturb specific coordinates based on gradient structure.
    
    The key insight: the max autoconv coefficient is at some lag k*. 
    To reduce it, we should adjust coordinates that contribute most to conv[k*].
    """
    best_h = h_init.copy()
    best_val, _ = _evaluate(best_h)
    rng = np.random.RandomState(42)
    
    for it in range(n_samples):
        if deadline is not None and time.time() > deadline:
            break
        # Compute which lag has the max
        conv = fftconvolve(best_h, best_h, mode='full')
        k_max = int(np.argmax(conv))
        
        # The coordinates that contribute to conv[k_max] are pairs (i, j) with i+j = k_max
        # Perturb those coordinates
        candidates = []
        for i in range(N):
            j = k_max - i
            if 0 <= j < N:
                candidates.append(i)
                candidates.append(j)
        
        if len(candidates) > 0:
            idx = rng.choice(candidates, size=min(5, len(candidates)), replace=False)
            h_trial = best_h.copy()
            for i in idx:
                h_trial[i] += rng.normal(0, 0.05)
            val, valid = _evaluate(h_trial)
            if valid and val < best_val:
                best_h = h_trial
                best_val = val
    
    return best_h, best_val


def construct():
    start = time.time()
    deadline = start + DEADLINE

    # Candidate pool: deterministic spike variants + structured starts
    candidates = [_spike_candidate()]

    # alternate-spike (h[0]=1, rest 0): value exactly 2*N = 800, valid fallback
    alt = np.zeros(N); alt[0] = 1.0
    candidates.append(alt)

    # single spike with tail of varying strength
    for K in (50, 100, 200, 300, 399):
        for eps in (0.0005, 0.001, 0.002, 0.005):
            if K - 2 <= 0:
                continue
            h = np.zeros(N)
            h[0] = 0.5
            h[1] = 0.5
            h[2:K] = -eps
            candidates.append(h)

    # two-spike with unequal split
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

    # spectral (sine-wave) candidates
    for seed in range(20):
        for n_sines in (3, 5, 8):
            candidates.append(_spectral_candidate(seed, n_sines=n_sines))

    # Chebyshev polynomial based candidates
    for degree in (5, 8, 12):
        x = np.linspace(-1, 1, N)
        h = np.polynomial.chebyshev.chebval(x, np.array([1.0] + [0.0] * degree))
        h -= np.mean(h)
        s = np.sum(h)
        if abs(s) > 1e-12:
            h /= s
        candidates.append(h)

    # Alternating sign candidates (new structural family)
    for period in (2, 3, 4, 5, 8):
        for ratio in (0.1, 0.3, 0.5, 0.7):
            candidates.append(_alternating_candidate(period, ratio))

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

    # L-BFGS-B polish from the best candidate
    if time.time() < deadline - 20:
        for beta in (60.0, 120.0, 200.0, 400.0):
            if time.time() > deadline - 15:
                break
            try:
                res = minimize(
                    lambda x, b=beta: _smooth_objective(x, b),
                    best_h.copy(),
                    jac=lambda x, b=beta: _smooth_grad(x, b),
                    method='L-BFGS-B',
                    options={'maxiter': 600, 'maxfun': 1200, 'gtol': 1e-10},
                )
                val, valid = _evaluate(res.x)
                if valid and val < best_value:
                    best_value = val
                    best_h = res.x.copy()
            except Exception:
                pass

    # Targeted search: focus on the lag that has the max autoconv
    if time.time() < deadline - 10:
        best_h, best_value = _targeted_search(best_h, n_samples=2000, deadline=deadline - 10)

    # Coordinate descent polish
    if time.time() < deadline - 5:
        best_h, best_value = _coordinate_descent(best_h, max_iters=2000, deadline=deadline - 5)

    # Final high-precision polish
    if time.time() < deadline - 2:
        try:
            res = minimize(
                lambda x: _smooth_objective(x, 800.0),
                best_h.copy(),
                jac=lambda x: _smooth_grad(x, 800.0),
                method='L-BFGS-B',
                options={'maxiter': 1000, 'maxfun': 2000, 'gtol': 1e-12},
            )
            val, valid = _evaluate(res.x)
            if valid and val < best_value:
                best_value = val
                best_h = res.x.copy()
        except Exception:
            pass

    # Final validity guard
    val, valid = _evaluate(best_h)
    if not valid or not np.isfinite(val):
        best_h = _spike_candidate()
        val, valid = _evaluate(best_h)
    if not valid or not np.isfinite(val):
        best_h = np.zeros(N)
        best_h[0] = 1.0

    return best_h
