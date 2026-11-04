"""
Structural change: Use a spectral-domain parameterization with a smooth
amplitude/phase design, optimized via BFGS on a smooth soft-max objective.
The key insight is that the autoconvolution in the time domain corresponds
to the squared magnitude of the FFT in the frequency domain (for circular
convolution). We can design the spectrum directly to minimize the peak
of the circular autoconvolution, which is related to the squared spectrum.

However, the task uses LINEAR autoconvolution. The relationship is:
linear_conv(x, x)[k] = circular_conv(x_padded, x_padded)[k] where x_padded is
zero-padded. So the spectrum of the zero-padded signal matters.

A different structural approach: represent x as a sum of a few sinusoids
with optimized frequencies, amplitudes, and phases. This gives a very
low-dimensional parameter space that can be explored efficiently.

Actually, let me think about this more carefully. The metric is:
max_{k != 0} |c_k| / S^2 where c = linear autoconvolution of x, S = sum(x).

For a sequence with S != 0, we can normalize: let y = x / S, so sum(y) = 1.
Then the metric becomes max_{k != 0} |c_k(y)|.

So we want a sequence y with sum 1 that minimizes the max absolute value of
its off-diagonal autoconvolution.

A key observation: if y is approximately a delta function, the autoconvolution
is small. But we need sum(y) = 1, so we can't make it too concentrated.

Let me try a different structural approach: use a quadratic/cubic phase
chirp signal. Chirps have well-known properties of having nearly constant
magnitude spectrum, which means their autoconvolution (which is related to
the squared spectrum) is nearly flat.

Actually, for a linear chirp x[n] = exp(i * pi * alpha * n^2 / N), the
autoconvolution has specific properties. Let me use real-valued versions.

Let me try: x[n] = cos(2*pi*alpha*n^2/N + phi) with optimized alpha and phi,
plus a DC offset to ensure the sum is not too small.

Actually, the most promising structural change: use a multi-frequency
superposition with a global optimization (differential evolution or
basinhopping) over the frequency/phase parameters, which is a much lower-
dimensional problem than optimizing 400 variables directly.
"""
import numpy as np
import time
import scipy.optimize

N = 400

def _autoconv_lin(x):
    """Compute linear autoconvolution using FFT."""
    n = len(x)
    size = 1
    while size < 2 * n - 1:
        size *= 2
    fx = np.fft.rfft(x, size)
    return np.fft.irfft(fx * fx, size)[:2 * n - 1]

def _metric_offdiag(x):
    """Metric: max |c_k| for k != 0 / S^2"""
    s = np.sum(x)
    if abs(s) < 1e-12:
        return float('inf')
    c = _autoconv_lin(x)
    if len(c) > 1:
        return float(np.max(np.abs(c[1:]))) / (s * s)
    return 0.0

def _soft_max_abs(c, beta):
    """Smooth approximation to max|c| using log-sum-exp."""
    abs_c = np.abs(c)
    m = np.max(abs_c)
    if m < 1e-15:
        return 0.0
    return m + np.log(np.sum(np.exp(beta * (abs_c - m)))) / beta

def _chirp_sequence(n, alpha, phi, dc_offset=0.0):
    """Generate a real-valued chirp sequence with DC offset."""
    t = np.arange(n, dtype=float)
    x = np.cos(2.0 * np.pi * alpha * t * t / n + phi) + dc_offset
    return x

def _multi_sine_sequence(n, params):
    """
    Generate a sequence as sum of K sinusoids + DC offset.
    params = [dc, a1, f1, p1, a2, f2, p2, ..., aK, fK, pK]
    """
    x = np.zeros(n)
    x[:] = params[0]
    t = np.arange(n, dtype=float)
    for k in range(1, len(params) // 3 + 1):
        a = params[3*k - 2]
        f = params[3*k - 1]
        p = params[3*k]
        x += a * np.sin(2.0 * np.pi * f * t / n + p)
    return x

def _objective_multi_sine(params, n=N, beta=20.0):
    """Objective for multi-sine parameter optimization."""
    x = _multi_sine_sequence(n, params)
    s = np.sum(x)
    if abs(s) < 1e-10:
        return 1e10
    c = _autoconv_lin(x)
    if len(c) <= 1:
        return 0.0
    abs_c = np.abs(c[1:])
    m = np.max(abs_c)
    if m < 1e-15:
        return 0.0
    soft_max = m + np.log(np.sum(np.exp(beta * (abs_c - m)))) / beta
    return soft_max / (s * s)

def _optimize_multi_sine(n=N, K=5, time_budget=60.0):
    """
    Optimize a multi-sine representation using differential evolution.
    """
    start = time.time()
    
    # Parameter vector: [dc, a1, f1, p1, ..., aK, fK, pK]
    n_params = 1 + 3 * K
    
    def bounds():
        b = [(-2.0, 2.0)]  # dc
        for _ in range(K):
            b.append((-1.0, 1.0))  # amplitude
            b.append((0.1, n / 2))  # frequency (cycles over N samples)
            b.append((-np.pi, np.pi))  # phase
        return b
    
    def objective(params):
        if time.time() - start > time_budget:
            return 1e10
        return _objective_multi_sine(params, n=n, beta=20.0)
    
    # Use differential evolution for global search
    try:
        result = scipy.optimize.differential_evolution(
            objective,
            bounds=bounds(),
            seed=42,
            maxiter=200,
            tol=1e-10,
            polish=True,
            update_interval=1,
            workers=-1,
            popsize=15
        )
        if time.time() - start < time_budget:
            best_params = result.x
            best_x = _multi_sine_sequence(n, best_params)
            best_score = _metric_offdiag(best_x)
            return best_x, best_score
    except Exception:
        pass
    
    return None, float('inf')

def _gaussian_window_sequence(n, center=0.5, width=0.1):
    """Gaussian window centered at 'center' fraction of n."""
    t = np.arange(n, dtype=float) / n
    return np.exp(-0.5 * ((t - center) / width) ** 2)

def _optimize_gaussian(n=N, time_budget=20.0):
    """Optimize Gaussian window parameters."""
    start = time.time()
    
    def gauss_obj(params):
        if time.time() - start > time_budget:
            return 1e10
        center, width, dc = params
        x = _gaussian_window_sequence(n, center, width) + dc
        s = np.sum(x)
        if abs(s) < 1e-10:
            return 1e10
        c = _autoconv_lin(x)
        if len(c) <= 1:
            return 0.0
        abs_c = np.abs(c[1:])
        m = np.max(abs_c)
        if m < 1e-15:
            return 0.0
        beta = 20.0
        soft_max = m + np.log(np.sum(np.exp(beta * (abs_c - m)))) / beta
        return soft_max / (s * s)
    
    try:
        result = scipy.optimize.differential_evolution(
            gauss_obj,
            bounds=[(0.2, 0.8), (0.02, 0.5), (-0.5, 0.5)],
            seed=7,
            maxiter=100,
            tol=1e-10,
            polish=True,
            popsize=15
        )
        if time.time() - start < time_budget:
            center, width, dc = result.x
            x = _gaussian_window_sequence(n, center, width) + dc
            score = _metric_offdiag(x)
            return x, score
    except Exception:
        pass
    
    return None, float('inf')

def _spectral_design(n=N):
    """
    Design a sequence by directly specifying its DFT magnitude to be flat
    (maximally spread), then taking the real part. This is a structural
    approach: we design in the frequency domain.
    """
    size = 1
    while size < 2 * n - 1:
        size *= 2
    
    # Start with a random phase spectrum with flat magnitude
    rng = np.random.default_rng(123)
    for trial in range(20):
        # Random phases
        phases = rng.uniform(-np.pi, np.pi, size // 2 + 1)
        # Flat magnitude (all ones) - this gives a sequence with flat spectrum
        mag = np.ones(size // 2 + 1)
        # Ensure Hermitian symmetry for real signal
        spectrum = mag * np.exp(1j * phases)
        spectrum[0] = mag[0]  # DC term real
        if size % 2 == 0:
            spectrum[-1] = mag[-1]  # Nyquist term real
        
        x_full = np.fft.irfft(spectrum, size)
        x = x_full[:n]
        
        # Try different DC offsets
        for dc in [0.0, 0.1, -0.1, 0.5, -0.5]:
            x_test = x + dc
            s = np.sum(x_test)
            if abs(s) < 1e-10:
                continue
            score = _metric_offdiag(x_test)
            if score < 0.5:  # Good enough threshold
                return x_test, score
    
    return None, float('inf')

def _refine_with_lbfgs(x_init, time_budget=30.0, beta=50.0):
    """Refine a candidate using L-BFGS-B on the soft-max objective."""
    start = time.time()
    
    def obj(x):
        if time.time() - start > time_budget:
            return 1e10
        s = np.sum(x)
        if abs(s) < 1e-10:
            return 1e10
        c = _autoconv_lin(x)
        if len(c) <= 1:
            return 0.0
        abs_c = np.abs(c[1:])
        m = np.max(abs_c)
        if m < 1e-15:
            return 0.0
        soft_max = m + np.log(np.sum(np.exp(beta * (abs_c - m)))) / beta
        return soft_max / (s * s)
    
    try:
        result = scipy.optimize.minimize(
            obj,
            x_init,
            method='L-BFGS-B',
            options={'maxiter': 300, 'ftol': 1e-14, 'gtol': 1e-10, 'maxls': 30},
            tol=1e-12
        )
        if time.time() - start < time_budget:
            x_opt = result.x
            score = _metric_offdiag(x_opt)
            return x_opt, score
    except Exception:
        pass
    
    return None, float('inf')

def _gradient_polish(x, max_iters=100, time_budget=20.0):
    """
    Gradient descent polishing using the subgradient of max|c_k|.
    Uses the active set (all lags near the maximum) for a smoother gradient.
    """
    start = time.time()
    x = x.copy()
    best_x = x.copy()
    best_score = _metric_offdiag(x)
    
    for it in range(max_iters):
        if time.time() - start > time_budget:
            break
        
        s = np.sum(x)
        if abs(s) < 1e-12:
            break
        
        c = _autoconv_lin(x)
        abs_c = np.abs(c[1:])
        n_lags = len(abs_c)
        
        # Use top 5% of lags for a smoother gradient
        threshold = np.percentile(abs_c, 95)
        active = abs_c > threshold
        if not np.any(active):
            active = abs_c == np.max(abs_c)
        
        # Gradient: sum over active lags of sign(c_k) * d(c_k)/dx
        grad = np.zeros_like(x)
        n = len(x)
        
        for k_idx in np.where(active)[0]:
            k = k_idx + 1
            sign_c = np.sign(c[k])
            if sign_c == 0:
                sign_c = 1.0
            
            # d(c_k)/dx_j = x_{j+k} + x_{j-k} (with bounds)
            for j in range(n):
                if j + k < n:
                    grad[j] += sign_c * x[j + k]
                if j - k >= 0:
                    grad[j] += sign_c * x[j - k]
        
        gnorm = np.linalg.norm(grad)
        if gnorm < 1e-15:
            break
        
        # Adaptive step size
        step = 0.05 * (1.0 - it / max_iters) + 0.005
        x_new = x - step * grad / gnorm
        
        s_new = np.sum(x_new)
        if abs(s_new) < 1.0:
            x_new += (1.0 - s_new) / n
        
        sc_new = _metric_offdiag(x_new)
        if sc_new < best_score:
            best_score = sc_new
            best_x = x_new.copy()
            x = x_new
        else:
            step *= 0.5
            x_new = x - step * grad / gnorm
            s_new = np.sum(x_new)
            if abs(s_new) < 1.0:
                x_new += (1.0 - s_new) / n
            sc_new = _metric_offdiag(x_new)
            if sc_new < best_score:
                best_score = sc_new
                best_x = x_new.copy()
                x = x_new
    
    return best_x, best_score

def construct():
    """
    Construct a signed step function minimizing max|autoconv| / (sum(x))^2.
    Structural approach: multi-frequency design + spectral design + refinement.
    """
    start_time = time.time()
    deadline = start_time + 140.0
    
    best_x = None
    best_score = float('inf')
    
    def update_best(x, score):
        nonlocal best_x, best_score
        if x is not None and score < best_score:
            best_score = score
            best_x = x.copy()
    
    # Phase 1: Try structural designs (low-dimensional parameter spaces)
    
    # 1a: Multi-sine with 3 harmonics
    if time.time() < deadline - 80.0:
        x_ms, sc_ms = _optimize_multi_sine(N, K=3, time_budget=30.0)
        update_best(x_ms, sc_ms)
    
    # 1b: Multi-sine with 5 harmonics
    if time.time() < deadline - 50.0:
        x_ms2, sc_ms2 = _optimize_multi_sine(N, K=5, time_budget=25.0)
        update_best(x_ms2, sc_ms2)
    
    # 1c: Gaussian window
    if time.time() < deadline - 40.0:
        x_g, sc_g = _optimize_gaussian(N, time_budget=15.0)
        update_best(x_g, sc_g)
    
    # 1d: Spectral design (flat spectrum with random phases)
    if time.time() < deadline - 30.0:
        x_sp, sc_sp = _spectral_design(N)
        update_best(x_sp, sc_sp)
    
    # Phase 2: Try some hand-crafted candidates
    candidates = []
    
    # All ones
    candidates.append(np.ones(N))
    
    # Hanning
    candidates.append(np.hanning(N))
    
    # Raised cosine
    t = np.linspace(0, np.pi, N)
    candidates.append(np.cos(t) ** 2)
    
    # Triangle wave
    tri = 2 * np.abs(2 * (np.arange(N) / N - 0.5)) - 1
    candidates.append(tri)
    
    # Chirp (real part)
    t = np.arange(N, dtype=float)
    candidates.append(np.cos(2 * np.pi * 0.5 * t * t / N))
    
    # Quadratic phase with DC
    candidates.append(np.cos(2 * np.pi * 0.3 * t * t / N) + 0.3)
    
    # Sum of two incommensurate sines
    candidates.append(np.sin(2 * np.pi * 7 * t / N) + np.sin(2 * np.pi * 13 * t / N) + 0.5)
    
    # B-spline-like (cubic)
    from scipy.signal import find_peaks
    # Simple cubic B-spline approximated
    bspline = np.zeros(N)
    for i in range(N):
        u = i / N - 0.5
        bspline[i] = max(0, 1 - abs(u) * 2) ** 3
    candidates.append(bspline)
    
    for x in candidates:
        s = np.sum(x)
        if abs(s) < 1e-10:
            continue
        sc = _metric_offdiag(x)
        update_best(x, sc)
    
    if best_x is None:
        return np.ones(N)
    
    # Phase 3: L-BFGS-B refinement of the best candidate
    if time.time() < deadline - 30.0:
        x_ref, sc_ref = _refine_with_lbfgs(best_x, time_budget=25.0, beta=50.0)
        update_best(x_ref, sc_ref)
    
    # Phase 4: Gradient polish
    if time.time() < deadline - 15.0:
        x_pol, sc_pol = _gradient_polish(best_x, max_iters=100, time_budget=12.0)
        update_best(x_pol, sc_pol)
    
    # Phase 5: Final L-BFGS-B with higher beta for sharper max approximation
    if time.time() < deadline - 5.0:
        x_ref2, sc_ref2 = _refine_with_lbfgs(best_x, time_budget=8.0, beta=100.0)
        update_best(x_ref2, sc_ref2)
    
    return best_x
