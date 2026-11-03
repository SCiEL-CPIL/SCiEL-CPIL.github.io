import numpy as np
from scipy.optimize import minimize_scalar, differential_evolution
import time

N = 600

# EVOLVE-BLOCK-START
def _score_from_heights(h):
    """Compute the normalized maximum of the autoconvolution.
    
    The step function has N equal intervals on [-1/4, 1/4].
    Interval width = (1/4 - (-1/4)) / N = 1/(2N).
    
    Autoconvolution at lag k (in interval units):
      c[k] = sum_i h[i] * h[i+k] * (interval_width)
    
    The "normalized maximum" is max(c) / (integral of f)^2 or similar.
    Let's think about what C1 is.
    
    For a step function f on [-1/4, 1/4] with N intervals of width w = 1/(2N),
    f(x) = h[i] for x in [-1/4 + i*w, -1/4 + (i+1)*w).
    
    The autoconvolution is (f * f)(t) = integral f(s) f(t-s) ds.
    
    For a step function, this is piecewise linear. The maximum of the autoconvolution
    occurs at some point. The "normalized maximum" likely means max(f*f) / (||f||_1)^2
    or max(f*f) / (||f||_2)^2.
    
    Actually, looking at the task: "minimize the normalized maximum of its autoconvolution; this gives an upper bound on C1."
    
    C1 is likely related to the first autocorrelation inequality. A common form:
    For a nonnegative function f, the maximum of (f*f)(t) / (||f||_1^2) is bounded by C1.
    
    Let's define:
    - w = 1/(2*N)
    - ||f||_1 = sum(h) * w
    - autoconvolution at discrete lags: for lag j (in units of w), 
      (f*f)(j*w) ≈ sum_i h[i] * h[i+j] * w  (this is an approximation)
    
    Actually, for step functions, the autoconvolution is exactly computable.
    Let's use FFT for efficiency.
    
    The normalized maximum = max_t (f*f)(t) / (||f||_1)^2
    
    We want to minimize this ratio.
    """
    w = 1.0 / (2.0 * N)
    integral = np.sum(h) * w
    if integral <= 0:
        return np.inf
    
    # Compute autoconvolution using FFT
    # The autoconvolution of a discrete sequence with itself
    # For step functions, the continuous autoconvolution at multiples of w:
    # (f*f)(k*w) = w * sum_i h[i] * h[i+k]  for k >= 0 (with appropriate bounds)
    # But we need the max over all t, not just multiples of w.
    
    # For a step function, (f*f)(t) is piecewise linear. The maximum occurs either
    # at a multiple of w or at a point where the derivative changes sign within an interval.
    # For simplicity and speed, let's first compute at discrete lags and then refine.
    
    # Use FFT to compute the discrete autoconvolution
    h = np.asarray(h, dtype=np.float64)
    # Zero-pad for linear convolution
    size = 2 * N
    fhat = np.fft.rfft(h, n=size)
    conv = np.fft.irfft(fhat * fhat, n=size)
    # conv[k] = sum_i h[i] * h[i+k] for k = 0, 1, ..., N-1 (and symmetric)
    # The actual autoconvolution value at t = k*w is: w * conv[k]
    # But we need the max over all t. The piecewise linear function has its max
    # at one of the breakpoints or at a critical point.
    
    # For now, use the discrete max as an approximation
    max_conv = np.max(conv) * w
    normalized = max_conv / (integral * integral)
    
    return normalized


def _score_from_heights_fast(h):
    """Faster scoring using FFT."""
    h = np.asarray(h, dtype=np.float64)
    integral = np.sum(h)
    if integral <= 0:
        return np.inf
    
    # Autoconvolution via FFT
    size = 2 * N
    fhat = np.fft.rfft(h, n=size)
    conv = np.fft.irfft(fhat * fhat, n=size)
    
    w = 1.0 / (2.0 * N)
    max_conv = np.max(conv) * w
    normalized = max_conv / (integral * integral)
    
    return normalized


def _random_nonnegative(n=N):
    """Generate a random nonnegative step function."""
    # Mix of different structures
    r = np.random.random()
    if r < 0.3:
        # Constant
        return np.ones(n) * np.random.random()
    elif r < 0.6:
        # Smooth bump
        x = np.linspace(0, 1, n)
        center = np.random.random()
        width = 0.1 + 0.4 * np.random.random()
        h = np.exp(-((x - center) ** 2) / (2 * width ** 2))
        return h / np.max(h) if np.max(h) > 0 else h
    elif r < 0.8:
        # Triangular
        x = np.linspace(0, 1, n)
        center = np.random.random()
        h = np.maximum(0, 1 - np.abs(x - center) / 0.5)
        return h
    else:
        # Random smooth
        h = np.random.random(n)
        # Smooth it
        kernel = np.ones(10) / 10
        h = np.convolve(h, kernel, mode='same')
        return np.maximum(0, h)


def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 140s deadline with 10s margin
    
    best_score = np.inf
    best_h = np.ones(N)
    
    # Initial random search
    n_random = 50
    for i in range(n_random):
        if time.time() > deadline:
            break
        h = _random_nonnegative()
        s = _score_from_heights_fast(h)
        if s < best_score:
            best_score = s
            best_h = h.copy()
    
    # Local optimization using scipy
    # Use a parameterization that ensures nonnegativity: h = exp(x)
    # But this can be ill-conditioned. Instead, use h = x^2 or softplus.
    
    # Let's use a few different optimization strategies
    
    # Strategy 1: Minimize with respect to a few key parameters
    # Try optimizing a Gaussian-like shape
    
    def _gaussian_param(params):
        center, width, height = params
        x = np.linspace(0, 1, N)
        h = height * np.exp(-((x - center) ** 2) / (2 * width ** 2))
        return h
    
    def _gaussian_objective(params):
        h = _gaussian_param(params)
        return _score_from_heights_fast(h)
    
    # Try several initial guesses for the Gaussian
    for center in [0.3, 0.4, 0.5, 0.6, 0.7]:
        for width in [0.1, 0.2, 0.3, 0.4]:
            if time.time() > deadline:
                break
            params0 = np.array([center, width, 1.0])
            try:
                result = minimize_scalar(
                    lambda w: _gaussian_objective([center, w, 1.0]),
                    bounds=(0.01, 0.5),
                    method='bounded',
                    options={'xatol': 1e-4}
                )
                if result.success:
                    h = _gaussian_param([center, result.x, 1.0])
                    s = _score_from_heights_fast(h)
                    if s < best_score:
                        best_score = s
                        best_h = h.copy()
            except:
                pass
    
    # Strategy 2: Direct coordinate descent on the heights
    # Normalize so that sum(h) = 1 (scaling doesn't change the ratio)
    h_work = best_h.copy()
    h_work = h_work / np.sum(h_work)
    
    # Coordinate descent with line search
    n_cd_iterations = 20
    for it in range(n_cd_iterations):
        if time.time() > deadline:
            break
        improved = False
        for i in range(0, N, 10):  # Skip some coordinates for speed
            if time.time() > deadline:
                break
            old_val = h_work[i]
            # Try a few perturbations
            for delta in [-0.5, -0.2, 0.0, 0.2, 0.5]:
                new_val = max(0, old_val + delta * 0.1)
                h_work[i] = new_val
                s = _score_from_heights_fast(h_work)
                if s < best_score:
                    best_score = s
                    best_h = h_work.copy()
                    improved = True
                else:
                    h_work[i] = old_val
                    break
            if not improved:
                h_work[i] = old_val
    
    # Strategy 3: Use differential evolution on a reduced parameter set
    # Represent h as a combination of basis functions
    
    n_basis = 20
    # Create basis functions
    x = np.linspace(0, 1, N)
    basis = np.zeros((n_basis, N))
    for j in range(n_basis):
        basis[j] = np.exp(-((x - j / (n_basis - 1)) ** 2) / (2 * 0.1 ** 2))
    
    def _basis_objective(coeffs):
        h = basis @ coeffs
        h = np.maximum(0, h)
        if np.sum(h) <= 0:
            return 1e10
        return _score_from_heights_fast(h)
    
    # Initialize with current best
    coeffs0 = np.zeros(n_basis)
    # Project best_h onto basis
    coeffs0 = np.linalg.lstsq(basis, best_h / np.sum(best_h), rcond=None)[0]
    
    if time.time() < deadline - 20:
        try:
            bounds = [(0, 1)] * n_basis
            result = differential_evolution(
                _basis_objective,
                bounds=bounds,
                seed=42,
                maxiter=50,
                tol=1e-3,
                polish=True,
                workers=1
            )
            if result.fun < best_score:
                h = basis @ result.x
                h = np.maximum(0, h)
                best_score = result.fun
                best_h = h.copy()
        except:
            pass
    
    # Final: ensure nonnegativity and return
    best_h = np.maximum(0, best_h)
    return list(best_h)
# EVOLVE-BLOCK-END
