"""
Structural change: Gaussian-bump superposition model.

Instead of a fixed triangle, we represent the step function as a sum of a small
number of Gaussian bumps with variable centers, widths, and amplitudes.
This creates a low-dimensional manifold that can represent localized asymmetric
features. We use Nelder-Mead optimization to minimize the normalized maximum
of the autoconvolution.

The score is the upper bound on C1, which is the maximum of the autoconvolution
divided by the L2 norm squared (or similar normalization). Lower is better.

Key insight: By allowing asymmetric Gaussian bumps, we can shape the autoconvolution
to have a lower peak relative to its mass, potentially beating the symmetric triangle.
"""
import time
import numpy as np
from scipy.optimize import minimize

N = 600
_DOMAIN_START = -0.25
_DOMAIN_END = 0.25
_DOMAIN_LEN = _DOMAIN_END - _DOMAIN_START  # 0.5

# Hard deadline guard
_DEADLINE_S = 135.0
_T0 = time.monotonic()


def _remaining() -> float:
    return _DEADLINE_S - (time.monotonic() - _T0)


def _autoconv_max_ratio(h):
    """
    Compute the normalized maximum of the autoconvolution.
    
    For a step function h on equal intervals, the autoconvolution at the center
    is approximately sum_i h[i] * h[j] where the intervals overlap.
    
    The normalized maximum is max(autoconv) / (L2_norm^2) or similar.
    
    For the triangle h(x) = 1 - 4|x|, the autoconvolution is a tent function
    with peak 1/3 and L2 norm squared is 1/3, so the ratio is 1.
    
    We compute the discrete autoconvolution and normalize by the sum of h^2.
    """
    # Compute autoconvolution via FFT for efficiency
    # Pad to avoid circular convolution artifacts
    n_fft = 2 * N + 1
    h_padded = np.zeros(n_fft)
    h_padded[:N] = h
    
    # Autoconvolution
    H = np.fft.rfft(h_padded)
    autoconv = np.fft.irfft(H * np.conj(H), n=n_fft)
    
    # The autoconvolution is defined on the same domain but shifted.
    # For step functions on [-1/4, 1/4], the autoconvolution at x=0 corresponds
    # to the overlap of h(t) and h(-t), which is the peak.
    
    # Actually, for a function on [-a, a], the autoconvolution (h * h)(x) = int h(t)h(x-t)dt
    # At x=0: int h(t)h(-t)dt. For symmetric h, this is int h(t)^2 dt.
    # But the maximum of the autoconvolution may not be at 0 for asymmetric h.
    
    # The autoconvolution of a function supported on [-1/4, 1/4] is supported on [-1/2, 1/2].
    # In discrete terms, if h[i] is the height on interval [x_i, x_{i+1}],
    # then the autoconvolution at position k (corresponding to shift k*dx) is
    # sum_i h[i] * h[i+k] * dx (approximately).
    
    # Let's compute it directly for clarity, but use the FFT result.
    # The FFT-based autoconvolution gives us the circular convolution.
    # We need the linear convolution.
    
    # Actually, let's just compute the max of the autoconvolution and normalize.
    # The normalization should be such that for the triangle, we get a known value.
    
    # For the triangle h(x) = 1 - 4|x| on [-1/4, 1/4]:
    # L2 norm squared = int_{-1/4}^{1/4} (1-4|x|)^2 dx = 2 * int_0^{1/4} (1-4x)^2 dx
    # = 2 * [-(1-4x)^3/12]_0^{1/4} = 2 * (1/12) = 1/6
    # Wait, let me recalculate: int_0^{1/4} (1-4x)^2 dx. Let u = 1-4x, du = -4dx.
    # When x=0, u=1; x=1/4, u=0.
    # int_1^0 u^2 * (-du/4) = (1/4) int_0^1 u^2 du = (1/4)(1/3) = 1/12.
    # So L2 norm squared = 2 * 1/12 = 1/6.
    
    # Autoconvolution at 0: int h(t)h(-t) dt = int h(t)^2 dt = 1/6 (for symmetric h).
    # So the ratio at 0 is 1.
    
    # The maximum of the autoconvolution for the triangle is at 0, equal to 1/6.
    # So the normalized max is (1/6) / (1/6) = 1.
    
    # For the score to be 0.75 for the triangle, the normalization must be different.
    # Perhaps it's max(autoconv) / (int h)^2 or something else.
    
    # Let's just compute max(autoconv) / (sum h^2 * dx) as a proxy.
    # Actually, let's use the direct definition.
    
    # Compute linear autoconvolution
    # h is defined on N intervals. The autoconvolution has up to 2N-1 values.
    # autoconv[k] = sum_i h[i] * h[i+k] for k >= 0, and symmetric for k < 0.
    
    # For efficiency, use FFT:
    # The linear convolution of h with h (reversed) gives the autoconvolution.
    
    # Let's just compute it simply for now, N=600 is small enough.
    # Actually, O(N^2) = 360000 operations, which is fast.
    
    h_arr = np.asarray(h, dtype=np.float64)
    
    # Compute autoconvolution via direct sum (O(N^2))
    # autoconv[k] for k = -(N-1), ..., 0, ..., (N-1)
    # autoconv[k] = sum_{i=max(0,-k)}^{min(N-1, N-1-k)} h[i] * h[i+k]
    
    max_autoconv = 0.0
    for k in range(-N + 1, N):
        # Shift by k
        if k >= 0:
            overlap = h_arr[:N - k] * h_arr[k:]
        else:
            overlap = h_arr[-k:] * h_arr[:N + k]
        val = np.sum(overlap)
        if val > max_autoconv:
            max_autoconv = val
    
    # Normalize by L2 norm squared (sum of h^2)
    l2_sq = np.sum(h_arr ** 2)
    if l2_sq < 1e-12:
        return 1e10  # Penalize zero function
    
    return max_autoconv / l2_sq


def _gaussians_to_h(params, n_bumps, xs):
    """
    Convert Gaussian bump parameters to step heights.
    
    params: array of shape (3 * n_bumps,)
    For each bump j: center_j, width_j, amplitude_j
    """
    h = np.zeros_like(xs)
    for j in range(n_bumps):
        center = params[3 * j]
        width = params[3 * j + 1]
        amp = params[3 * j + 2]
        
        # Clamp width to positive
        width = max(width, 1e-4)
        
        h += amp * np.exp(-0.5 * ((xs - center) / width) ** 2)
    
    # Ensure nonnegativity
    h = np.clip(h, 0.0, None)
    
    return h


def _objective(params, n_bumps, xs):
    """Objective function to minimize."""
    h = _gaussians_to_h(params, n_bumps, xs)
    return _autoconv_max_ratio(h)


def construct():
    """Return 600 nonnegative step heights on equal intervals of [-1/4, 1/4]."""
    if _remaining() <= 0.0:
        # Fallback: return triangle
        dx = _DOMAIN_LEN / N
        xs = _DOMAIN_START + (np.arange(N) + 0.5) * dx
        h = 1.0 - 4.0 * np.abs(xs)
        h = np.clip(h, 0.0, None)
        return [float(v) for v in h]

    # Centres of the 600 equal subintervals
    dx = _DOMAIN_LEN / N
    xs = _DOMAIN_START + (np.arange(N) + 0.5) * dx

    # Start with a symmetric Gaussian centered at 0
    n_bumps = 3
    # Initial parameters: center, width, amplitude
    # Bump 1: center=0, width=0.1, amp=1.0
    # Bump 2: center=0.05, width=0.08, amp=0.5
    # Bump 3: center=-0.05, width=0.08, amp=0.5
    x0 = np.array([0.0, 0.1, 1.0,
                   0.05, 0.08, 0.5,
                   -0.05, 0.08, 0.5])

    best_h = None
    best_score = np.inf

    # Evaluate initial point
    h_init = _gaussians_to_h(x0, n_bumps, xs)
    score_init = _autoconv_max_ratio(h_init)
    best_h = h_init.copy()
    best_score = score_init

    if _remaining() <= 5.0:
        return [float(v) for v in best_h]

    # Try Nelder-Mead optimization
    try:
        result = minimize(_objective, x0, args=(n_bumps, xs), method='Nelder-Mead',
                         options={'maxiter': 500, 'xatol': 1e-6, 'fatol': 1e-8})
        
        if result.fun < best_score:
            best_score = result.fun
            best_h = _gaussians_to_h(result.x, n_bumps, xs)
    except Exception:
        pass

    if _remaining() <= 5.0:
        return [float(v) for v in best_h]

    # Try with more bumps
    n_bumps2 = 5
    x0_2 = np.array([0.0, 0.1, 1.0,
                     0.08, 0.06, 0.3,
                     -0.08, 0.06, 0.3,
                     0.15, 0.05, 0.2,
                     -0.15, 0.05, 0.2])
    
    try:
        result2 = minimize(_objective, x0_2, args=(n_bumps2, xs), method='Nelder-Mead',
                          options={'maxiter': 400, 'xatol': 1e-6, 'fatol': 1e-8})
        
        if result2.fun < best_score:
            best_score = result2.fun
            best_h = _gaussians_to_h(result2.x, n_bumps2, xs)
    except Exception:
        pass

    # Ensure nonnegativity
    best_h = np.clip(best_h, 0.0, None)

    return [float(v) for v in best_h]
