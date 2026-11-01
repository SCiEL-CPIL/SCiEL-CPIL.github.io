"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

N = 400

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time

def _compute_autoconv_exact(heights):
    """
    Compute the exact autoconvolution of a signed step function.
    heights[i] is the value on interval [x_i, x_{i+1}] where x_i = -1/4 + i*d, d = 1/(2N)
    
    The autoconvolution (f*f)(t) = integral f(s) f(t-s) ds is piecewise linear.
    We compute it at the breakpoints where it can have kinks.
    
    For step functions on equal intervals, the autoconvolution can be computed
    efficiently. The support is [-1/2, 1/2].
    
    Key insight: for a step function with constant values on N intervals of width d,
    the autoconvolution at lag k*d (for integer k) can be computed from the
    discrete autocorrelation of the heights.
    
    Actually, let's use the fact that:
    (f*f)(t) = sum_i sum_j h_i h_j * overlap(I_i, t - I_j)
    
    where overlap gives the measure of intersection.
    
    For efficiency, we'll compute on a grid of 2N+1 points (the breakpoints).
    """
    d = 1.0 / (2 * N)  # interval width
    
    # The autoconvolution is piecewise linear with breakpoints at multiples of d
    # in the range [-1/2, 1/2], which gives 2N+1 breakpoints
    
    # For a step function, the autoconvolution at t = k*d can be computed as:
    # (f*f)(k*d) = d * sum_i h_i * h_{i-k} (with appropriate indexing)
    # This is d times the discrete autocorrelation at lag k.
    
    # Discrete autocorrelation: R[k] = sum_i h[i] * h[i-k] for k = -(N-1), ..., N-1
    # Using FFT for efficiency
    size = 2 * N
    h_padded = np.zeros(size)
    h_padded[:N] = heights
    
    H = np.fft.fft(h_padded)
    autocorr = np.fft.ifft(H * H).real[:N]
    
    # autocorr[k] = sum_i h[i] * h[i-k] for k = 0, ..., N-1
    # We need the full autocorrelation for lags -(N-1) to N-1
    full_autocorr = np.concatenate([autocorr[0], autocorr[1:][::-1], autocorr[1:]])
    # Actually, let me be more careful.
    
    # R[k] = sum_{i=0}^{N-1-k} h[i] * h[i+k] for k >= 0
    # R[k] = R[-k] for the symmetric case
    
    # Using FFT: if we compute ifft(fft(h)^2), we get circular autocorrelation
    # With zero-padding, we get linear autocorrelation
    
    # Let me redo this properly
    size = 2 * N
    h_padded = np.zeros(size)
    h_padded[:N] = heights
    
    H = np.fft.fft(h_padded)
    # Linear autocorrelation via FFT
    autocorr_full = np.fft.ifft(H * np.conj(H)).real
    # autocorr_full[k] for k = 0, ..., size-1
    # The linear autocorrelation R[k] = sum_i h[i] * h[i+k] is in autocorr_full[:N]
    # and R[-k] = R[k]
    
    R = np.zeros(2 * N - 1)
    # R[k] for k = -(N-1), ..., N-1
    # autocorr_full[0] = R[0]
    # autocorr_full[k] = R[k] for k = 1, ..., N-1
    # autocorr_full[size-k] = R[-k] = R[k] for k = 1, ..., N-1
    
    R[N-1] = autocorr_full[0]  # lag 0
    for k in range(1, N):
        R[N-1+k] = autocorr_full[k]  # positive lags
        R[N-1-k] = autocorr_full[k]  # negative lags (symmetric)
    
    # The autoconvolution at t = k*d is d * R[k]
    # But we need to be careful about the exact scaling
    
    # Actually, for a step function f(x) = h_i on [x_i, x_{i+1}],
    # (f*f)(t) = integral f(s) f(t-s) ds
    # At t = 0: (f*f)(0) = sum_i h_i^2 * d = d * sum h_i^2
    # This matches R[0] * d where R[0] = sum h_i^2
    
    # At t = d: (f*f)(d) = sum_i h_i * h_{i+1} * d (approximately)
    # More precisely, it's a weighted sum
    
    # For the purpose of finding the max |autoconv|, we can evaluate at the
    # breakpoints and also at midpoints between breakpoints (where the linear
    # pieces might have their max at endpoints for piecewise linear functions)
    
    # The autoconvolution is piecewise linear, so its max absolute value occurs
    # at the breakpoints or at t = 0 (which is a breakpoint)
    
    # Evaluate at all breakpoints: t = k*d for k = -(N), ..., N
    # But the support is [-1/2, 1/2] = [-N*d, N*d], so k = -N, ..., N
    # However, for k with |k| > N-1, the autoconvolution is 0
    
    # So we need t = k*d for k = -(N-1), ..., N-1
    # That's 2N-1 points
    
    # The value at t = k*d is d * R[k] where R[k] is the discrete autocorrelation
    
    # But wait, this isn't quite right. Let me think more carefully.
    
    # For a step function, the autoconvolution is actually a convolution of
    # the "pulse train" with itself. If we define g(x) = sum_i h_i * rect((x - x_i)/d)
    # where rect is 1 on [-1/2, 1/2], then (f*f)(t) = (g*g)(t)
    
    # The convolution of two step functions with the same grid spacing is
    # piecewise linear, and its values at the grid points t = k*d are:
    # (g*g)(k*d) = d * sum_i h_i * h_{i-k}
    
    # And between grid points, it's linear interpolation.
    
    # So the max |autoconv| occurs at either a grid point or at a point where
    # the derivative changes sign (but since it's piecewise linear, the max is
    # at a vertex, which is a grid point).
    
    # Therefore, max |autoconv| = d * max_k |R[k]| for k = -(N-1), ..., N-1
    
    # Compute R[k] for all k
    R_pos = np.zeros(N)
    for k in range(N):
        if k == 0:
            R_pos[k] = np.sum(heights * heights)
        else:
            R_pos[k] = np.sum(heights[k:] * heights[:-k])
    
    # Full autocorrelation
    R_full = np.concatenate([R_pos[:0:-1], R_pos])
    # R_full[k] = R[k] for k = -(N-1), ..., N-1
    
    max_abs = np.max(np.abs(R_full))
    bound = d * max_abs
    
    return bound, R_full, d

def _compute_autoconv_fast(heights):
    """Fast computation using FFT for the autoconvolution bound."""
    d = 1.0 / (2 * N)
    
    # Use FFT to compute linear autocorrelation
    size = 1
    while size < 2 * N:
        size *= 2
    
    h_padded = np.zeros(size)
    h_padded[:N] = heights
    
    H = np.fft.fft(h_padded)
    autocorr = np.fft.ifft(H * np.conj(H)).real[:N]
    
    # autocorr[k] = sum_{i=0}^{N-1-k} h[i] * h[i+k] for k = 0, ..., N-1
    # This is R[k] for k >= 0
    
    # Full autocorrelation for k = -(N-1), ..., N-1
    R_full = np.concatenate([autocorr[:0:-1], autocorr])
    
    max_abs = np.max(np.abs(R_full))
    bound = d * max_abs
    
    return bound, R_full, d

def _soft_max(values, temperature=0.01):
    """Smooth approximation to max using log-sum-exp."""
    max_val = np.max(values)
    if max_val < 0:
        max_val = 0
    exp_vals = np.exp((values - max_val) / temperature)
    log_sum = np.log(np.sum(exp_vals))
    return max_val + temperature * log_sum

def _objective_soft(heights, temperature=0.01):
    """
    Smooth objective: soft-max of |autoconv| values.
    This is differentiable and can be optimized with gradient methods.
    """
    bound, R_full, d = _compute_autoconv_fast(heights)
    
    # The autoconvolution values are d * R_full
    autoconv_vals = d * R_full
    
    # We want to minimize max |autoconv_vals|
    # Use soft-max of |autoconv_vals|
    abs_vals = np.abs(autoconv_vals)
    soft_max_val = _soft_max(abs_vals, temperature)
    
    return soft_max_val

def _objective_and_gradient(heights, temperature=0.01):
    """
    Compute the soft-max objective and its gradient.
    """
    bound, R_full, d = _compute_autoconv_fast(heights)
    autoconv_vals = d * R_full
    abs_vals = np.abs(autoconv_vals)
    
    # Soft-max value
    max_val = np.max(abs_vals)
    exp_vals = np.exp((abs_vals - max_val) / temperature)
    weights = exp_vals / np.sum(exp_vals)
    soft_max_val = max_val + temperature * np.log(np.sum(exp_vals))
    
    # Gradient: d/dh_i [soft_max(|d*R|)]
    # = sum_k weights[k] * sign(autoconv_vals[k]) * d * dR[k]/dh_i
    
    # R[k] = sum_j h[j] * h[j+k] for k >= 0
    # dR[k]/dh[i] = h[i+k] (if i+k < N) + h[i-k] (if i-k >= 0) for k > 0
    #              = 2*h[i] for k = 0
    
    # This is getting complex. Let's use numerical gradient or a simpler approach.
    # For now, return just the objective and use a finite-difference gradient.
    
    return soft_max_val

def construct():
    """
    Construct a signed step function with 400 equal intervals on [-1/4, 1/4]
    to minimize the upper bound on C3.
    """
    start_time = time.time()
    deadline = start_time + 115  # safety margin
    
    # Start with alternating signs: this should give good cancellation
    best_heights = np.array([1.0 if i % 2 == 0 else -1.0 for i in range(N)])
    best_bound, _, _ = _compute_autoconv_fast(best_heights)
    
    # Also try all-ones as baseline
    ones_bound, _, _ = _compute_autoconv_fast(np.ones(N))
    if ones_bound < best_bound:
        best_bound = ones_bound
        best_heights = np.ones(N)
    
    # Try structured patterns
    patterns = [
        np.array([1.0 if i % 4 < 2 else -1.0 for i in range(N)]),  # period 4
        np.array([1.0 if i % 3 < 2 else -1.0 for i in range(N)]),  # period 3
        np.array([1.0 if (i // 10) % 2 == 0 else -1.0 for i in range(N)]),  # blocks of 10
    ]
    
    for pattern in patterns:
        b, _, _ = _compute_autoconv_fast(pattern)
        if b < best_bound:
            best_bound = b
            best_heights = pattern.copy()
    
    # Optimize using L-BFGS-B with the soft-max objective
    # Use a small temperature for near-exact max
    
    def objective(heights):
        return _objective_soft(heights, temperature=0.05)
    
    # Try optimization from the best initial point
    bounds = [(-1, 1)] * N
    
    # First, do a quick optimization with moderate temperature
    if time.time() < deadline - 30:
        try:
            result = minimize(
                objective,
                best_heights,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 200, 'maxfun': 2000, 'ftol': 1e-8}
            )
            new_bound, _, _ = _compute_autoconv_fast(result.x)
            if new_bound < best_bound:
                best_bound = new_bound
                best_heights = result.x
        except Exception:
            pass
    
    # Refine with smaller temperature
    if time.time() < deadline - 20:
        def objective_refine(heights):
            return _objective_soft(heights, temperature=0.01)
        
        try:
            result = minimize(
                objective_refine,
                best_heights,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 300, 'maxfun': 3000, 'ftol': 1e-9}
            )
            new_bound, _, _ = _compute_autoconv_fast(result.x)
            if new_bound < best_bound:
                best_bound = new_bound
                best_heights = result.x
        except Exception:
            pass
    
    # Final polish: project to nearest signed values and check
    if time.time() < deadline - 5:
        # Try rounding to ±1
        rounded = np.sign(best_heights)
        # Where sign is 0, use 1
        rounded[rounded == 0] = 1.0
        rounded_bound, _, _ = _compute_autoconv_fast(rounded)
        if rounded_bound < best_bound:
            best_bound = rounded_bound
            best_heights = rounded
        
        # Try a few random perturbations of the best solution
        for _ in range(20):
            if time.time() > deadline - 5:
                break
            perturbed = best_heights.copy()
            # Flip a few random signs
            n_flips = np.random.randint(1, 10)
            flip_indices = np.random.choice(N, n_flips, replace=False)
            perturbed[flip_indices] *= -1
            p_bound, _, _ = _compute_autoconv_fast(perturbed)
            if p_bound < best_bound:
                best_bound = p_bound
                best_heights = perturbed
    
    return best_heights

# EVOLVE-BLOCK-END
