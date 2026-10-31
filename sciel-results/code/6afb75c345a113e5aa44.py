"""Rewritten program: signed step function optimization for autoconvolution bound."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time


def _compute_autoconvolution_bound(f):
    """Compute the upper bound on C3 from autoconvolution of f.
    
    The autoconvolution is f * f (convolution of f with itself).
    The upper bound is likely the maximum absolute value or the L-infinity norm
    of the autoconvolution, possibly normalized.
    """
    # Compute autoconvolution via FFT for efficiency
    n = len(f)
    # Zero-pad to avoid circular convolution
    size = 2 * n
    f_ext = np.zeros(size)
    f_ext[:n] = f
    F = np.fft.fft(f_ext, size)
    autoconv = np.fft.ifft(F * F).real
    # The autoconvolution has length 2n-1 effectively, but we get 2n with padding
    # Take the valid part
    autoconv = autoconv[:2*n-1]
    
    # The upper bound on C3: likely the maximum of the autoconvolution
    # or related to the L2 norm. Let's use the max absolute value as a proxy.
    # Actually, for autocorrelation inequalities, it might be the peak sidelobe level
    # or the maximum value of the autoconvolution.
    
    # Common interpretation: the bound involves the maximum of |autoconv|
    # or the ratio of peak to average. Let's return the max absolute value.
    return np.max(np.abs(autoconv))


def _compute_bound_normalized(f):
    """Alternative: normalized autoconvolution bound."""
    n = len(f)
    size = 2 * n
    f_ext = np.zeros(size)
    f_ext[:n] = f
    F = np.fft.fft(f_ext, size)
    autoconv = np.fft.ifft(F * F).real
    autoconv = autoconv[:2*n-1]
    
    # Normalize by the DC component (lag 0 value)
    peak = autoconv[0] if autoconv[0] != 0 else 1.0
    # The bound might be the maximum sidelobe / peak
    sidelobes = np.concatenate([autoconv[1:], autoconv[-1]])  # exclude center
    if len(sidelobes) > 0:
        max_sidelobe = np.max(np.abs(sidelobes))
        return max_sidelobe / abs(peak) if abs(peak) > 1e-15 else 0.0
    return 0.0


def _try_random_signed(rng, n_iter=50):
    """Try random ±1 sequences and find the one with lowest bound."""
    best_f = None
    best_bound = np.inf
    
    for _ in range(n_iter):
        f = rng.choice([-1.0, 1.0], size=N)
        bound = _compute_autoconvolution_bound(f)
        if bound < best_bound:
            best_bound = bound
            best_f = f.copy()
    
    return best_f, best_bound


def _optimize_signs(f, n_iter=1000, step_size=1):
    """Local optimization: flip signs to reduce the autoconvolution bound."""
    current_f = f.copy()
    current_bound = _compute_autoconvolution_bound(current_f)
    best_f = current_f.copy()
    best_bound = current_bound
    
    for i in range(n_iter):
        # Pick a random index to flip
        idx = np.random.randint(N)
        old_val = current_f[idx]
        current_f[idx] = -old_val
        
        new_bound = _compute_autoconvolution_bound(current_f)
        if new_bound < current_bound:
            current_bound = new_bound
            if new_bound < best_bound:
                best_bound = new_bound
                best_f = current_f.copy()
        else:
            # Revert
            current_f[idx] = old_val
    
    return best_f, best_bound


def construct():
    """Construct a signed step function that minimizes the autoconvolution upper bound."""
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin before 120s limit
    
    rng = np.random.default_rng(42)
    
    # Start with a random ±1 sequence
    best_f, best_bound = _try_random_signed(rng, n_iter=20)
    
    # Local optimization
    while time.time() < deadline - 10.0:
        f, bound = _optimize_signs(best_f, n_iter=200)
        if bound < best_bound:
            best_bound = bound
            best_f = f
        else:
            # Try a new random restart
            f, bound = _try_random_signed(rng, n_iter=5)
            if bound < best_bound:
                best_bound = bound
                best_f = f
    
    return best_f
# EVOLVE-BLOCK-END
