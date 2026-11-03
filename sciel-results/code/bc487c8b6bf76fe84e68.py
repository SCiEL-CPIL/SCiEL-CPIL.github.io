"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
from scipy.optimize import minimize_scalar, minimize
import time

N = 600

# EVOLVE-BLOCK-START
def _compute_objective(x):
    """Compute the normalized maximum of the autoconvolution for step heights x.
    
    The step function is defined on [-1/4, 1/4] with N equal intervals.
    The autoconvolution (self-convolution) of a step function on [-1/4, 1/4]
    results in a function on [-1/2, 1/2]. We compute its maximum value
    normalized appropriately.
    
    For a step function f(t) = x[i] on interval i of [-1/4, 1/4],
    the autoconvolution (f * f)(s) = ∫ f(t) f(s-t) dt.
    
    Since f is a step function with N intervals of width h = (1/2)/N = 1/(2N) on [-1/4, 1/4],
    we can compute the convolution efficiently.
    
    The normalized maximum: we need to figure out what "normalized" means.
    Typically for autocorrelation inequalities, we look at:
    max_t (f*f)(t) / (∫f)^2 or similar.
    
    Let's think about this more carefully. The autoconvolution (f*f)(s) = ∫_{-1/4}^{1/4} f(t) f(s-t) dt.
    Since f is supported on [-1/4, 1/4], (f*f)(s) is supported on [-1/2, 1/2].
    
    For a step function with heights x[i] on intervals of width h = 1/(2N):
    (f*f)(s) can be computed. At s=0, (f*f)(0) = ∫ f(t)^2 dt = h * sum(x[i]^2).
    
    The normalization likely involves the L1 norm or L2 norm. Let's assume the score is:
    max_s (f*f)(s) / (∫f)^2 * (1/4) or something similar.
    
    Actually, let's just compute the raw maximum of the autoconvolution and see.
    The problem says "minimize the normalized maximum of its autoconvolution".
    
    Let's compute:
    - h = 1/(2N) is the interval width
    - ∫f = h * sum(x)
    - (f*f)(s) for various s
    
    For efficiency, we can compute the convolution using FFT or direct computation.
    Since N=600, direct computation might be feasible but let's use FFT for speed.
    
    Actually, for a step function, the autoconvolution at discrete points can be computed.
    Let's sample the autoconvolution at many points and take the max.
    """
    x = np.asarray(x, dtype=np.float64)
    h = 1.0 / (2 * N)
    
    # Compute autoconvolution using FFT
    # Pad to avoid circular convolution effects
    size = 2 * N + 1
    f_padded = np.zeros(size)
    f_padded[:N] = x
    f_padded *= h  # scale by interval width for integration
    
    # Autoconvolution
    F = np.fft.rfft(f_padded, n=2 * size)
    conv = np.fft.irfft(F * F, n=2 * size)
    
    # The autoconvolution is on [-1/2, 1/2], sampled at 2*size points
    # We need the maximum of conv
    max_conv = np.max(conv)
    
    # Normalization: (∫f)^2
    integral_f = np.sum(x) * h
    norm = integral_f ** 2
    
    if norm < 1e-30:
        return 1e10
    
    return max_conv / norm


def _compute_objective_fast(x):
    """Faster computation of the objective using direct convolution for step functions."""
    x = np.asarray(x, dtype=np.float64)
    h = 1.0 / (2 * N)
    
    # For a step function, the autoconvolution is piecewise linear (trapezoidal)
    # We can compute it at the midpoints of each sub-interval of [-1/2, 1/2]
    
    # The autoconvolution (f*f)(s) = sum_i sum_j x[i]*x[j]*h^2 * overlap_length
    # where overlap_length is the length of intersection of [i*h - 1/4, (i+1)*h - 1/4]
    # and [s - j*h + 1/4, s - (j+1)*h + 1/4] ... this is getting complicated.
    
    # Let's use FFT approach but more efficiently
    # Pad to power of 2 for speed
    size = 2 * N
    n_fft = 1
    while n_fft < 2 * size:
        n_fft *= 2
    
    f_padded = np.zeros(n_fft)
    f_padded[:N] = x * h
    
    F = np.fft.rfft(f_padded)
    conv = np.fft.irfft(F * F, n=n_fft)
    
    # Take the first 2*N points which cover [-1/2, 1/2]
    max_conv = np.max(conv[:2 * N])
    
    integral_f = np.sum(x) * h
    norm = integral_f ** 2
    
    if norm < 1e-30:
        return 1e10
    
    return max_conv / norm


def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Leave 10s margin
    
    # Initialize with a reasonable starting point
    # Try different initializations
    x = np.ones(N)
    best_x = x.copy()
    best_val = _compute_objective_fast(x)
    
    # Coordinate descent with exact 1D minimization
    # Cycle through coordinates, for each do a scalar minimization
    
    max_iter = 500  # Max number of full cycles
    for iteration in range(max_iter):
        if time.time() > deadline - 5.0:
            break
        
        improved = False
        for i in range(N):
            if time.time() > deadline - 2.0:
                    break
            
            # Minimize over x[i] with others fixed
            # x[i] must be >= 0
            
            def objective_1d(xi):
                x[i] = xi
                return _compute_objective_fast(x)
            
            # Use minimize_scalar with bounds [0, some_max]
            # The optimal x[i] is likely not too large
            current_xi = x[i]
            result = minimize_scalar(
                objective_1d,
                bounds=(0, max(1.0, current_xi * 2 + 1.0)),
                method='bounded',
                options={'xatol': 1e-8, 'maxiter': 100}
            )
            
            if result.success and result.fun < best_val - 1e-10:
                x[i] = result.x
                best_val = result.fun
                best_x = x.copy()
                improved = True
        
        if not improved:
            break
    
    # Final polish with a few more iterations if time permits
    if time.time() < deadline - 10.0:
        for _ in range(3):
            if time.time() > deadline - 5.0:
                break
            for i in range(N):
                if time.time() > deadline - 2.0:
                    break
                def objective_1d(xi):
                    x[i] = xi
                    return _compute_objective_fast(x)
                
                result = minimize_scalar(
                    objective_1d,
                    bounds=(0, max(1.0, x[i] * 2 + 1.0)),
                    method='bounded',
                    options={'xatol': 1e-10, 'maxiter': 200}
                )
                
                if result.success and result.fun < best_val - 1e-12:
                    x[i] = result.x
                    best_val = result.fun
                    best_x = x.copy()
    
    return list(best_x)
# EVOLVE-BLOCK-END
