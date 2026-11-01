"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

N = 400

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time

def _compute_autoconv_bound(heights):
    """
    Compute the upper bound on C3 for a signed step function.
    The autoconvolution of a step function with heights h on N equal intervals
    on [-1/4, 1/4] is computed. The bound is the L-infinity norm of the 
    autoconvolution, which we approximate by evaluating at the grid points.
    
    Actually, for the third autocorrelation inequality, we need to minimize
    the maximum of the autoconvolution. The autoconvolution of a step function
    with constant values on intervals can be computed exactly.
    
    Let's define the function f(x) = h_i for x in [x_i, x_{i+1}] where 
    x_i = -1/4 + i*(1/2)/N for i=0,...,N-1.
    
    The autoconvolution (f*f)(t) = integral f(s) f(t-s) ds.
    
    For a step function, this is a piecewise linear function. We can compute
    it on a fine grid.
    
    Actually, let's think about this differently. The task says "improve an 
    autoconvolution upper bound" and "score is the upper bound on C3".
    
    The seed program returns np.ones(N), which gives a score of 1.0.
    The evaluator measures the raw value (lower is better).
    
    So we want to minimize the upper bound. For all-ones, the autoconvolution
    is a triangle function with peak 1/2 (since the interval length is 1/2,
    and the function is 1 on that interval, the autoconvolution at 0 is 
    integral_{-1/4}^{1/4} 1*1 ds = 1/2).
    
    Wait, let me recalculate. If f(x) = 1 for x in [-1/4, 1/4], then
    (f*f)(0) = integral f(s) f(-s) ds = integral_{-1/4}^{1/4} 1 * 1 ds = 1/2.
    
    But the score for np.ones(N) is 1.0, not 0.5. So maybe the metric is
    different. Perhaps it's the L-infinity norm of the autoconvolution 
    divided by something, or perhaps it's a different quantity.
    
    Let me reconsider. The task mentions "third autocorrelation inequality" 
    and "C3". This might refer to a specific constant in harmonic analysis.
    
    Given the ambiguity, let me try a different approach: use optimization
    to find signed step heights that minimize the maximum absolute value of
    the autoconvolution, and see what score we get.
    
    Actually, looking at the observed changes, p202 got a score of 0.0216,
    which is much better than 1.0. So there's definitely room for improvement.
    
    Let me implement a proper autoconvolution calculation and use optimization.
    """
    # Compute autoconvolution on a fine grid
    # The support of f is [-1/4, 1/4], so the autoconvolution has support [-1/2, 1/2]
    # We'll evaluate on a grid of M points
    
    M = 4000  # fine grid for evaluation
    
    # Create the grid for f
    x = np.linspace(-0.25, 0.25, N+1)
    # heights[i] is the value on [x[i], x[i+1]]
    
    # For autoconvolution, we need to compute (f*f)(t) = int f(s) f(t-s) ds
    # This is a piecewise linear function. We can compute it by convolution.
    
    # Use FFT for efficiency
    # Pad the signal to avoid circular convolution
    size = 2 * N
    f_padded = np.zeros(size)
    f_padded[:N] = heights
    
    # The autoconvolution in the discrete sense
    # But we need the continuous autoconvolution of the step function.
    
    # Alternative: compute directly using the fact that f is piecewise constant.
    # (f*f)(t) = sum_i sum_j h_i h_j * measure({s: s in I_i and t-s in I_j})
    # where I_i = [x_i, x_{i+1}]
    
    # This is complex. Let's use a simpler numerical approach.
    
    # Evaluate f on a fine grid
    x_fine = np.linspace(-0.25, 0.25, 2*N)
    # For each point in x_fine, find which interval it's in
    indices = np.searchsorted(x, x_fine) - 1
    indices = np.clip(indices, 0, N-1)
    f_fine = heights[indices]
    
    # Now compute autoconvolution via FFT
    # We need to be careful about the scaling
    dx = 0.5 / (2*N)  # spacing of the fine grid
    # Pad to avoid circular convolution
    conv_size = 4 * 2 * N
    f_fft = np.zeros(conv_size)
    f_fft[:2*N] = f_fine
    
    F = np.fft.fft(f_fft)
    autoconv = np.fft.ifft(F * F).real
    # Scale by dx
    autoconv *= dx
    
    # The autoconvolution is defined on [-0.5, 0.5]
    # Find the maximum absolute value
    bound = np.max(np.abs(autoconv))
    
    return bound

def construct():
    """
    Construct a signed step function with 400 equal intervals on [-1/4, 1/4]
    to minimize the upper bound on C3.
    """
    start_time = time.time()
    deadline = start_time + 120  # 120 second limit
    
    # Start with all ones
    best_heights = np.ones(N)
    best_bound = _compute_autoconv_bound(best_heights)
    
    # Use gradient-based optimization
    # We'll optimize the heights to minimize the autoconvolution bound
    
    def objective(heights):
        return _compute_autoconv_bound(heights)
    
    # Try a few random restarts with optimization
    n_restarts = 5
    for r in range(n_restarts):
        if time.time() > deadline - 20:
            break
        
        # Random initialization with signs
        if r == 0:
            x0 = best_heights.copy()
        else:
            x0 = np.random.choice([-1, 1], size=N).astype(float)
        
        # Use L-BFGS-B with bounds
        bounds = [(-1, 1)] * N
        
        try:
            result = minimize(
                objective,
                x0,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 100, 'maxfun': 1000}
            )
            
            if result.fun < best_bound:
                best_bound = result.fun
                best_heights = result.x
        except Exception:
            pass
    
    # Final polish: try to improve with more iterations
    if time.time() < deadline - 10:
        try:
            bounds = [(-1, 1)] * N
            result = minimize(
                objective,
                best_heights,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 500, 'maxfun': 5000}
            )
            if result.fun < best_bound:
                best_bound = result.fun
                best_heights = result.x
        except Exception:
            pass
    
    # Return the heights
    return best_heights

# EVOLVE-BLOCK-END
