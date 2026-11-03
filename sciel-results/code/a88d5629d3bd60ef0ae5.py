"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
from scipy.optimize import minimize
from scipy.fft import fft, ifft

N = 600

def _autoconvolution(h):
    """Compute the autoconvolution of step function h on N equal intervals.
    
    h is an array of N nonnegative heights. The step function is constant on
    each interval of length 1/(4N) over [-1/4, 1/4].
    The autoconvolution (f*f)(x) = integral f(x-t)f(t) dt.
    
    For a step function with heights h_i on intervals of width w=1/(4N),
    the autoconvolution at point x is computed via the discrete convolution
    of the heights scaled appropriately.
    
    Specifically, if we define g(t) = h_i for t in [i*w - 1/4, (i+1)*w - 1/4),
    then (g*g)(x) = sum_i h_i * h_j * overlap_length(x; i, j)
    
    A simpler approach: the autoconvolution of a piecewise constant function
    can be computed via FFT. The continuous autoconvolution at grid points can
    be approximated by:
    
    Let w = 1/(4N). The function g(t) = h_{floor((t+1/4)/w)} for t in [-1/4, 1/4).
    The autoconvolution (g*g)(x) for x in [-1/2, 1/2] is:
    (g*g)(x) = integral_{max(-1/4, x-1/4)}^{min(1/4, x+1/4)} g(x-t)g(t) dt
    
    For a step function, this is a piecewise linear function. The maximum of
    the autoconvolution occurs at one of the grid points or midpoints.
    
    Using FFT: The discrete autoconvolution of h with itself (circular) gives
    values proportional to the autoconvolution at grid points.
    
    Actually, let's use the direct formula. For step functions on N intervals of
    width w, the autoconvolution at the center of interval k (i.e., at position
    x_k = -1/4 + (k+0.5)*w) is:
    
    (g*g)(x_k) = sum_{i,j} h_i * h_j * L_{i,j}(x_k)
    
    where L_{i,j}(x) is the length of the intersection of [x-w_i_start, x-w_i_end] 
    with interval j... this is getting complex.
    
    Simpler: Use the fact that for a step function, the autoconvolution is a
    piecewise quadratic function. But for our purposes, we can compute it
    efficiently using FFT.
    
    The key insight: if we sample the step function at N points (one per interval,
    at the center), then the discrete autoconvolution (via FFT) approximates
    the continuous autoconvolution scaled by w.
    
    Specifically, let h be the array of N heights. The continuous autoconvolution
    at the grid points x_k = -1/4 + (k+0.5)*w is approximately:
    
    (g*g)(x_k) ≈ w * (h * h)[k]
    
    where (h*h) is the circular convolution of h with itself.
    
    The maximum of the continuous autoconvolution is bounded above by the max
    of the values at grid points plus a correction term. For a step function,
    the autoconvolution is piecewise quadratic, and its maximum on each piece
    can be computed exactly. But for a good upper bound, using the grid-point
    values gives a reasonable proxy.
    
    Actually, for the normalized maximum, we need:
    C1_upper = max_x (g*g)(x) / (integral g)^2
    
    Since integral g = w * sum(h), and max (g*g)(x) ≈ w * max(conv(h,h)),
    we have C1_upper ≈ w * max(conv(h,h)) / (w * sum(h))^2 = max(conv(h,h)) / (w * sum(h)^2)
    
    Wait, let me redo this. Let S = sum(h). Then integral g = w*S.
    (g*g)(x) has units of height^2 * length. So (g*g)(x) = w * (discrete_conv)(x/w).
    The normalized value is (g*g)(x) / (integral g)^2 = w * conv(x) / (w^2 * S^2) = conv(x) / (w * S^2).
    
    So we want to minimize max(conv(h,h)) / (w * S^2), where conv is the circular
    convolution of h with itself.
    
    Equivalently, minimize max(conv(h,h)) / S^2, since w is constant.
    """
    # Circular convolution of h with itself
    conv = ifft(fft(h) ** 2).real
    return conv

def _score(h):
    """Compute the score: upper bound on C1. Lower is better."""
    h = np.asarray(h, dtype=float)
    S = np.sum(h)
    if S <= 0:
        return float('inf')
    conv = _autoconvolution(h)
    max_conv = np.max(conv)
    w = 1.0 / (4 * N)
    return max_conv / (w * S * S)

def construct():
    """Construct a nonnegative step function with N intervals to minimize C1 upper bound."""
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s limit
    
    best_h = np.ones(N)
    best_score = _score(best_h)
    
    # Initialize with a smooth profile to avoid poor local minima
    # Use a raised cosine or similar smooth shape
    x = np.linspace(-1/4, 1/4, N, endpoint=False)
    h_init = np.maximum(0, 1 + np.cos(2 * np.pi * x))  # Raised cosine
    h_init = h_init / np.max(h_init) * np.mean(np.ones(N))  # Normalize
    
    # Try multiple initializations
    initializations = [
        np.ones(N),
        h_init,
        np.maximum(0, 1 - 4 * np.abs(x)),  # Triangle
        np.exp(-10 * x**2),  # Gaussian-like
    ]
    
    for h0 in initializations:
        if time.time() > deadline:
            break
        h0 = np.asarray(h0, dtype=float)
        h0 = np.maximum(0, h0)
        if np.sum(h0) == 0:
            continue
        
        # Use gradient-based optimization with soft constraints
        # We optimize in log-space to maintain non-negativity
        # Let h = exp(y) / sum(exp(y)) * S_target, but we can fix S since
        # the objective is homogeneous: score(h) = score(c*h) for c>0
        # So we can fix sum(h) = N (or any constant) and minimize max(conv)
        
        # Fix sum(h) = N by normalizing
        h0 = h0 / np.sum(h0) * N
        
        # Optimize y where h = exp(y) / sum(exp(y)) * N
        def objective(y):
            h = np.exp(y - np.max(y))
            h = h / np.sum(h) * N
            conv = _autoconvolution(h)
            # Use soft-max to smooth the max
            M = np.max(conv)
            if M <= 0:
                return 0
            # Soft max approximation
            beta = 50
            soft_max = np.log(np.sum(np.exp(beta * conv / M))) / beta * M
            return soft_max / (w * N * N)
        
        # Gradient of the objective
        def gradient(y):
            eps = 1e-7
            grad = np.zeros_like(y)
            for i in range(N):
                y_plus = y.copy()
                y_plus[i] += eps
                y_minus = y.copy()
                y_minus[i] -= eps
                grad[i] = (objective(y_plus) - objective(y_minus)) / (2 * eps)
            return grad
        
        # Use L-BFGS-B with finite differences for gradient
        try:
            result = minimize(
                objective,
                np.zeros(N),  # Start from uniform in log-space
                method='L-BFGS-B',
                options={'maxiter': 200, 'ftol': 1e-10, 'gtol': 1e-8},
            )
            if time.time() > deadline:
                break
            
            h_opt = np.exp(result.x - np.max(result.x))
            h_opt = h_opt / np.sum(h_opt) * N
            h_opt = np.maximum(0, h_opt)
            
            if np.sum(h_opt) > 0:
                score_opt = _score(h_opt)
                if score_opt < best_score:
                    best_score = score_opt
                    best_h = h_opt
        except Exception:
            pass
    
    # Final polish: coordinate descent
    if time.time() < deadline:
        h = best_h.copy()
        for iteration in range(100):
            if time.time() > deadline:
                break
            improved = False
            for i in range(N):
                old_h = h[i]
                # Optimize h[i] while keeping others fixed
                # score = max(conv) / (w * S^2)
                # conv depends on h[i] quadratically
                # This is complex; use numerical line search
                best_val = old_h
                best_s = _score(h)
                for factor in [0.5, 0.7, 0.8, 1.2, 1.5, 2.0]:
                    h[i] = old_h * factor
                    if h[i] < 0:
                        h[i] = 0
                    s = _score(h)
                    if s < best_s:
                        best_s = s
                        best_val = h[i]
                h[i] = best_val
                if abs(h[i] - old_h) > 1e-10:
                    improved = True
            if not improved:
                break
        
        final_score = _score(h)
        if final_score < best_score:
            best_score = final_score
            best_h = h
    
    # Return as list of floats
    return list(best_h)

# EVOLVE-BLOCK-START
# (construct is defined above)
# EVOLVE-BLOCK-END
