import numpy as np
import time
from scipy.optimize import minimize, differential_evolution

N = 600

def construct():
    """
    Returns a sequence of nonnegative step heights on equal intervals of [-1/4, 1/4].
    We minimize the normalized maximum of the autoconvolution (upper bound on C1).
    
    The objective is: max(autoconv(h)) / (sum(h^2) * dx)
    where dx = (1/2)/N and autoconv is the continuous autoconvolution of the step function.
    
    For a step function with heights h_i on equal intervals of width dx:
    - ||h||_2^2 = sum(h_i^2) * dx
    - The autoconvolution at point x is integral h(t)h(x-t) dt
    
    We use a structural approach:
    1. Start with uniform (baseline score 1.0)
    2. Use differential evolution to find better shapes in a reduced parameter space
    3. The key insight: we want to minimize peak autoconv while keeping L2 norm fixed.
       This means spreading mass in a way that reduces peak overlap.
    
    Strategy: Parameterize h as sum of a few basis functions (e.g., cosine modes or
    localized bumps) and optimize the coefficients. Then refine with direct optimization
    on the full 600-dim vector using scipy.
    """
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    dx = 0.5 / N  # interval width
    
    def autoconv_peak(h):
        """Compute peak of continuous autoconvolution of step function h on [-1/4, 1/4].
        
        The autoconvolution (h*h)(x) = integral_{-1/4}^{1/4} h(t) h(x-t) dt
        
        For step functions, we can compute this via FFT.
        The autoconvolution at grid points can be computed as:
        - Pad h to length 2N, compute linear convolution via FFT
        - The autoconv at the original grid points corresponds to specific indices
        
        Actually, for a function on [-1/4, 1/4] with N intervals:
        Let's think of h as defined on points x_i = -1/4 + i*dx, i=0..N-1
        The autoconv at point x is int h(t)h(x-t) dt.
        
        We can compute this by:
        1. Create array of length 2N-1 (or padded) 
        2. Use np.convolve or FFT
        
        More carefully: h is piecewise constant. Let h_i be the value on interval [x_i, x_{i+1}].
        The autoconv at the center of interval i is approximately sum_j h_i * h_j * (overlap length).
        
        Actually, the simplest correct approach: 
        The autoconvolution of two piecewise constant functions on equal intervals is a piecewise
        linear function. Its peak occurs at one of the interval centers or boundaries.
        
        For computation, we can use the fact that if we treat h as a discrete sequence and compute
        the circular convolution, we need to be careful. Let's use linear convolution.
        
        Let's define: h has N values. The autoconv at the "middle" positions:
        (h*h)(x_k) for k = -(N-1), ..., N-1
        
        Using FFT: 
        - Zero-pad h to length at least 2N
        - Compute FFT, square, inverse FFT
        - The result at appropriate indices gives the autoconv values
        
        The peak of the autoconv over all real x: for piecewise constant h, the autoconv is
        piecewise quadratic (or piecewise linear if we think of it differently). The maximum
        occurs at a point where the derivative changes sign, which for autoconv of step functions
        is at the midpoints of intervals or at interval boundaries.
        
        Simplest robust approach: evaluate autoconv at all 2N-1 "natural" points (centers of
        sub-intervals of the convolution support) and take the max.
        """
        # Use FFT to compute linear autoconvolution
        # h is length N. Linear convolution of h with h has length 2N-1.
        # We want the autoconv of the continuous step function.
        
        # The autoconv of step functions: if h(t) = h_i for t in [x_i, x_{i+1}],
        # then (h*h)(x) = sum_{i,j} h_i h_j * length of {t: t in [x_i,x_{i+1}] and x-t in [x_j,x_{j+1}]}
        
        # This equals sum_{i,j} h_i h_j * max(0, min(x_{i+1}, x - x_j) - max(x_i, x - x_{j+1}))
        
        # For efficiency, use FFT. The autoconv at the "natural" grid points (midpoints of
        # the 2N-1 sub-intervals of the convolution support) can be computed as:
        
        # Actually, let's just compute it directly for the grid points where we evaluate.
        # The convolution support is [-1/2, 1/2]. We evaluate at 2N-1 points:
        # x_k = -1/2 + (k+0.5)*dx for k=0,...,2N-2
        
        # But for peak, we should also check boundaries. For piecewise linear autoconv,
        # the max is at a vertex or where slope changes. The vertices are at the interval
        # boundaries of the original function mapped through convolution.
        
        # Let me use a simpler but correct method: 
        # The autoconv of step functions on equal intervals, evaluated at the centers of
        # the 2N-1 sub-intervals of the support, is:
        # C_k = dx * sum_i h_i * h_{i+k-N+1}  (with appropriate indexing)
        
        # More precisely: if we index h from 0 to N-1, and the convolution index k from 0 to 2N-2,
        # then C_k = dx * sum_{i} h_i * h_{k-i} where h_j = 0 for j outside [0,N-1]
        
        # Wait, that's the discrete convolution scaled by dx. But is that the correct
        # continuous autoconv at the sub-interval centers?
        
        # Let's verify: h(t) = sum_i h_i * 1_{[x_i, x_{i+1}]}(t)
        # (h*h)(x) = sum_{i,j} h_i h_j * measure({t: t in [x_i,x_{i+1}] and x-t in [x_j,x_{j+1}]})
        # = sum_{i,j} h_i h_j * max(0, min(x_{i+1}, x - x_j) - max(x_i, x - x_{j+1}))
        
        # For x = x_k + dx/2 (center of sub-interval k in the convolution support):
        # The overlap of [x_i, x_{i+1}] and [x - x_{j+1}, x - x_j] = [x - x_{j+1}, x - x_j]
        # = [x_k + dx/2 - x_{j+1}, x_k + dx/2 - x_j]
        
        # This is getting complex. Let me just use the discrete convolution approximation
        # which is exact for the integral when evaluated at the right points.
        
        # Actually, the cleanest approach: use np.convolve with mode='full' on the array h,
        # then multiply by dx. This gives the autoconv at the "natural" grid points.
        
        conv = np.convolve(h, h, mode='full') * dx
        return np.max(conv)
    
    def objective(h):
        """Objective: max(autoconv) / ||h||_2^2, where ||h||_2^2 = sum(h^2)*dx"""
        l2sq = np.sum(h**2) * dx
        if l2sq < 1e-15:
            return 1e10
        peak = autoconv_peak(h)
        return peak / l2sq
    
    # Baseline: uniform function
    h_uniform = np.ones(N)
    best_h = h_uniform.copy()
    best_score = objective(h_uniform)  # Should be ~1.0
    
    # Strategy: Use differential evolution on a reduced parameter space,
    # then refine with L-BFGS-B on the full space.
    
    # Parameterization: h = max(0, a0 + a1*cos(2*pi*1*x) + a2*cos(2*pi*2*x) + ... )
    # where x is in [-1/4, 1/4], mapped to [0, 1] for the cosine.
    # Actually, let's use a different structural approach.
    
    # Key insight: To minimize peak autoconv, we want the function to be as "flat" as possible
    # in its convolution. The uniform function is optimal in some sense, but we can do better
    # by using functions that have less self-overlap at the peak.
    
    # Try: h(x) = 1 + epsilon * f(x) where f is chosen to reduce peak autoconv.
    # The peak autoconv of uniform is at x=0 (center), and it's proportional to the integral.
    # By making h smaller in the center and larger at the edges, we might reduce the peak.
    
    # Let's parameterize with a few Fourier modes and optimize.
    
    # Define grid
    x = -0.25 + (np.arange(N) + 0.5) * dx  # centers of intervals
    
    def fourier_to_h(params):
        """Convert Fourier coefficients to step heights.
        params: [a0, b1, a1, b2, a2, ...] 
        h(x) = a0 + sum_k (a_k cos(2*pi*k*x/(0.5)) + b_k sin(2*pi*k*x/(0.5)))
        Clamped to nonnegative.
        """
        n_modes = (len(params) - 1) // 2
        h = np.full(N, params[0])
        for k in range(1, n_modes + 1):
            a_k = params[1 + 2*(k-1)]
            b_k = params[2 + 2*(k-1)]
            h += a_k * np.cos(2*np.pi*k*x/0.5) + b_k * np.sin(2*np.pi*k*x/0.5)
        return np.maximum(h, 0)
    
    def fourier_objective(params):
        h = fourier_to_h(params)
        return objective(h)
    
    # Phase 1: Differential evolution on Fourier coefficients
    # Use a few modes
    n_modes = 5
    n_params = 1 + 2*n_modes
    
    # Bounds: a0 in [0.1, 2.0], others in [-0.5, 0.5]
    bounds = [(0.1, 2.0)] + [(-0.5, 0.5)] * (2*n_modes)
    
    elapsed = time.time() - start_time
    if elapsed < deadline - 30:
        # Run DE with time limit
        max_iter = max(10, int((deadline - time.time() - 20) / 0.5))
        try:
            result = differential_evolution(
                fourier_objective, bounds, 
                maxiter=max_iter, popsize=15, tol=1e-4,
                seed=42, polish=False, init='sobol'
            )
            if result.fun < best_score:
                best_h = fourier_to_h(result.x)
                best_score = result.fun
        except Exception:
            pass
    
    # Phase 2: Refine with L-BFGS-B on full 600-dim space
    elapsed = time.time() - start_time
    if elapsed < deadline - 15:
        # Use the current best as starting point
        h0 = best_h.copy()
        
        def full_objective(h):
            return objective(h)
        
        def full_grad(h):
            # Numerical gradient
            eps = 1e-7
            grad = np.zeros_like(h)
            for i in range(N):
                h_plus = h.copy()
                h_plus[i] += eps
                h_minus = h.copy()
                h_minus[i] -= eps
                grad[i] = (full_objective(h_plus) - full_objective(h_minus)) / (2*eps)
            return grad
        
        try:
            result = minimize(
                full_objective, h0, method='L-BFGS-B',
                bounds=[(0, None)] * N,
                options={'maxiter': 200, 'gtol': 1e-5}
            )
            if result.fun < best_score:
                best_h = np.maximum(result.x, 0)
                best_score = result.fun
        except Exception:
            pass
    
    # Phase 3: Local search - try perturbing and re-optimizing
    elapsed = time.time() - start_time
    if elapsed < deadline - 10:
        # Try a few random perturbations of the best solution
        rng = np.random.RandomState(123)
        for trial in range(5):
            if time.time() > deadline - 10:
                break
            h_trial = best_h.copy()
            # Add small random perturbation
            h_trial += rng.normal(0, 0.1, N)
            h_trial = np.maximum(h_trial, 0)
            
            try:
                result = minimize(
                    full_objective, h_trial, method='L-BFGS-B',
                    bounds=[(0, None)] * N,
                    options={'maxiter': 50, 'gtol': 1e-5}
                )
                if result.fun < best_score:
                    best_h = np.maximum(result.x, 0)
                    best_score = result.fun
            except Exception:
                pass
    
    # Ensure nonnegative
    best_h = np.maximum(best_h, 0)
    
    return list(best_h)
