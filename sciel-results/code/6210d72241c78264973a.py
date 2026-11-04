"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
from scipy.optimize import minimize
import time

N = 50

# EVOLVE-BLOCK-START
def construct():
    deadline = time.time() + 110  # Safety margin below 120s limit
    
    # Fast exact score: ratio = ||h||_2^2 / (||h||_1 * ||autoconv||_inf)
    # Since the problem is scale-invariant, we can fix ||h||_2 = 1 and
    # minimize (1/||h||_1) * ||autoconv||_inf, equivalently maximize ||h||_1 / ||autoconv||_inf
    # with ||h||_2 = 1. But let's just compute the ratio directly for clarity.
    
    def score(h):
        h = np.asarray(h, dtype=float)
        sum_sq = np.sum(h**2)
        sum_h = np.sum(h)
        if sum_h < 1e-15 or sum_sq < 1e-30:
            return 0.0
        # Linear autocorrelation (not circular) for the step function on [-1/4, 1/4]
        # The autoconvolution of a step function: (f*f)(t) = integral f(t-s)f(s)ds
        # For step heights h_i on intervals of width d, the autoconvolution at lag k*d
        # involves sum of h_i * h_{i+k} * d for overlapping parts.
        # The supremum norm of the autoconvolution: for a step function, the autoconvolution
        # is piecewise linear (trapezoidal), and its max occurs at the center of each 
        # "overlap region". For discrete steps, the max of the continuous autoconvolution
        # is bounded by max over lags of the discrete autocorrelation sum.
        # Actually, for a step function, f*f at time t is a piecewise linear function.
        # The peak value at the center of the k-th overlap is:
        #   d * sum_{i} h_i * h_{i+k}  (for the central part of the overlap)
        # But there are also edge effects. The supremum is actually:
        #   max over all t of (f*f)(t)
        # For a step function with N steps, the autoconvolution has a trapezoidal shape
        # for each lag. The global max is at t=0 (self-overlap) which is sum h_i^2 * d.
        # Wait, no. Let me think more carefully.
        #
        # f(x) = h_i for x in [x_i, x_{i+1}), where x_i = -1/4 + i*d, d = 1/(2*N)
        # Wait, the interval is [-1/4, 1/4], so length = 1/2. With N=50 intervals, d = 1/(2*50) = 1/100.
        #
        # (f*f)(t) = integral_{-inf}^{inf} f(s) f(t-s) ds
        # For step functions, this is a piecewise linear function. At t=0:
        #   (f*f)(0) = sum_i h_i^2 * d
        # At t = k*d (for integer k), the overlap of step i and step i+k is d (full overlap),
        # so (f*f)(k*d) = d * sum_i h_i * h_{i+k} (where indices are valid)
        # But between integer multiples of d, there are partial overlaps.
        # The maximum of the autoconvolution occurs either at t=0 or at some k*d.
        # Actually, for nonnegative step functions, the autoconvolution is unimodal near t=0
        # for each "pair", and the global max is at t=0.
        #
        # Wait, that's not right either. Consider two spikes far apart. The autoconvolution
        # would have peaks at t=0 and at t = +/- distance between spikes.
        #
        # Let me just compute the discrete autocorrelation and use that as a proxy.
        # The supremum of the continuous autoconvolution for a step function is:
        #   max over k of [ d * sum_i h_i * h_{i+k} ]
        # where the sum is over valid i, and k ranges from -(N-1) to N-1.
        # This is because the autoconvolution of two step functions (same function)
        # has its local maxima at integer multiples of d, and the value there is
        # d * (discrete autocorrelation at lag k).
        #
        # Actually, I need to be more careful. The autoconvolution (f*f)(t) for step functions
        # is a continuous piecewise-linear function. Its local maxima occur at t = k*d.
        # The value at t = k*d is d * sum_i h_i * h_{i+k}.
        # The global supremum is the max of these values over all k.
        #
        # So: ||f*f||_inf = d * max_k sum_i h_i * h_{i+k}
        # where d = 1/(2*N).
        #
        # And ||f||_1 = d * sum_i h_i
        # ||f||_2^2 = d * sum_i h_i^2
        #
        # Ratio = (d * sum h_i^2) / ((d * sum h_i) * (d * max_k sum_i h_i h_{i+k}))
        #       = (sum h_i^2) / (d * sum h_i * max_k sum_i h_i h_{i+k})
        
        d = 1.0 / (2 * N)
        # Compute autocorrelation
        # Use np.correlate for efficiency
        ac = np.correlate(h, h, mode='full')
        # ac has length 2*N-1, with ac[N-1+k] = sum_i h_i * h_{i+k} for k in [-(N-1), N-1]
        max_ac = np.max(ac)
        
        ratio = sum_sq / (d * sum_h * max_ac)
        return ratio
    
    # Initial best: uniform
    best_h = np.ones(N)
    best_score = score(best_h)
    
    # Strategy: Fix ||h||_2 = 1 (scale invariance), maximize ||h||_1 / ||autoconv||_inf
    # Equivalently, minimize log(||autoconv||_inf / ||h||_1) subject to ||h||_2 = 1, h >= 0
    #
    # For L-BFGS-B, we need a smooth objective. The max is non-smooth.
    # Use a smooth surrogate: soft-max via log-sum-exp.
    
    def soft_max(x, beta=100.0):
        """Smooth approximation to max(x)"""
        x_max = np.max(x)
        return x_max + np.log(np.sum(np.exp((x - x_max) * beta))) / beta
    
    def objective_constrained(x, beta=50.0):
        """Minimize log(||autoconv||_inf / ||h||_1) with ||h||_2 = 1, h >= 0.
        We parameterize h = exp(y) / ||exp(y)||_2 to enforce ||h||_2 = 1 and h > 0."""
        h = np.exp(x)
        h_norm = np.sqrt(np.sum(h**2))
        h = h / h_norm
        
        sum_h = np.sum(h)
        if sum_h < 1e-15:
            return 1e10
        
        ac = np.correlate(h, h, mode='full')
        # Use soft-max for smoothness
        max_ac_soft = soft_max(ac, beta)
        
        # We want to minimize max_ac_soft / sum_h
        # Equivalent to minimizing log(max_ac_soft) - log(sum_h)
        val = np.log(max_ac_soft) - np.log(sum_h)
        return val
    
    def score_from_y(y):
        """Compute the actual score from the y-parameterization."""
        h = np.exp(y)
        h_norm = np.sqrt(np.sum(h**2))
        h = h / h_norm
        return score(h)
    
    # Run L-BFGS-B from multiple initializations
    np.random.seed(42)
    
    initializations = []
    
    # 1. Uniform (y = 0)
    initializations.append(np.zeros(N))
    
    # 2. Spike-like: one large value
    for peak_pos in [0, N//4, N//2, 3*N//4, N-1]:
        y = np.full(N, -5.0)
        y[peak_pos] = 0.0
        initializations.append(y.copy())
        
    # 3. Two spikes
    for p1, p2 in [(0, N-1), (N//4, 3*N//4), (0, N//2), (N//4, N//2)]:
        y = np.full(N, -5.0)
        y[p1] = 0.0
        y[p2] = 0.0
        initializations.append(y.copy())
    
    # 4. Random smooth
    for _ in range(10):
        y = np.random.randn(N) * 1.0
        initializations.append(y.copy())
    
    # 5. Cosine-based
    t = np.arange(N)
    for freq in [1, 2, 3]:
        for phase in [0, np.pi/2, np.pi]:
            y = np.cos(2*np.pi*freq*t/N + phase) * 2.0
            initializations.append(y.copy())
    
    for y0 in initializations:
        if time.time() > deadline:
            break
        
        try:
            res = minimize(
                objective_constrained,
                y0,
                method='L-BFGS-B',
                options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-8}
            )
            y_opt = res.x
            s = score_from_y(y_opt)
            if s > best_score:
                best_score = s
                best_h = np.exp(y_opt)
                best_h = best_h / np.sqrt(np.sum(best_h**2))
        except Exception:
            continue
    
    # Nelder-Mead polish from best and a few other points
    nm_initials = [best_h.copy() / np.sqrt(np.sum(best_h**2))]
    
    # Add some diverse points
    for y0 in initializations[:5]:
        h = np.exp(y0)
        h = h / np.sqrt(np.sum(h**2))
        nm_initials.append(h)
    
    # Nelder-Mead on the ratio directly (non-smooth but works well for polish)
    for h0 in nm_initials:
        if time.time() > deadline:
            break
        
        def nm_obj(h):
            return -score(h)
        
        try:
            res = minimize(
                nm_obj,
                h0,
                method='Nelder-Mead',
                options={'maxiter': 500, 'xatol': 1e-10, 'fatol': 1e-12, 'adaptive': True}
            )
            s = -res.fun
            if s > best_score:
                best_score = s
                best_h = res.x.copy()
        except Exception:
            continue
    
    # Final coordinate descent polish
    if time.time() < deadline - 10:
        h = best_h.copy()
        for _ in range(2000):
            if time.time() > deadline:
                break
            improved = False
            for i in range(N):
                old = h[i]
                # Try multiplicative perturbations
                for mult in [0.5, 0.7, 0.85, 1.0, 1.15, 1.3, 2.0, 0.01]:
                    h[i] = max(0.0, old * mult)
                    s = score(h)
                    if s > best_score:
                        best_score = s
                        best_h = h.copy()
                        improved = True
                    else:
                        h[i] = old
                        break
                h[i] = old
            if not improved:
                break
    
    # Ensure nonnegativity and return
    best_h = np.maximum(0, best_h)
    if np.sum(best_h) < 1e-15:
        best_h = np.ones(N)
    
    return list(best_h)

# EVOLVE-BLOCK-END
