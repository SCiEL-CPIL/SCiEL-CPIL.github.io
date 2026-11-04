"""Rewritten program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
from scipy.optimize import minimize
import time

N = 50
d = 1.0 / (2 * N)  # interval width

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin below 150s limit

    def autoconv_max(h):
        """Compute the maximum of the autoconvolution of h with itself.
        
        For a step function with heights h_i on intervals of width d,
        the autoconvolution at point x is the integral of f(t)f(x-t)dt.
        For discrete step functions, this can be computed via the convolution
        of the height sequence. The peak is at the origin (or near it for
        symmetric functions).
        
        We use np.correlate to compute the discrete autoconvolution, which
        gives us the values at the grid points. The maximum is what we need.
        """
        h = np.asarray(h, dtype=float)
        ac = np.correlate(h, h, mode='full')
        return np.max(ac)

    def ratio(h):
        """Compute ratio ||h||_2^2 / (||h||_1 * ||h*h||_inf)."""
        h = np.asarray(h, dtype=float)
        h = np.maximum(0, h)
        sum_sq = np.sum(h**2)
        sum_h = np.sum(h)
        if sum_h < 1e-15 or sum_sq < 1e-30:
            return 0.0
        max_ac = autoconv_max(h)
        if max_ac < 1e-30:
            return 0.0
        return sum_sq / (d * sum_h * max_ac)

    def neg_ratio_smooth(h, alpha=500.0):
        """Smooth negative objective using soft-max approximation of max."""
        h = np.asarray(h, dtype=float)
        h = np.maximum(0, h)
        sum_sq = np.sum(h**2)
        sum_h = np.sum(h)
        if sum_h < 1e-15 or sum_sq < 1e-30:
            return 1e10
        
        ac = np.correlate(h, h, mode='full')
        ac_max = np.max(ac)
        if ac_max < 1e-30:
            return 1e10
        
        # Soft max approximation
        shifted = alpha * (ac - ac_max)
        # Clip to avoid underflow
        shifted = np.clip(shifted, -50, 0)
        sum_exp = np.sum(np.exp(shifted))
        if sum_exp < 1e-300:
            sum_exp = 1e-300
        log_sum_exp = np.log(sum_exp)
        soft_max = ac_max + log_sum_exp / alpha
        
        if soft_max < 1e-30:
            return 1e10
            
        r = sum_sq / (d * sum_h * soft_max)
        return -r

    def neg_ratio_grad(h, alpha=500.0):
        """Negative smooth ratio with numerical gradient for trust-region methods."""
        val = neg_ratio_smooth(h, alpha)
        # Numerical gradient
        eps = 1e-7
        grad = np.zeros_like(h)
        for i in range(len(h)):
            h_plus = h.copy()
            h_plus[i] += eps
            grad[i] = (neg_ratio_smooth(h_plus, alpha) - val) / eps
        return val, grad

    def local_optimize(h0, method='L-BFGS-B', maxiter=300, maxfun=8000):
        """Run local optimization from h0 and return best found."""
        h0 = np.asarray(h0, dtype=float).copy()
        h0 = np.maximum(0, h0)
        s = np.sum(h0)
        if s > 1e-10:
            h0 = h0 * (N / s)  # Normalize sum to N for better conditioning
        
        bounds = [(0, None)] * N
        
        try:
            if method == 'L-BFGS-B':
                res = minimize(
                    neg_ratio_smooth, h0, method='L-BFGS-B',
                    bounds=bounds,
                    options={'maxiter': maxiter, 'maxfun': maxfun, 
                             'ftol': 1e-14, 'gtol': 1e-10, 'maxls': 50}
                )
            elif method == 'trust-constr':
                res = minimize(
                    neg_ratio_smooth, h0, method='trust-constr',
                    bounds=bounds,
                    options={'maxiter': maxiter // 2, 'gtol': 1e-10,
                             'xtol': 1e-12, 'max_nfev': maxfun}
                )
            else:
                res = minimize(
                    neg_ratio_smooth, h0, method='L-BFGS-B',
                    bounds=bounds,
                    options={'maxiter': maxiter, 'maxfun': maxfun,
                             'ftol': 1e-14, 'gtol': 1e-10}
                )
            h_opt = np.maximum(0, res.x)
            return h_opt
        except Exception:
            return h0

    def coordinate_descent_polish(h, max_iter=200):
        """Polish using coordinate descent with line search."""
        h = np.asarray(h, dtype=float).copy()
        best = ratio(h)
        
        for iteration in range(max_iter):
            if time.time() > deadline - 5:
                break
            improved = False
            for i in range(N):
                old = h[i]
                # Try a range of multipliers
                for mult in [0.0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.1, 1.3, 1.5, 2.0, 3.0, 5.0]:
                    if old > 1e-10:
                        h[i] = max(0.0, old * mult)
                    else:
                        h[i] = mult
                    s = ratio(h)
                    if s > best + 1e-12:
                        best = s
                        improved = True
                    else:
                        h[i] = old
                        break
                h[i] = old
            if not improved:
                break
        
        return h, best

    def generate_gaussian(center, width):
        t = np.arange(N)
        return np.exp(-0.5 * ((t - center) / width)**2)

    def generate_lorentzian(center, width):
        t = np.arange(N)
        return 1.0 / (1.0 + ((t - center) / width)**2)

    def generate_bimodal(center, gap, width):
        t = np.arange(N)
        h1 = np.exp(-0.5 * ((t - (center - gap)) / width)**2)
        h2 = np.exp(-0.5 * ((t - (center + gap)) / width)**2)
        return h1 + h2

    def generate_box(center, half_width):
        t = np.arange(N)
        h = np.zeros(N)
        mask = np.abs(t - center) <= half_width
        h[mask] = 1.0
        return h

    def generate_triangle(center, half_width):
        t = np.arange(N)
        h = np.maximum(0, 1.0 - np.abs(t - center) / half_width)
        return h

    # Phase 1: Grid search over parametric families
    best_h = np.zeros(N)
    best_h[N // 2] = 1.0
    best_score = ratio(best_h)

    # Single Gaussian/Lorentzian/Box/Triangle with varying center and width
    centers = np.linspace(0, N-1, 25)
    widths = np.logspace(-0.5, 2.0, 40)
    
    for center in centers:
        if time.time() > deadline - 60:
            break
        for width in widths:
            if time.time() > deadline - 60:
                break
            
            # Gaussian
            h = generate_gaussian(center, width)
            s = ratio(h)
            if s > best_score:
                best_score = s
                best_h = h.copy()
            
            # Lorentzian
            h = generate_lorentzian(center, width)
            s = ratio(h)
            if s > best_score:
                best_score = s
                best_h = h.copy()
            
            # Box
            hw = max(1, int(width))
            h = generate_box(center, hw)
            s = ratio(h)
            if s > best_score:
                best_score = s
                best_h = h.copy()
            
            # Triangle
            hw = max(1, width)
            h = generate_triangle(center, hw)
            s = ratio(h)
            if s > best_score:
                best_score = s
                best_h = h.copy()

    # Bimodal distributions
    for center in [N//4, N//2, 3*N//4]:
        if time.time() > deadline - 40:
            break
        for gap in [3, 5, 7, 10, 15, 20]:
            for width in [1.0, 2.0, 3.0, 5.0]:
                h = generate_bimodal(center, gap, width)
                s = ratio(h)
                if s > best_score:
                    best_score = s
                    best_h = h.copy()

    # Phase 2: Local optimization from diverse starting points
    starts = [best_h.copy()]
    
    # Add various parametric starts
    for center in [N//4, N//2, 3*N//4]:
        for width in [1.0, 2.0, 5.0]:
            starts.append(generate_gaussian(center, width))
            starts.append(generate_lorentzian(center, width))
    
    # Random smooth starts
    rng = np.random.default_rng(123)
    for _ in range(10):
        h = np.abs(rng.normal(1.0, 0.8, N))
        # Smooth
        kernel = np.array([1, 2, 3, 2, 1]) / 9.0
        h = np.convolve(h, kernel, mode='same')
        starts.append(h)
    
    # Uniform and other simple starts
    starts.append(np.ones(N))
    starts.append(np.zeros(N))
    starts[N//2] = 1.0

    for h0 in starts:
        if time.time() > deadline - 20:
            break
        
        h0 = np.asarray(h0, dtype=float)
        h0 = np.maximum(0, h0)
        
        # L-BFGS-B optimization
        h_opt = local_optimize(h0, method='L-BFGS-B', maxiter=400, maxfun=10000)
        s_opt = ratio(h_opt)
        if s_opt > best_score:
            best_score = s_opt
            best_h = h_opt.copy()
        
        # Trust-constr from the L-BFGS-B result
        if time.time() < deadline - 15:
            h_opt2 = local_optimize(h_opt, method='trust-constr', maxiter=200, maxfun=5000)
            s_opt2 = ratio(h_opt2)
            if s_opt2 > best_score:
                best_score = s_opt2
                best_h = h_opt2.copy()

    # Phase 3: Coordinate descent polish
    if time.time() < deadline - 10:
        best_h, best_score = coordinate_descent_polish(best_h, max_iter=300)

    # Phase 4: Final high-precision optimization
    if time.time() < deadline - 5:
        h0 = best_h.copy()
        s = np.sum(h0)
        if s > 1e-10:
            h0 = h0 * (N / s)
        
        # Multiple final optimization attempts with different methods
        for method in ['L-BFGS-B', 'trust-constr', 'L-BFGS-B']:
            if time.time() > deadline - 2:
                break
            h_opt = local_optimize(h0, method=method, maxiter=500, maxfun=15000)
            s_opt = ratio(h_opt)
            if s_opt > best_score:
                best_score = s_opt
                best_h = h_opt.copy()
                h0 = best_h.copy()

    # Phase 5: Final coordinate descent
    if time.time() < deadline - 2:
        best_h, best_score = coordinate_descent_polish(best_h, max_iter=100)

    # Ensure nonnegativity
    best_h = np.maximum(0, best_h)
    if np.sum(best_h) < 1e-15:
        best_h = np.zeros(N)
        best_h[N // 2] = 1.0

    return list(best_h)

# EVOLVE-BLOCK-END
