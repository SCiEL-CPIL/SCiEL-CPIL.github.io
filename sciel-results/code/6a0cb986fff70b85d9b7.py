import numpy as np
import time

N = 50

# EVOLVE-BLOCK-START
def construct():
    """
    Constructs a sequence of 50 nonnegative heights.
    Strategy: Use 2-point cluster with smooth optimization via soft-max surrogate.
    """
    deadline = time.time() + 145.0  # Safety margin before 150s kill
    
    # Domain: [-1/4, 1/4], 50 equal intervals
    x_min, x_max = -0.25, 0.25
    dx = (x_max - x_min) / N
    x_grid = x_min + np.arange(N + 1) * dx  # N+1 points
    x_centers = x_min + (np.arange(N) + 0.5) * dx  # center of each interval
    
    def compute_ratio(h):
        """Compute the ratio ||h||_2^2 / (||h||_1 * ||h*h||_inf) for discrete h."""
        h = np.asarray(h, dtype=float)
        l1 = np.sum(h)
        if l1 == 0:
            return 0.0
        l2_sq = np.sum(h**2)
        # Autoconvolution: (h*h)[k] = sum_i h[i]*h[i+k]
        conv = np.convolve(h, h, mode='full')
        conv_inf = np.max(conv)
        if conv_inf == 0:
            return 0.0
        return l2_sq / (l1 * conv_inf)
    
    def compute_ratio_smooth(h, beta=100.0):
        """Compute ratio with soft-max surrogate for differentiability."""
        h = np.asarray(h, dtype=float)
        l1 = np.sum(h)
        if l1 == 0:
            return 0.0
        l2_sq = np.sum(h**2)
        conv = np.convolve(h, h, mode='full')
        # Soft-max: (1/beta) * log(sum(exp(beta * conv / max_conv)))
        conv_max = np.max(conv)
        if conv_max == 0:
            return 0.0
        # Use log-sum-exp for numerical stability
        shifted = beta * (conv - conv_max) / conv_max
        log_sum_exp = np.log(np.sum(np.exp(shifted))) + beta
        conv_inf_smooth = np.exp(log_sum_exp) / conv_max * conv_max  # This is just conv_max * sum(exp(shifted))
        # Actually, soft-max of conv values:
        conv_inf_smooth = np.max(conv) * np.exp(log_sum_exp - beta)
        # Simpler: use the actual max but with smooth penalty
        return l2_sq / (l1 * conv_inf_smooth)
    
    def build_h_2point(a1, a2, p1, p2):
        """Build h from two top-hats at positions p1, p2 with amplitudes a1, a2."""
        h = np.zeros(N)
        if a1 > 0:
            mask1 = np.abs(x_centers - p1) <= dx/2.0
            h[mask1] += a1
        if a2 > 0:
            mask2 = np.abs(x_centers - p2) <= dx/2.0
            h[mask2] += a2
        return h
    
    def objective(params):
        """Negative ratio for minimization."""
        a1, a2, p1, p2 = params
        if a1 < 0 or a2 < 0:
            return 1e10
        p1 = np.clip(p1, x_min, x_max)
        p2 = np.clip(p2, x_min, x_max)
        h = build_h_2point(a1, a2, p1, p2)
        return -compute_ratio(h)
    
    # Initial: 2-point cluster at positions 0 and 1
    h_init = np.zeros(N)
    h_init[0] = 2.0/3.0
    h_init[1] = 1.0/3.0
    best_ratio = compute_ratio(h_init)
    best_h = h_init.copy()
    
    # Nelder-Mead optimization with multiple starting points
    from scipy.optimize import minimize
    
    # Diverse starting points
    start_points = [
        np.array([2.0/3.0, 1.0/3.0, x_centers[0], x_centers[1]]),
        np.array([1.0, 1.0, x_centers[0], x_centers[1]]),
        np.array([1.5, 0.5, x_centers[0], x_centers[1]]),
        np.array([0.8, 0.7, x_centers[0], x_centers[1]]),
        np.array([1.0, 1.0, x_centers[0] - dx/2, x_centers[1] + dx/2]),
        np.array([1.0, 1.0, x_centers[0] + dx/2, x_centers[1] - dx/2]),
        np.array([2.0, 1.0, x_centers[0], x_centers[2]]),
        np.array([1.5, 1.5, x_centers[0], x_centers[1] + dx]),
        np.array([3.0/4.0, 1.0/4.0, x_centers[0], x_centers[1]]),
        np.array([1.0, 0.5, x_centers[0], x_centers[1]]),
    ]
    
    for sp in start_points:
        if time.time() > deadline - 10.0:
            break
        try:
            # Phase 1: Coarse optimization
            result = minimize(objective, sp, method='Nelder-Mead', 
                            options={'maxiter': 500, 'xatol': 1e-6, 'fatol': 1e-8})
            if result.fun < -best_ratio:
                a1, a2, p1, p2 = result.x
                if a1 >= 0 and a2 >= 0:
                    h_candidate = build_h_2point(max(0, a1), max(0, a2), p1, p2)
                    r = compute_ratio(h_candidate)
                    if r > best_ratio:
                        best_ratio = r
                        best_h = h_candidate.copy()
                        
                        # Phase 2: High-precision polish
                        if time.time() < deadline - 5.0:
                            x0_fine = np.array([a1, a2, p1, p2])
                            result_fine = minimize(objective, x0_fine, method='Nelder-Mead',
                                                options={'maxiter': 2000, 'xatol': 1e-10, 'fatol': 1e-12})
                            if result_fine.fun < -best_ratio:
                                a1f, a2f, p1f, p2f = result_fine.x
                                if a1f >= 0 and a2f >= 0:
                                    h_fine = build_h_2point(max(0, a1f), max(0, a2f), p1f, p2f)
                                    rf = compute_ratio(h_fine)
                                    if rf > best_ratio:
                                        best_ratio = rf
                                        best_h = h_fine.copy()
        except Exception:
            continue
    
    # Final: Try direct grid search on the two amplitudes for the best positions
    if time.time() < deadline - 5.0:
        # Extract best positions from best_h
        nonzero = np.nonzero(best_h)[0]
        if len(nonzero) == 2:
            p1_best = x_centers[nonzero[0]]
            p2_best = x_centers[nonzero[1]]
            # Fine grid search on amplitudes
            a1_grid = np.linspace(0.1, 3.0, 100)
            a2_grid = np.linspace(0.1, 3.0, 100)
            for a1g in a1_grid:
                for a2g in a2_grid:
                    if time.time() > deadline - 2.0:
                        break
                    h_test = build_h_2point(a1g, a2g, p1_best, p2_best)
                    rt = compute_ratio(h_test)
                    if rt > best_ratio:
                        best_ratio = rt
                        best_h = h_test.copy()
    
    # Ensure non-negativity
    best_h = np.maximum(0, best_h)
    
    # Final safety check
    if np.sum(best_h) == 0:
        best_h[0] = 1.0
    
    return list(best_h)
# EVOLVE-BLOCK-END
