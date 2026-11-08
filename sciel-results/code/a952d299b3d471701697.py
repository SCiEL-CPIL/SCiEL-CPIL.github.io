import numpy as np
import time

N = 50

# EVOLVE-BLOCK-START
def construct():
    """
    Constructs a sequence of 50 nonnegative heights.
    Strategy: Use top-hat basis functions with analytical convolution,
    then optimize weights and positions via gradient descent.
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
        # For non-negative h, the peak is at lag 0: sum h[i]^2 = l2_sq
        # But wait - for cross-correlation or different conv definition?
        # Let's compute full linear convolution and take max
        conv = np.convolve(h, h, mode='full')
        conv_inf = np.max(conv)
        if conv_inf == 0:
            return 0.0
        return l2_sq / (l1 * conv_inf)
    
    def top_hat_basis(center, width, x_grid):
        """Create a top-hat basis function: 1 inside [center-width/2, center+width/2], 0 outside."""
        half_w = width / 2.0
        mask = (x_grid >= center - half_w) & (x_grid <= center + half_w)
        return mask.astype(float)
    
    def build_h_from_basis(weights, centers, widths, x_centers):
        """Build h as superposition of top-hat basis functions."""
        h = np.zeros(N)
        for w, c, wd in zip(weights, centers, widths):
            if w <= 0:
                continue
            # Top hat: 1 where |x - c| <= wd/2
            mask = np.abs(x_centers - c) <= wd / 2.0
            h[mask] += w
        return h
    
    def ratio_from_basis(weights, centers, widths):
        """Compute ratio for given basis parameters."""
        h = build_h_from_basis(weights, centers, widths, x_centers)
        return compute_ratio(h)
    
    # Start with 2-point cluster as known best
    # In the discrete grid, this corresponds to two adjacent intervals
    # Let's parameterize: two top-hats at positions p1, p2 with widths w1, w2 and amplitudes a1, a2
    
    best_ratio = 0.0
    best_weights = np.array([2.0/3.0, 1.0/3.0])
    best_centers = np.array([x_centers[0], x_centers[1]])
    best_widths = np.array([dx, dx])
    
    # Initial: 2-point cluster at positions 0 and 1
    h_init = np.zeros(N)
    h_init[0] = 2.0/3.0
    h_init[1] = 1.0/3.0
    best_ratio = compute_ratio(h_init)
    
    # Try to improve via local optimization of weights and positions
    # Use simple gradient-free optimization (Nelder-Mead style)
    
    def objective(params):
        """Negative ratio for minimization."""
        a1, a2, p1, p2 = params
        if a1 < 0 or a2 < 0:
            return 1e10
        # Clip positions to valid range
        p1 = np.clip(p1, x_min, x_max)
        p2 = np.clip(p2, x_min, x_max)
        
        # Build h from two top-hats
        h = np.zeros(N)
        # Top hat 1
        mask1 = np.abs(x_centers - p1) <= dx/2.0
        h[mask1] += a1
        # Top hat 2
        mask2 = np.abs(x_centers - p2) <= dx/2.0
        h[mask2] += a2
        
        return -compute_ratio(h)
    
    # Nelder-Mead optimization
    from scipy.optimize import minimize
    
    x0 = np.array([2.0/3.0, 1.0/3.0, x_centers[0], x_centers[1]])
    
    # Try multiple starting points
    start_points = [
        x0,
        np.array([1.0, 1.0, x_centers[0], x_centers[1]]),
        np.array([1.5, 0.5, x_centers[0], x_centers[1]]),
        np.array([0.8, 0.7, x_centers[0], x_centers[1]]),
        np.array([1.0, 1.0, x_centers[0] - dx/2, x_centers[1] + dx/2]),
        np.array([1.0, 1.0, x_centers[0] + dx/2, x_centers[1] - dx/2]),
    ]
    
    for sp in start_points:
        if time.time() > deadline - 5.0:
            break
        try:
            result = minimize(objective, sp, method='Nelder-Mead', 
                            options={'maxiter': 1000, 'xatol': 1e-8, 'fatol': 1e-10})
            if result.fun < -best_ratio:
                best_ratio = -result.fun
                a1, a2, p1, p2 = result.x
                if a1 < 0 or a2 < 0:
                    continue
                # Build final h
                h_candidate = np.zeros(N)
                mask1 = np.abs(x_centers - p1) <= dx/2.0
                h_candidate[mask1] += max(0, a1)
                mask2 = np.abs(x_centers - p2) <= dx/2.0
                h_candidate[mask2] += max(0, a2)
                
                # Verify
                r = compute_ratio(h_candidate)
                if r > best_ratio:
                    best_ratio = r
                    best_h = h_candidate.copy()
        except Exception:
            continue
    
    # If optimization didn't improve, use the initial 2-point cluster
    if 'best_h' not in dir():
        best_h = np.zeros(N)
        best_h[0] = 2.0/3.0
        best_h[1] = 1.0/3.0
    
    # Ensure non-negativity
    best_h = np.maximum(0, best_h)
    
    # Final safety check: make sure we have a valid result
    if np.sum(best_h) == 0:
        best_h[0] = 1.0
    
    return list(best_h)
# EVOLVE-BLOCK-END
