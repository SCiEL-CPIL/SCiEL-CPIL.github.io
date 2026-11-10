import numpy as np
from scipy.spatial.distance import pdist
import time
from scipy.optimize import minimize

def min_max_dist_dim2_16() -> np.ndarray:
    """
    Maximizes the min/max distance ratio for 16 points in 2D.
    
    Strategy:
    1. Use a smooth surrogate objective: 
       log(soft_min(d)) - log(soft_max(d)) 
       approximates log(d_min/d_max) but is differentiable.
       Minimizing the negative of this maximizes the ratio.
    2. Initialize with several symmetric configurations (4x4 grid, ring, etc.).
    3. Use L-BFGS-B for gradient-based optimization.
    4. Return the best point set found within the time limit.
    """
    start_time = time.time()
    deadline = start_time + 580.0  # Safety margin

    def compute_score(points):
        """Compute the exact score R = (d_min / d_max)^2."""
        if points.shape != (16, 2):
            return 0.0
        d = pdist(points)
        if d.size == 0:
            return 0.0
        d_min = np.min(d)
        d_max = np.max(d)
        if d_max == 0:
            return 0.0
        ratio = d_min / d_max
        return ratio * ratio

    def smooth_objective(x, beta=100.0):
        """
        Smooth surrogate for minimizing -log(d_min/d_max).
        Uses log-sum-exp tricks for numerical stability.
        """
        pts = x.reshape(16, 2)
        d = pdist(pts)
        
        # Avoid log(0)
        d_safe = np.maximum(d, 1e-12)
        
        # Soft min: log(1/beta * sum(exp(-beta*d))) approx -d_min
        # Soft max: log(1/beta * sum(exp(beta*d))) approx d_max
        # We want to maximize d_min/d_max, so minimize -log(d_min) + log(d_max)
        # log(d_min) approx -soft_min_val
        # log(d_max) approx soft_max_val
        
        # Stable log-sum-exp
        def logsumexp(a):
            a_max = np.max(a)
            return a_max + np.log(np.sum(np.exp(a - a_max)))
        
        # soft_min_val = -1/beta * logsumexp(-beta * d)
        # soft_max_val = 1/beta * logsumexp(beta * d)
        
        # Objective: minimize -log(soft_min_val) + log(soft_max_val)
        # This is equivalent to minimizing log(soft_max_val / soft_min_val)
        
        # Let's just compute the surrogate ratio directly for simplicity and stability
        # Surrogate min: exp(-logsumexp(-beta*d)/beta)
        # Surrogate max: exp(logsumexp(beta*d)/beta)
        
        # To avoid underflow/overflow, work in log space if possible, 
        # but for optimization we just need a smooth scalar.
        
        # A simpler smooth approximation often used:
        # min_d approx (prod(d_i))^(1/N) is geometric mean, not min.
        # Let's stick to the log-sum-exp based soft min/max.
        
        log_soft_min = -np.log(np.sum(np.exp(-beta * d_safe)) / beta)
        log_soft_max = np.log(np.sum(np.exp(beta * d_safe)) / beta)
        
        # We want to maximize min/max, so minimize -log(min) + log(max)
        # Which is log(max) - log(min)
        return log_soft_max - log_soft_min

    def objective_wrapper(x):
        """Wrapper to ensure finite output."""
        val = smooth_objective(x)
        if not np.isfinite(val):
            return 1e10
        return val

    # Generate diverse initial points
    def get_initial_points():
        inits = []
        
        # 1. Regular 4x4 Grid
        coords = np.array([[-1.5, -0.5, 0.5, 1.5] * 4,
                           [-1.5, -1.5, -1.5, -1.5,
                            -0.5, -0.5, -0.5, -0.5,
                             0.5,  0.5,  0.5,  0.5,
                             1.5,  1.5,  1.5,  1.5]]).T
        inits.append(coords.flatten())

        # 2. Two concentric squares (8 points each)
        # Inner square
        r_in = 0.5
        angles_in = np.linspace(0, 2*np.pi, 8, endpoint=False) + np.pi/8
        inner = np.column_stack([r_in * np.cos(angles_in), r_in * np.sin(angles_in)])
        # Outer square
        r_out = 1.0
        angles_out = np.linspace(0, 2*np.pi, 8, endpoint=False)
        outer = np.column_stack([r_out * np.cos(angles_out), r_out * np.sin(angles_out)])
        pts2 = np.vstack([inner, outer])
        inits.append(pts2.flatten())

        # 3. Single ring with alternating radii? No, let's try a perturbed grid
        np.random.seed(42)
        noise = np.random.normal(0, 0.1, 32)
        inits.append((coords + noise).flatten())
        
        # 4. Hexagonal-like packing?
        # Just use a few random perturbations of the grid
        for i in range(5):
            np.random.seed(100 + i)
            noise = np.random.normal(0, 0.2, 32)
            inits.append((coords + noise).flatten())
            
        return inits

    best_points = None
    best_score = -np.inf
    best_x = None

    inits = get_initial_points()
    
    # Optimize each initial point
    for i, x0 in enumerate(inits):
        if time.time() > deadline - 60:
            break
            
        try:
            res = minimize(objective_wrapper, x0, method='L-BFGS-B', 
                           options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-8})
            
            if res.success or res.fun < 1e9:
                pts = res.x.reshape(16, 2)
                # Evaluate exact score
                score = compute_score(pts)
                
                if score > best_score:
                    best_score = score
                    best_points = pts
                    best_x = res.x
        except Exception:
            continue

    # If no good result, fall back to the best initial point's exact score
    if best_points is None:
        for x0 in inits:
            pts = x0.reshape(16, 2)
            score = compute_score(pts)
            if score > best_score:
                best_score = score
                best_points = pts

    # Final polish with Nelder-Mead on the best candidate if time permits
    if best_points is not None and time.time() < deadline - 30:
        try:
            def obj_exact(x):
                pts = x.reshape(16, 2)
                return -compute_score(pts)
            
            res_nm = minimize(obj_exact, best_points.flatten(), method='Nelder-Mead',
                              options={'maxiter': 200, 'fatol': 1e-10, 'xatol': 1e-10})
            if res_nm.success:
                final_pts = res_nm.x.reshape(16, 2)
                final_score = compute_score(final_pts)
                if final_score > best_score:
                    best_points = final_pts
                    best_score = final_score
        except Exception:
            pass

    if best_points is None:
        # Absolute fallback
        best_points = np.zeros((16, 2))
        
    return best_points
