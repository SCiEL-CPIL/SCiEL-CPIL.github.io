import numpy as np
import time
import scipy.optimize
import scipy.spatial.distance

def min_max_dist_dim2_16() -> np.ndarray:
    """
    Creates 16 points in 2 dimensions in order to maximize the ratio of minimum to maximum distance.

    Returns
        points: np.ndarray of shape (16,2) containing the (x,y) coordinates of the (x,y) points.

    """
    n = 16
    d = 2
    
    # Start time for deadline guard
    start_time = time.time()
    deadline = 600 - 30  # 570 seconds to be safe

    def get_distances(points):
        """Calculate pairwise distances for a (N, 2) array."""
        return scipy.spatial.distance.pdist(points)

    def score(points):
        """Calculate the raw score R = (d_min / d_max)^2."""
        dists = get_distances(points)
        d_min = np.min(dists)
        d_max = np.max(dists)
        if d_max == 0:
            return 0.0
        return (d_min / d_max) ** 2

    def smooth_objective(x):
        """
        Smooth surrogate objective.
        We want to maximize d_min / d_max.
        This is equivalent to maximizing log(d_min) - log(d_max).
        To handle non-differentiability and zeros, we use a soft-min and soft-max.
        
        Let D^2 be the matrix of squared distances.
        soft_min(D^2) approximates min(D^2).
        soft_max(D^2) approximates max(D^2).
        
        We minimize -log(soft_min) + log(soft_max).
        """
        points = x.reshape(n, d)
        dists_sq = scipy.spatial.distance.pdist(points) ** 2
        # Add small epsilon to avoid log(0)
        eps = 1e-12
        dists_sq_safe = dists_sq + eps
        
        # Temperature for soft-min/max
        # Lower temperature -> sharper min/max
        # We want a balance. 
        # Using log-sum-exp for max: log(sum(exp(x)))
        # Using -log-sum-exp(-x) for min: -log(sum(exp(-x)))
        
        # To prevent overflow in exp, we subtract max for max-sum-exp and add min for min-sum-exp
        # But for simplicity and stability with 120 elements, we can just use a fixed temp if values are bounded.
        # However, distances can vary. Let's normalize by current max to keep scale invariance in the exponent.
        
        d_max_sq = np.max(dists_sq_safe)
        if d_max_sq < eps:
            return 0.0
            
        # Normalize distances to be in [0, 1] roughly for the exponential calculation
        # This makes the soft-min/max behavior consistent regardless of scale
        norm_dists = dists_sq_safe / d_max_sq
        
        # Soft Max of normalized dists is just log(sum(exp(norm_dists)))
        # Soft Min of normalized dists is -log(sum(exp(-norm_dists)))
        
        # We want to maximize min/max.
        # log(min) - log(max)
        # Using normalized: log(min_norm) - log(max_norm)
        # max_norm is close to 1. min_norm is the target.
        
        temp = 0.1
        # log(sum(exp(x/temp))) is the soft max
        # log(sum(exp(-x/temp))) is related to soft min
        
        # Let's just use a simpler penalty: 
        # Minimize (1/d_min - 1/d_max) ? No, that's not scale invariant nicely.
        
        # Let's stick to the log-ratio with soft operators.
        # R = (d_min/d_max)^2. Maximize log(R) = 2(log(d_min) - log(d_max)).
        
        # soft_min = 1 / (sum(exp(dists/temp))) ? No.
        
        # Let's use the standard log-sum-exp trick.
        # log_max = log(sum(exp(dists_sq / temp)))
        # log_min = -log(sum(exp(-dists_sq / temp)))
        
        # To avoid overflow:
        m_max = np.max(dists_sq)
        m_min = np.min(dists_sq)
        
        log_max = m_max + temp * np.log(np.sum(np.exp((dists_sq - m_max) / temp)))
        log_min = m_min - temp * np.log(np.sum(np.exp((m_min - dists_sq) / temp)))
        
        # We want to maximize log_min - log_max
        # So we minimize log_max - log_min
        return log_max - log_min

    # 1. Generate diverse initial candidates
    # Candidate 1: 4x4 Grid
    grid_x, grid_y = np.meshgrid(np.linspace(-1, 1, 4), np.linspace(-1, 1, 4))
    grid_pts = np.column_stack((grid_x.ravel(), grid_y.ravel()))
    
    # Candidate 2: Circle (regular polygon)
    angles = np.linspace(0, 2*np.pi, n, endpoint=False)
    circle_pts = np.column_stack((np.cos(angles), np.sin(angles)))
    
    # Candidate 3: Random
    np.random.seed(42)
    rand_pts = np.random.randn(n, d)
    
    candidates = [grid_pts, circle_pts, rand_pts]
    
    best_points = None
    best_score = -np.inf

    # 2. Optimize using Smooth Surrogate + Simulated Annealing / Local Refinement
    # Since the smooth objective is easier to optimize, we can use L-BFGS-B or Nelder-Mead on it.
    # Then we refine with the true objective.
    
    for i, init_pts in enumerate(candidates):
        if time.time() > deadline:
            break
            
        # Step 2a: Optimize the smooth surrogate
        # Use Nelder-Mead as it doesn't require gradients and is robust for this kind of problem
        try:
            # Scale the initial points to a reasonable range
            init_flat = init_pts.flatten()
            
            # Run a local optimizer on the smooth objective
            # We use a moderate number of iterations
            res_smooth = scipy.optimize.minimize(
                smooth_objective, 
                init_flat, 
                method='Nelder-Mead',
                options={'maxiter': 1000, 'xatol': 1e-4, 'fatol': 1e-4}
            )
            
            # Step 2b: Refine with the true objective (non-smooth)
            # The smooth optimization should have brought us close to a local optimum of the true metric.
            # Now we do a local search on the true metric.
            
            # We can use a simple coordinate descent or Nelder-Mead on the true metric.
            # Nelder-Mead on the true metric is non-smooth but often works well for packing problems if started close.
            
            res_true = scipy.optimize.minimize(
                lambda x: -score(x.reshape(n, d)), 
                res_smooth.x, 
                method='Nelder-Mead',
                options={'maxiter': 2000, 'xatol': 1e-6, 'fatol': 1e-6}
            )
            
            current_score = -res_true.fun
            if current_score > best_score:
                best_score = current_score
                best_points = res_true.x.reshape(n, d)
                
            # Also check the smooth result itself, just in case
            smooth_score = score(res_smooth.x.reshape(n, d))
            if smooth_score > best_score:
                best_score = smooth_score
                best_points = res_smooth.x.reshape(n, d)
                
        except Exception as e:
            # Fallback to initial candidate if optimization fails
            init_score = score(init_pts)
            if init_score > best_score:
                best_score = init_score
                best_points = init_pts

    # Fallback if nothing was found
    if best_points is None:
        best_points = grid_pts

    # Final global refinement pass:
    # Sometimes a single Nelder-Mead run gets stuck.
    # If we have time left, try a few random restarts from the best point found,
    # perturbing it slightly and running a short local search.
    
    if time.time() < deadline - 60:
        current_best_flat = best_points.flatten()
        for restart in range(10):
            if time.time() > deadline - 30:
                break
            # Perturb
            noise = np.random.normal(0, 0.1, current_best_flat.shape)
            perturbed = current_best_flat + noise
            
            try:
                res_restart = scipy.optimize.minimize(
                    lambda x: -score(x.reshape(n, d)), 
                    perturbed, 
                    method='Nelder-Mead',
                    options={'maxiter': 500, 'xatol': 1e-6, 'fatol': 1e-6}
                )
                restart_score = -res_restart.fun
                if restart_score > best_score:
                    best_score = restart_score
                    best_points = res_restart.x.reshape(n, d)
            except Exception:
                pass

    return best_points
