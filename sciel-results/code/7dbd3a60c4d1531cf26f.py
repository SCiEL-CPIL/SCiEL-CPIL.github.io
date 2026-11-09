# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.spatial.distance import pdist
from scipy.optimize import minimize

def _score(points):
    """Compute (d_min / d_max)^2 for a set of points."""
    if points.shape != (14, 3):
        return 0.0
    if not np.all(np.isfinite(points)):
        return 0.0
    dists = pdist(points)
    if len(dists) == 0:
        return 0.0
    d_min = np.min(dists)
    d_max = np.max(dists)
    if d_max == 0:
        return 0.0
    return (d_min / d_max) ** 2

def _force_model(points_flat, n_points, dim):
    """
    Objective function for minimization.
    We want to maximize (d_min / d_max).
    Equivalent to minimizing log(d_max / d_min) or similar.
    
    A robust heuristic:
    1. Center the points (subtract mean) to fix translation.
    2. Scale points so that the max distance is 1 (fixes scale).
    3. Then, we want to maximize the min distance.
    4. This is equivalent to minimizing -d_min.
    
    However, scaling inside the objective makes it non-smooth if d_max crosses 0 or if we don't handle it carefully.
    
    Alternative Force-Directed Approach:
    Let's use a potential energy approach where points repel each other.
    U = sum_{i<j} 1/d_{ij}^2 is common but doesn't directly constrain d_max.
    
    Better: 
    We want to maximize r = d_min / d_max.
    Let's fix the scale by normalizing the points such that d_max = 1.
    Then we just need to maximize d_min.
    
    Let x be the flattened points.
    1. p = x.reshape(n, dim)
    2. p = p - mean(p)
    3. d_max = max(pdist(p))
    4. if d_max == 0: return 0 (penalty)
    5. p = p / d_max
    6. d_min = min(pdist(p))
    7. return -d_min  (we minimize negative min distance)
    
    This function is non-differentiable at the points where the identity of max/min distance pairs changes,
    but gradient-based optimizers like Nelder-Mead or COBYLA can handle it, or we can use a smooth approximation.
    Actually, L-BFGS-B requires gradients. We can use a smooth approximation for 1/d or just use Nelder-Mead which doesn't require gradients.
    Given the small dimension (42), Nelder-Mead is viable but might be slow.
    
    Let's try a different smooth objective:
    Minimize sum(1/d_ij^2) subject to max(d_ij) <= 1? No, that's constrained.
    
    Let's stick to the "normalize then maximize min" but use a smooth proxy for min.
    Soft-min: -log(sum(exp(-k * d_ij))) / k. As k -> inf, this approaches min(d_ij).
    So we minimize: - ( -log(sum(exp(-k * d_ij))) / k ) = log(sum(exp(-k * d_ij))) / k
    Plus we need to handle the normalization d_max=1.
    
    Actually, simpler:
    Use a repulsive force that is very strong for close pairs and weak for far pairs, 
    but also include a term that penalizes large distances to keep the cluster compact.
    
    Let's use the objective:
    f(x) = sum_{i<j} (1/d_ij^2) + lambda * (d_max)^2
    This is tricky.
    
    Let's go back to the deterministic local search on the normalized space.
    We will use Nelder-Mead on the function:
    def obj(x):
        p = x.reshape(14, 3)
        p = p - p.mean(axis=0)
        d = pdist(p)
        if np.any(d == 0): return 1e6
        d_max = np.max(d)
        p = p / d_max
        d_min = np.min(pdist(p))
        return -d_min
    
    This is non-smooth. Nelder-Mead is derivative-free and robust to this.
    """
    p = points_flat.reshape(n_points, dim)
    # Center
    p = p - np.mean(p, axis=0)
    # Compute distances
    d = pdist(p)
    if np.any(d == 0):
        return 1e6
    d_max = np.max(d)
    if d_max == 0:
        return 1e6
    # Normalize
    p = p / d_max
    d_min = np.min(pdist(p))
    # We want to maximize d_min, so minimize -d_min
    return -d_min

def min_max_dist_dim3_14() -> np.ndarray:
    """
    Creates 14 points in 3 dimensions in order to maximize the ratio of minimum to maximum distance.
    
    Returns
        points: np.ndarray of shape (14,3) containing the coordinates of the 14 points.
    """
    start_time = time.time()
    deadline = start_time + 600  # Safety margin before 630s limit
    
    n_points = 14
    dim = 3
    
    best_score = 0.0
    best_points = None
    
    # Generate multiple random initializations
    # 1. Random on sphere
    # 2. Fibonacci sphere
    # 3. Random simplex-like
    # 4. Small perturbations of known good structures (if we had them, but we don't)
    
    def init_random():
        return np.random.randn(n_points, dim)
    
    def init_fibonacci():
        # Fibonacci sphere for n points
        n = n_points
        i = np.arange(0, n) + 0.5
        phi = np.arccos(1 - 2 * i / n)
        theta = np.pi * (1 + 5 ** 0.5) * i
        x = np.cos(theta) * np.sin(phi)
        y = np.sin(theta) * np.sin(phi)
        z = np.cos(phi)
        return np.column_stack((x, y, z))
    
    def init_grid_perturb():
        # Start with a rough grid or simplex and perturb
        pts = []
        # Simplex in 3D has 4 vertices. 14 points is more.
        # Let's just use random but with a tighter bound
        return np.random.uniform(-1, 1, (n_points, dim))
        
    initializers = [init_random, init_fibonacci, init_grid_perturb]
    
    # We will run local optimization from multiple starts
    # Nelder-Mead is slow in 42D. Let's limit the number of starts and max iterations.
    # Or use a faster derivative-free method?
    # COBYLA might be better?
    # Let's try Nelder-Mead with a limited number of iterations per start.
    
    num_starts = 200 # Adjust based on time
    max_iter_per_start = 2000
    
    for i in range(num_starts):
        if time.time() > deadline - 30:
            break
            
        # Select initializer
        init_func = initializers[i % len(initializers)]
        x0 = init_func()
        
        # Flatten
        x0_flat = x0.flatten()
        
        # Run Nelder-Mead
        try:
            res = minimize(
                _force_model, 
                x0_flat, 
                args=(n_points, dim), 
                method='Nelder-Mead', 
                options={'maxiter': max_iter_per_start, 'xatol': 1e-6, 'fatol': 1e-6}
            )
            
            if res.success or res.fun < 0: # Check if it found something valid (negative fun means positive d_min)
                p_opt = res.x.reshape(n_points, dim)
                # Re-center and normalize for final score calculation
                p_opt = p_opt - np.mean(p_opt, axis=0)
                d_max = np.max(pdist(p_opt))
                if d_max > 0:
                    p_opt = p_opt / d_max
                
                s = _score(p_opt)
                if s > best_score:
                    best_score = s
                    best_points = p_opt.copy()
                    
        except Exception:
            continue
            
    # If no optimization succeeded or score is 0, return a valid fallback
    if best_points is None or best_score == 0:
        # Return points on a sphere
        best_points = init_fibonacci()
        
    return best_points

# EVOLVE-BLOCK-END
