# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.spatial.distance import pdist
from scipy.optimize import minimize, differential_evolution, minimize_scalar
from itertools import combinations

def min_max_dist_dim2_16() -> np.ndarray:
    """
    Creates 16 points in 2 dimensions in order to maximize the ratio of minimum to maximum distance.

    Returns
        points: np.ndarray of shape (16,2) containing the (x,y) coordinates of the 16 points.

    """

    n = 16
    d = 2
    start_time = time.time()
    deadline = start_time + 580

    def exact_score_from_pts(pts):
        pts = np.asarray(pts, dtype=np.float64)
        if pts.shape != (n, d):
            pts = pts.reshape(n, d)
        dists = pdist(pts)
        d_max = np.max(dists)
        if d_max < 1e-12:
            return 0.0
        d_min = np.min(dists)
        return (d_min / d_max) ** 2

    def normalized_min_dist(pts):
        """
        Return the minimum pairwise distance after normalizing so that
        the maximum pairwise distance is 1.0. This eliminates scale invariance
        and makes the objective smooth and bounded.
        """
        pts = np.asarray(pts, dtype=np.float64)
        if pts.shape != (n, d):
            pts = pts.reshape(n, d)
        dists = pdist(pts)
        d_max = np.max(dists)
        if d_max < 1e-12:
            return 0.0
        return np.min(dists) / d_max

    def normalized_min_dist_soft(pts, t=0.05):
        """
        Soft approximation of normalized minimum distance for gradient-based methods.
        Uses log-sum-exp to approximate the minimum in a differentiable way.
        """
        pts = np.asarray(pts, dtype=np.float64)
        if pts.shape != (n, d):
            pts = pts.reshape(n, d)
        dists = pdist(pts)
        d_max = np.max(dists)
        if d_max < 1e-12:
            return 0.0
        dists_norm = dists / d_max
        # Soft minimum via log-sum-exp with negative sign
        # min(x) ≈ -log(sum(exp(-x/t)))/n * t for small t
        max_val = np.max(-dists_norm / t)
        soft_min = -t * np.log(np.sum(np.exp(-dists_norm / t - max_val)) / len(dists_norm)) - t * max_val
        return soft_min

    def objective_normalized(x, t=0.05):
        """
        Objective for optimization: negative of soft normalized minimum distance.
        We minimize this to maximize the normalized minimum distance.
        """
        pts = x.reshape(n, d)
        return -normalized_min_dist_soft(pts, t)

    def objective_exact(x):
        """
        Exact objective: negative of normalized minimum distance.
        """
        pts = x.reshape(n, d)
        return -normalized_min_dist(pts)

    best_x = None
    best_score = 0.0
    best_norm_min = 0.0

    def update_best(x, norm_min_val):
        nonlocal best_x, best_score, best_norm_min
        score = norm_min_val ** 2
        if score > best_score:
            best_score = score
            best_norm_min = norm_min_val
            best_x = x.copy()

    # Generate initial points from various structured configurations
    def generate_hex_grid(s=1.0, aspect=1.0):
        """Generate a 4x4 hexagonal grid."""
        a1 = np.array([s, 0.0])
        a2 = np.array([s * aspect * 0.5, s * aspect * np.sqrt(3) / 2])
        pts = np.zeros((n, 2))
        for i in range(4):
            for j in range(4):
                idx = i * 4 + j
                pts[idx] = i * a1 + j * a2
        pts -= np.mean(pts, axis=0)
        return pts

    def generate_square_grid(s=1.0):
        """Generate a 4x4 square grid."""
        pts = np.zeros((n, 2))
        for i in range(4):
            for j in range(4):
                idx = i * 4 + j
                pts[idx] = [i * s, j * s]
        pts -= np.mean(pts, axis=0)
        return pts

    def generate_circle(pts_count=16):
        """Generate points on a circle."""
        angles = np.linspace(0, 2 * np.pi, pts_count, endpoint=False)
        pts = np.column_stack([np.cos(angles), np.sin(angles)])
        return pts

    def generate_random_points(seed=0):
        """Generate random points in a unit square."""
        rng = np.random.RandomState(seed)
        pts = rng.uniform(-1, 1, (n, d))
        return pts

    # Phase 1: Generate diverse initial configurations
    initial_configs = []
    
    # Hexagonal grids with different parameters
    for s in [0.8, 1.0, 1.2]:
        for aspect in [0.9, 1.0, 1.1]:
            initial_configs.append(generate_hex_grid(s, aspect))
    
    # Square grid
    for s in [0.8, 1.0, 1.2]:
        initial_configs.append(generate_square_grid(s))
    
    # Circle
    initial_configs.append(generate_circle())
    
    # Random configurations
    for seed in range(10):
        initial_configs.append(generate_random_points(seed))
    
    # Evaluate all initial configurations
    for pts in initial_configs:
        norm_min = normalized_min_dist(pts)
        update_best(pts.ravel(), norm_min)
    
    # Phase 2: L-BFGS-B optimization from each promising initial configuration
    # Sort by score and take top configurations
    scored_configs = []
    for pts in initial_configs:
        norm_min = normalized_min_dist(pts)
        scored_configs.append((norm_min, pts))
    scored_configs.sort(key=lambda x: -x[0])
    
    top_configs = [pts for _, pts in scored_configs[:8]]
    
    for config_idx, init_pts in enumerate(top_configs):
        if time.time() > deadline - 30:
            break
        
        try:
            # Use L-BFGS-B with the soft objective for smoothness
            res = minimize(
                objective_normalized,
                init_pts.ravel(),
                method='L-BFGS-B',
                args=(0.05,),
                options={'maxiter': 2000, 'ftol': 1e-14, 'gtol': 1e-10, 'maxls': 50}
            )
            
            res_pts = res.x.reshape(n, d)
            norm_min = normalized_min_dist(res_pts)
            update_best(res.x, norm_min)
            
            # Refine with a smaller temperature for sharper minimum
            res2 = minimize(
                objective_normalized,
                res.x,
                method='L-BFGS-B',
                args=(0.02,),
                options={'maxiter': 1000, 'ftol': 1e-14, 'gtol': 1e-10, 'maxls': 50}
            )
            
            res2_pts = res2.x.reshape(n, d)
            norm_min2 = normalized_min_dist(res2_pts)
            update_best(res2.x, norm_min2)
            
        except Exception as e:
            continue
    
    # Phase 3: Differential Evolution for global search
    if time.time() < deadline - 60:
        try:
            # Use a reasonable bound for the optimization
            bounds = [(-2.0, 2.0)] * (n * d)
            
            # Use the soft objective for DE (it's smoother)
            res_de = differential_evolution(
                objective_normalized,
                bounds,
                args=(0.05,),
                maxiter=100,
                popsize=25,
                seed=42,
                tol=1e-8,
                init='latinhypercube',
                polish=False,
                updating='immediate'
            )
            
            de_pts = res_de.x.reshape(n, d)
            norm_min_de = normalized_min_dist(de_pts)
            update_best(res_de.x, norm_min_de)
            
            # Refine the DE result with L-BFGS-B
            if time.time() < deadline - 20:
                res_refine = minimize(
                    objective_normalized,
                    res_de.x,
                    method='L-BFGS-B',
                    args=(0.02,),
                    options={'maxiter': 1000, 'ftol': 1e-14, 'gtol': 1e-10, 'maxls': 50}
                )
                res_refine_pts = res_refine.x.reshape(n, d)
                norm_min_refine = normalized_min_dist(res_refine_pts)
                update_best(res_refine.x, norm_min_refine)
                
        except Exception:
            pass
    
    # Phase 4: Final refinement with Nelder-Mead on exact objective
    if best_x is not None and time.time() < deadline - 10:
        try:
            res_nm = minimize(
                objective_exact,
                best_x,
                method='Nelder-Mead',
                options={'maxiter': 3000, 'xatol': 1e-12, 'fatol': 1e-14, 'adaptive': True}
            )
            nm_pts = res_nm.x.reshape(n, d)
            norm_min_nm = normalized_min_dist(nm_pts)
            update_best(res_nm.x, norm_min_nm)
        except Exception:
            pass
    
    # Fallback
    if best_x is None:
        best_x = generate_hex_grid(1.0, 1.0).ravel()

    return best_x.reshape(n, d)

# EVOLVE-BLOCK-END
