# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial.distance import pdist
from scipy.optimize import minimize, differential_evolution
import time

def min_max_dist_dim3_14() -> np.ndarray:
    """
    Creates 14 points in 3 dimensions to maximize (d_min / d_max)^2.
    """
    n = 14
    d = 3
    start_time = time.time()
    time_limit = 600  # seconds

    def get_score(pts):
        dists = pdist(pts, metric='euclidean')
        if len(dists) == 0:
            return 0.0
        d_min = np.min(dists)
        d_max = np.max(dists)
        if d_max == 0:
            return 0.0
        return (d_min / d_max) ** 2

    def logsumexp(arr, beta):
        """Stable log-sum-exp scaled by beta."""
        scaled = arr / beta
        m = np.max(scaled)
        return beta * (m + np.log(np.sum(np.exp(scaled - m))))

    def objective_softmin(x, beta):
        """
        Maximize d_min (soft min) subject to d_max (soft max) <= 1.
        We minimize: -(soft_min) + penalty * max(0, soft_max - 1)^2
        """
        p = x.reshape((n, d))
        if not np.all(np.isfinite(p)):
            return 1e10
        
        # Compute all pairwise distances
        # Using pdist for efficiency
        dists = pdist(p, metric='euclidean')
        
        soft_min = logsumexp(-dists, beta)  # approximates min(dists)
        soft_max = logsumexp(dists, beta)   # approximates max(dists)
        
        # We want to maximize soft_min while keeping soft_max close to 1
        # Since the problem is scale-invariant, we can normalize
        # Objective: minimize -(soft_min) + lambda * (soft_max - 1)^2
        # Actually, let's use a different approach:
        # Maximize soft_min / soft_max
        # Which is equivalent to minimizing soft_max / soft_min
        
        if soft_min <= 1e-10:
            return 1e10
        
        ratio = soft_max / soft_min
        return ratio ** 2

    def objective_normalized(x, beta):
        """
        Alternative: normalize by soft_max, maximize soft_min.
        """
        p = x.reshape((n, d))
        if not np.all(np.isfinite(p)):
            return 1e10
        
        dists = pdist(p, metric='euclidean')
        
        soft_min = logsumexp(-dists, beta)
        soft_max = logsumexp(dists, beta)
        
        if soft_max <= 1e-10:
            return 1e10
        
        # Normalize: all distances divided by soft_max
        # Then maximize the min of normalized distances
        # This is equivalent to maximizing soft_min / soft_max
        normalized_min = soft_min / soft_max
        # We want to maximize this, so minimize negative
        return -normalized_min

    def objective_tight_band(x, beta):
        """
        Penalize both:
        1. Distances too small (below target)
        2. Distances too large (above target)
        Target is set dynamically or we just equalize.
        """
        p = x.reshape((n, d))
        if not np.all(np.isfinite(p)):
            return 1e10
        
        dists = pdist(p, metric='euclidean')
        
        # Soft min and max
        soft_min = logsumexp(-dists, beta)
        soft_max = logsumexp(dists, beta)
        
        if soft_min <= 1e-10 or soft_max <= 1e-10:
            return 1e10
        
        # We want soft_min close to soft_max
        # Minimize (soft_max - soft_min)^2 + epsilon * soft_max^2
        # Actually, the ratio is what matters.
        # Let's minimize (soft_max/soft_min)^2
        ratio = soft_max / soft_min
        return ratio ** 2

    # --- Initialization Strategies ---

    def fibonacci_sphere(n_points, seed=0):
        points = np.zeros((n_points, 3))
        golden_angle = np.pi * (3.0 - np.sqrt(5.0))
        rng = np.random.RandomState(seed)
        for i in range(n_points):
            z = 1.0 - (i / (n_points - 1)) * 2.0
            radius = np.sqrt(1.0 - z**2)
            theta = golden_angle * i + rng.rand() * 0.1 # slight jitter
            points[i] = [radius * np.cos(theta), radius * np.sin(theta), z]
        return points

    def icosahedron_poles(pole_offset=1.0):
        phi = (1 + np.sqrt(5)) / 2
        ico_verts = np.array([
            [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
            [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
            [phi, 0, 1], [phi, 0, -1], [-phi, 0, 1], [-phi, 0, -1]
        ])
        norms_ico = np.linalg.norm(ico_verts, axis=1)
        ico_verts_norm = ico_verts / norms_ico[:, np.newaxis]
        poles = np.array([[0, 0, pole_offset], [0, 0, -pole_offset]])
        pts = np.vstack([ico_verts_norm, poles])
        return pts

    # Generate a diverse set of initial candidates
    initial_configs = []
    
    # 1. Icosahedron + Poles (various offsets)
    for offset in [0.5, 0.7, 0.9, 1.0, 1.2, 1.5]:
        initial_configs.append(icosahedron_poles(offset))
        
    # 2. Fibonacci Spheres (various seeds)
    for seed in range(10):
        initial_configs.append(fibonacci_sphere(n, seed=seed))
        
    # 3. Random Points on Sphere
    for seed in range(5):
        rng = np.random.RandomState(seed)
        points = rng.randn(n, 3)
        norms = np.linalg.norm(points, axis=1)
        points /= norms[:, np.newaxis]
        initial_configs.append(points)

    # Normalize all initial configs to unit sphere (center at origin)
    # Note: The optimal solution for max-min ratio on a bounded domain is often on the boundary,
    # but since the ratio is scale-invariant, we can fix the scale.
    # However, the points don't have to be on the unit sphere, but for 3D, it's usually beneficial
    # to be well-spread. Let's keep them as is but centered.
    
    best_points = None
    best_score = -1e10

    # Evaluate initial configs
    for cfg in initial_configs:
        cfg = cfg - np.mean(cfg, axis=0) # Center
        if not np.all(np.isfinite(cfg)):
            continue
        score = get_score(cfg)
        if score > best_score:
            best_score = score
            best_points = cfg

    if best_points is None:
        # Fallback
        best_points = icosahedron_poles(1.0)
        best_score = get_score(best_points)

    # --- Optimization Loop ---

    # Phase 1: Global Search with Differential Evolution
    if time.time() - start_time < time_limit - 120:
        try:
            beta_de = 0.2
            bounds = [(-2, 2)] * (n * d)
            
            # Use a wrapper that handles the objective
            def de_obj(x):
                return objective_tight_band(x, beta_de)
            
            res_de = differential_evolution(
                de_obj, 
                bounds=bounds, 
                maxiter=50, 
                popsize=15, 
                tol=1e-5,
                seed=42,
                init='latinhypercube'
            )
            
            if res_de.fun < 1e9:
                cand = res_de.x.reshape((n, d))
                cand = cand - np.mean(cand, axis=0)
                score = get_score(cand)
                if score > best_score:
                    best_score = score
                    best_points = cand
        except Exception:
            pass

    # Phase 2: Local Refinement with L-BFGS-B on Soft Objective
    # We'll run this from multiple starting points
    
    beta_schedule = [0.1, 0.05, 0.02, 0.01, 0.005, 0.002]
    
    # Starting points: Best found so far + a few random perturbations
    start_pts_list = [best_points.copy()]
    
    # Add perturbed versions
    rng = np.random.RandomState(123)
    for scale in [0.1, 0.3, 0.5]:
        perturb = best_points + rng.randn(n, d) * scale
        start_pts_list.append(perturb)
    
    # Also add a few of the best initial configs
    # Sort initial configs by score
    scored_inits = []
    for cfg in initial_configs:
        cfg_c = cfg - np.mean(cfg, axis=0)
        s = get_score(cfg_c)
        scored_inits.append((s, cfg_c))
    scored_inits.sort(reverse=True)
    
    for s, cfg in scored_inits[:3]:
        start_pts_list.append(cfg)

    for beta in beta_schedule:
        if time.time() - start_time > time_limit - 30:
            break
            
        for init_pts in start_pts_list:
            if time.time() - start_time > time_limit - 20:
                break
                
            try:
                def local_obj(x):
                    return objective_tight_band(x, beta)
                
                res = minimize(
                    local_obj, 
                    init_pts.ravel(), 
                    method='L-BFGS-B', 
                    options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-10}
                )
                
                if res.fun < 1e9:
                    opt_pts = res.x.reshape((n, d))
                    opt_pts = opt_pts - np.mean(opt_pts, axis=0)
                    score = get_score(opt_pts)
                    
                    if score > best_score:
                        best_score = score
                        best_points = opt_pts
                        # Update start points for next iteration
                        start_pts_list = [opt_pts.copy()]
                        # Add some perturbations of the new best
                        for scale in [0.05, 0.1]:
                            p = opt_pts + rng.randn(n, d) * scale
                            start_pts_list.append(p)
            except Exception:
                pass

    # Phase 3: Exact Objective Polish (Powell or Nelder-Mead might be better for non-smooth)
    # Since the exact objective (min/max ratio) is non-smooth, gradient-based methods struggle.
    # However, if we are close, L-BFGS-B on the soft objective with very small beta works well.
    # Alternatively, use Powell which is derivative-free.
    
    if time.time() - start_time < time_limit - 10:
        # Try a few steps of Nelder-Mead or Powell on the exact objective
        # But exact objective evaluation is expensive (pdist) and non-smooth.
        # Let's stick with the soft objective with extremely small beta for final polish.
        
        final_beta = 0.001
        
        def final_obj(x):
            return objective_tight_band(x, final_beta)
        
        try:
            res_final = minimize(
                final_obj, 
                best_points.ravel(), 
                method='L-BFGS-B', 
                options={'maxiter': 2000, 'ftol': 1e-15, 'gtol': 1e-13}
            )
            
            if res_final.fun < 1e9:
                final_pts = res_final.x.reshape((n, d))
                final_pts = final_pts - np.mean(final_pts, axis=0)
                final_score = get_score(final_pts)
                
                if final_score > best_score:
                    best_score = final_score
                    best_points = final_pts
        except Exception:
            pass

    # Final Validity Check
    if not np.all(np.isfinite(best_points)):
        # Return a valid fallback
        return icosahedron_poles(1.0)

    return best_points

# EVOLVE-BLOCK-END
