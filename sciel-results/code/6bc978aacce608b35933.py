"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.spatial import ConvexHull
from scipy.optimize import minimize

def construct():
    start_time = time.time()
    deadline = start_time + 118.0  # Safety margin for 120s limit
    
    def fibonacci_sphere(n, offset=0.5):
        i = np.arange(n) + offset
        z = 1.0 - 2.0 * i / n
        phi = np.pi * (1.0 + np.sqrt(5.0)) * i
        r = np.sqrt(1.0 - z * z)
        pts = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
        return pts
    
    def make_initial_config():
        """Create a better initial configuration using icosahedral symmetry + perturbation."""
        # Start with Fibonacci sphere but add some structured perturbation
        pts = fibonacci_sphere(N).copy()
        # Add small random perturbations to break symmetry
        np.random.seed(42)
        noise = np.random.randn(N, 3) * 0.01
        pts += noise
        pts /= np.linalg.norm(pts, axis=1, keepdims=True)
        return pts

    def compute_facet_data(pts):
        """
        Computes convex hull, facet radii, centers, and for each point,
        the center of the worst facet it belongs to.
        Returns (radii, centers, target_centers, worst_radii) or None if hull fails.
        """
        try:
            hull = ConvexHull(pts)
        except Exception:
            return None, None, None, None

        num_facets = len(hull.simplices)
        radii = np.zeros(num_facets)
        centers = np.zeros((num_facets, 3))
        
        for i, simplex in enumerate(hull.simplices):
            v1 = pts[simplex[0]]
            v2 = pts[simplex[1]]
            v3 = pts[simplex[2]]
            
            edge1 = v2 - v1
            edge2 = v3 - v1
            n = np.cross(edge1, edge2)
            norm_n = np.linalg.norm(n)
            if norm_n < 1e-10:
                continue
                
            c = n / norm_n
            if np.dot(c, v1) < 0:
                c = -c
                
            # Compute circumradius of the spherical triangle
            # The circumcenter on the sphere is c, and the angular radius is
            # the angle between c and any vertex
            d = np.linalg.norm(c - v1)
            radius = 2.0 * np.arcsin(np.clip(d / 2.0, 0.0, 1.0))
            
            radii[i] = radius
            centers[i] = c
            
        # Map points to facets
        point_to_facets = [[] for _ in range(N)]
        for i, simplex in enumerate(hull.simplices):
            for idx in simplex:
                point_to_facets[idx].append(i)
                
        target_centers = np.zeros((N, 3))
        worst_radii = np.zeros(N)
        
        for i in range(N):
            if not point_to_facets[i]:
                target_centers[i] = pts[i]
                continue
                
            max_r = -1
            max_f = -1
            for f_idx in point_to_facets[i]:
                if radii[f_idx] > max_r:
                    max_r = radii[f_idx]
                    max_f = f_idx
                    
            worst_radii[i] = max_r
            target_centers[i] = centers[max_f]
            
        return radii, centers, target_centers, worst_radii

    def calc_covering_radius(pts):
        radii, _, _, _ = compute_facet_data(pts)
        if radii is None:
            return np.pi
        return np.max(radii)
    
    def smooth_covering_radius(pts, beta=50.0):
        """
        Smooth approximation of the max facet radius using log-sum-exp.
        This provides a differentiable surrogate for optimization.
        """
        radii, _, _, _ = compute_facet_data(pts)
        if radii is None:
            return np.pi
        # Log-sum-exp for numerical stability
        max_r = np.max(radii)
        if max_r < 1e-15:
            return 0.0
        log_sum_exp = max_r + np.log(np.sum(np.exp(beta * (radii - max_r)))) / beta
        return log_sum_exp

    def energy_function(pts):
        """
        Combined energy: smooth covering radius + regularization for uniformity.
        """
        # Smooth covering radius
        cr = smooth_covering_radius(pts, beta=30.0)
        
        # Pairwise distance variance (promotes uniform distribution)
        # This acts as a regularizer to avoid clustering
        dists = np.sum((pts[:, np.newaxis, :] - pts[np.newaxis, :, :]) ** 2, axis=2)
        mean_dist = np.mean(dists)
        var_dist = np.var(dists)
        uniformity_penalty = var_dist * 0.01
        
        return cr + uniformity_penalty

    # 1. Initialization with multiple starting points
    np.random.seed(42)
    
    best_points = None
    best_radius = np.pi
    
    # Try several initial configurations
    initial_configs = [
        fibonacci_sphere(N).copy(),
        make_initial_config(),
    ]
    
    # Add rotated versions
    for _ in range(3):
        R = np.random.randn(3, 3)
        R, _ = np.linalg.qr(R)
        initial_configs.append(fibonacci_sphere(N) @ R.T)
        initial_configs[-1] /= np.linalg.norm(initial_configs[-1], axis=1, keepdims=True)
    
    for init_pts in initial_configs:
        if time.time() > deadline - 30:
            break
        init_pts = init_pts.copy()
        init_pts /= np.linalg.norm(init_pts, axis=1, keepdims=True)
        init_radius = calc_covering_radius(init_pts)
        if init_radius < best_radius:
            best_radius = init_radius
            best_points = init_pts.copy()
    
    if best_points is None:
        best_points = fibonacci_sphere(N).copy()
        best_radius = calc_covering_radius(best_points)
    
    current_points = best_points.copy()
    
    # 2. Lloyd Iteration with adaptive step size
    base_step = 0.03
    decay_rate = 0.997
    min_step = 0.00005
    max_iter = 8000
    
    current_step = base_step
    stuck_count = 0
    best_radius_lloyd = best_radius
    best_points_lloyd = best_points.copy()
    
    for iter in range(max_iter):
        if time.time() > deadline - 20:
            break
            
        radii, centers, target_centers, worst_radii = compute_facet_data(current_points)
        
        if radii is None:
            break
            
        current_radius = np.max(radii)
        
        if current_radius < best_radius_lloyd - 1e-10:
            best_radius_lloyd = current_radius
            best_points_lloyd = current_points.copy()
            stuck_count = 0
        else:
            stuck_count += 1
            
        # Calculate weights based on worst radius
        max_wr = np.max(worst_radii)
        if max_wr > 1e-10:
            weights = worst_radii / max_wr
        else:
            weights = np.ones(N)
            
        # Update points
        new_points = current_points.copy()
        for i in range(N):
            dir_vec = target_centers[i] - current_points[i]
            norm_dir = np.linalg.norm(dir_vec)
            if norm_dir > 1e-10:
                dir_vec /= norm_dir
                move = current_step * weights[i]
                new_points[i] += move * dir_vec
                
        # Normalize
        new_points /= np.linalg.norm(new_points, axis=1, keepdims=True)
        
        current_points = new_points
        
        # Decay step size
        current_step *= decay_rate
        
        # If stuck or step too small, perturb and reset step
        if stuck_count > 80 or current_step < min_step:
            noise = np.random.randn(N, 3) * 0.02
            current_points = current_points + noise
            current_points /= np.linalg.norm(current_points, axis=1, keepdims=True)
            current_step = base_step * 0.7
            stuck_count = 0

        # Occasional restart from best found
        if stuck_count > 300:
            current_points = best_points_lloyd.copy()
            noise = np.random.randn(N, 3) * 0.01
            current_points = current_points + noise
            current_points /= np.linalg.norm(current_points, axis=1, keepdims=True)
            current_step = base_step
            stuck_count = 0
    
    if best_radius_lloyd < best_radius:
        best_radius = best_radius_lloyd
        best_points = best_points_lloyd.copy()
    
    # 3. Smooth energy minimization using scipy
    if time.time() < deadline - 10:
        current_best = best_points.copy()
        
        def flatten(pts):
            return pts.flatten()
        
        def unflatten(flat):
            pts = flat.reshape(N, 3)
            pts /= np.linalg.norm(pts, axis=1, keepdims=True)
            return pts
        
        def objective(flat):
            pts = unflatten(flat)
            return energy_function(pts)
        
        # Try L-BFGS-B on the smooth energy
        x0 = flatten(current_best)
        try:
            result = minimize(objective, x0, method='L-BFGS-B', 
                             options={'maxiter': 50, 'ftol': 1e-10, 'gtol': 1e-8})
            if result.fun < energy_function(current_best) - 1e-8:
                candidate = unflatten(result.x)
                cr = calc_covering_radius(candidate)
                if cr < best_radius:
                    best_radius = cr
                    best_points = candidate
        except Exception:
            pass
    
    # 4. Final Polish: Local search on worst facets
    if time.time() < deadline - 3:
        current_best = best_points.copy()
        current_score = best_radius
        step = 0.005
        
        polish_iter = 0
        while time.time() < deadline - 3 and polish_iter < 50:
            polish_iter += 1
            improved = False
            indices = np.random.permutation(N)
            for j in indices:
                if time.time() > deadline - 3:
                    break
                # Try multiple random directions
                best_local = current_score
                best_local_p = current_best[j].copy()
                for _trial in range(20):
                    rand_vec = np.random.randn(3)
                    rand_vec /= np.linalg.norm(rand_vec)
                    
                    new_p = current_best[j] + step * rand_vec
                    new_p /= np.linalg.norm(new_p)
                    
                    current_best[j] = new_p
                    new_score = calc_covering_radius(current_best)
                    
                    if new_score < best_local - 1e-10:
                        best_local = new_score
                        best_local_p = new_p.copy()
                    
                    current_best[j] = best_local_p
                
                if best_local < current_score - 1e-10:
                    current_score = best_local
                    improved = True
                current_best[j] = best_local_p
                    
            if not improved:
                step *= 0.7
                if step < 1e-6:
                    break
                    
        if current_score < best_radius:
            best_points = current_best
            best_radius = current_score

    # Final normalization
    best_points = best_points / np.linalg.norm(best_points, axis=1, keepdims=True)
    
    return best_points
# EVOLVE-BLOCK-END
