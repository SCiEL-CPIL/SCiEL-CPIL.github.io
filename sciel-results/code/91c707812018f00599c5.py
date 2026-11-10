# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.spatial.distance import pdist
from scipy.optimize import minimize


def _score(points):
    """Compute (d_min/d_max)^2 for a set of points."""
    if len(points) < 2:
        return 0.0
    dists = pdist(points)
    d_min = dists.min()
    d_max = dists.max()
    if d_max == 0 or not np.isfinite(d_min) or not np.isfinite(d_max):
        return 0.0
    ratio = d_min / d_max
    return ratio * ratio


def min_max_dist_dim3_14() -> np.ndarray:
    """
    Creates 14 points in 3 dimensions to maximize (d_min/d_max)^2.
    
    Strategy:
    1. Use a static high-quality initial guess (pre-computed good configuration).
    2. Use L-BFGS-B optimization on the log-ratio objective with sphere constraint.
    3. Multi-start with diverse initial configurations.
    4. SLSQP polish for final refinement.
    """
    n = 14
    d = 3
    
    start_time = time.time()
    time_limit = 630.0
    safety_margin = 10.0
    
    def elapsed():
        return time.time() - start_time
    
    def check_deadline():
        return elapsed() > (time_limit - safety_margin)
    
    def normalize_sphere(points):
        norms = np.linalg.norm(points, axis=1, keepdims=True)
        norms[norms < 1e-12] = 1.0
        return points / norms
    
    def score(points):
        dists = pdist(points)
        if len(dists) == 0:
            return 0.0
        d_min = dists.min()
        d_max = dists.max()
        if d_max == 0 or not np.isfinite(d_min) or not np.isfinite(d_max):
            return 0.0
        ratio = d_min / d_max
        return ratio * ratio
    
    def log_ratio_objective(flat_pts):
        """Negative log ratio for minimization. flat_pts is 42-dim vector."""
        pts = flat_pts.reshape(n, d)
        # Project to sphere
        norms = np.linalg.norm(pts, axis=1, keepdims=True)
        norms[norms < 1e-12] = 1.0
        pts = pts / norms
        
        dists = pdist(pts)
        d_min = dists.min()
        d_max = dists.max()
        
        if d_min < 1e-10 or d_max < 1e-10:
            return 1e10
        
        # We want to maximize log(d_min) - log(d_max)
        # So minimize -(log(d_min) - log(d_max)) = log(d_max) - log(d_min)
        val = np.log(d_max) - np.log(d_min)
        return val
    
    def log_ratio_gradient(flat_pts):
        """Gradient of the log ratio objective."""
        pts = flat_pts.reshape(n, d)
        norms = np.linalg.norm(pts, axis=1, keepdims=True)
        norms[norms < 1e-12] = 1.0
        pts = pts / norms
        
        # Compute pairwise distances
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dist_sq = np.sum(diff**2, axis=2)
        np.fill_diagonal(dist_sq, 1.0)
        dists_mat = np.sqrt(dist_sq)
        
        # Get min and max distances
        i_idx, j_idx = np.triu_indices(n, k=1)
        dists = dists_mat[i_idx, j_idx]
        
        d_min = dists.min()
        d_max = dists.max()
        
        if d_min < 1e-10 or d_max < 1e-10:
            return np.zeros_like(flat_pts)
        
        # Gradient of log(d_max) - log(d_min)
        # d/dp_i log(||p_i - p_j||) = (p_i - p_j) / ||p_i - p_j||^2
        # for the max distance pair, we add this
        # for the min distance pair, we subtract this
        
        grad = np.zeros((n, d))
        
        # Find all pairs achieving (near) min and max
        min_pairs = np.where(np.abs(dists - d_min) < 1e-8)[0]
        max_pairs = np.where(np.abs(dists - d_max) < 1e-8)[0]
        
        for k in min_pairs:
            i, j = i_idx[k], j_idx[k]
            diff_ij = pts[i] - pts[j]
            d_ij = dists_mat[i, j]
            if d_ij > 1e-10:
                grad[i] -= diff_ij / (d_ij ** 2)
                grad[j] += diff_ij / (d_ij ** 2)
        
        for k in max_pairs:
            i, j = i_idx[k], j_idx[k]
            diff_ij = pts[i] - pts[j]
            d_ij = dists_mat[i, j]
            if d_ij > 1e-10:
                grad[i] += diff_ij / (d_ij ** 2)
                grad[j] -= diff_ij / (d_ij ** 2)
        
        # Project to tangent space
        for i in range(n):
            grad[i] -= np.dot(grad[i], pts[i]) * pts[i]
        
        return grad.flatten()
    
    def optimize_lbfgsb(init_pts, max_iter=2000):
        """Optimize using L-BFGS-B."""
        flat_init = init_pts.flatten()
        
        result = minimize(
            log_ratio_objective,
            flat_init,
            jac=log_ratio_gradient,
            method='L-BFGS-B',
            options={'maxiter': max_iter, 'ftol': 1e-12, 'gtol': 1e-10}
        )
        
        if result.success or result.fun < 1e9:
            opt_pts = result.x.reshape(n, d)
            opt_pts = normalize_sphere(opt_pts)
            return opt_pts
        return init_pts
    
    def optimize_slsqp(init_pts, max_iter=500):
        """Optimize using SLSQP with sphere constraints."""
        flat_init = init_pts.flatten()
        
        # Sphere constraints: ||p_i||^2 - 1 = 0 for each i
        def constraint(i):
            def con(x):
                pts = x.reshape(n, d)
                return np.dot(pts[i], pts[i]) - 1.0
            return con
        
        constraints = [{'type': 'eq', 'fun': constraint(i)} for i in range(n)]
        
        result = minimize(
            log_ratio_objective,
            flat_init,
            jac=log_ratio_gradient,
            method='SLSQP',
            constraints=constraints,
            options={'maxiter': max_iter, 'ftol': 1e-12}
        )
        
        if result.success or result.fun < 1e9:
            opt_pts = result.x.reshape(n, d)
            opt_pts = normalize_sphere(opt_pts)
            return opt_pts
        return init_pts
    
    def fibonacci_sphere(n_points):
        points = []
        for i in range(n_points):
            phi = np.arccos(1 - 2 * (i + 0.5) / n_points)
            theta = np.pi * (1 + 5**0.5) * i
            x = np.sin(phi) * np.cos(theta)
            y = np.sin(phi) * np.sin(theta)
            z = np.cos(phi)
            points.append([x, y, z])
        return np.array(points)
    
    def random_sphere(n_points, seed):
        rng = np.random.default_rng(seed)
        points = rng.standard_normal((n_points, 3))
        return normalize_sphere(points)
    
    def icosahedron_plus_antipodal():
        """Icosahedron (12 vertices) + 2 antipodal points."""
        phi = (1 + 5**0.5) / 2
        icos = np.array([
            [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
            [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
            [phi, 0, 1], [-phi, 0, 1], [phi, 0, -1], [-phi, 0, -1]
        ], dtype=float)
        icos = normalize_sphere(icos)
        p1 = normalize_sphere(np.array([[1.0, 1.0, 1.0]]))[0]
        p2 = normalize_sphere(np.array([[-1.0, -1.0, -1.0]]))[0]
        return np.vstack([icos, [p1, p2]])
    
    # Static high-quality initial guess (pre-computed)
    def static_init():
        """A good starting configuration."""
        # Use a combination of icosahedron vertices and tetrahedral points
        phi = (1 + 5**0.5) / 2
        pts = []
        
        # Icosahedron vertices
        icos = np.array([
            [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
            [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
            [phi, 0, 1], [-phi, 0, 1], [phi, 0, -1], [-phi, 0, -1]
        ], dtype=float)
        icos = normalize_sphere(icos)
        pts.extend(icos.tolist())
        
        # Two additional points
        p1 = normalize_sphere(np.array([[1.0, 1.0, 1.0]]))[0]
        p2 = normalize_sphere(np.array([[-1.0, -1.0, -1.0]]))[0]
        pts.extend([p1.tolist(), p2.tolist()])
        
        return np.array(pts)
    
    # Generate initial configurations
    inits = []
    inits.append(static_init())
    inits.append(fibonacci_sphere(n))
    inits.append(icosahedron_plus_antipodal())
    
    # Random starts
    for seed in range(30):
        inits.append(random_sphere(n, seed))
    
    best_points = None
    best_score = -1.0
    
    # Phase 1: L-BFGS-B optimization from all inits
    for idx, init in enumerate(inits):
        if check_deadline():
            break
        
        # Quick L-BFGS-B
        opt_pts = optimize_lbfgsb(init, max_iter=1000)
        s = score(opt_pts)
        if s > best_score:
            best_score = s
            best_points = opt_pts.copy()
    
    # Phase 2: SLSQP polish on best
    if best_points is not None and not check_deadline():
        polished = optimize_slsqp(best_points, max_iter=300)
        s = score(polished)
        if s > best_score:
            best_score = s
            best_points = polished.copy()
    
    # Phase 3: Multi-restart with perturbations from best
    if best_points is not None and not check_deadline():
        rng = np.random.default_rng(42)
        for trial in range(15):
            if check_deadline():
                break
            
            perturbed = best_points + rng.standard_normal((n, d)) * 0.05
            perturbed = normalize_sphere(perturbed)
            
            opt_pts = optimize_lbfgsb(perturbed, max_iter=500)
            s = score(opt_pts)
            if s > best_score:
                best_score = s
                best_points = opt_pts.copy()
            
            # SLSQP polish
            if not check_deadline():
                polished = optimize_slsqp(opt_pts, max_iter=200)
                s2 = score(polished)
                if s2 > best_score:
                    best_score = s2
                    best_points = polished.copy()
    
    # Phase 4: Final SLSQP polish
    if best_points is not None and not check_deadline():
        final = optimize_slsqp(best_points, max_iter=500)
        s = score(final)
        if s > best_score:
            best_score = s
            best_points = final.copy()
    
    # Fallback
    if best_points is None:
        best_points = fibonacci_sphere(n)
    
    # Validate
    if not np.all(np.isfinite(best_points)) or best_points.shape != (n, d):
        best_points = fibonacci_sphere(n)
    
    return best_points

# EVOLVE-BLOCK-END
