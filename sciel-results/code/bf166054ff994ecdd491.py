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
    1. Use structured initial configurations (Fibonacci sphere, icosahedron-based, etc.).
    2. Use L-BFGS-B optimization on a smooth soft-min/soft-max log-ratio objective.
    3. Multi-start with diverse initial configurations.
    4. SLSQP polish for final refinement.
    5. High-precision final polish with very tight tolerances.
    """
    n = 14
    d = 3
    
    start_time = time.time()
    time_limit = 625.0
    safety_margin = 15.0
    
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
    
    def log_ratio_objective_and_grad(flat_params, beta_min=100.0, beta_max=20.0):
        """
        Objective: maximize log(d_min) - log(d_max)
        Gradient: computed analytically using softmax-weighted gradients.
        """
        pts = flat_params.reshape(n, d)
        
        # Compute all pairwise distances
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dist_sq = np.sum(diff**2, axis=2)
        dist_sq = np.maximum(dist_sq, 1e-15)
        dists_mat = np.sqrt(dist_sq)
        
        # Get upper triangle indices
        i_idx, j_idx = np.triu_indices(n, k=1)
        dists = dists_mat[i_idx, j_idx]
        
        d_min = dists.min()
        d_max = dists.max()
        
        if d_min < 1e-10 or d_max < 1e-10:
            return 0.0, np.zeros_like(flat_params)
        
        # Soft weights with higher precision parameters
        w_min = np.exp(-beta_min * dists)
        w_min /= w_min.sum()
        
        w_max = np.exp(beta_max * dists)
        w_max /= w_max.sum()
        
        # Gradient computation
        grad = np.zeros_like(pts)
        
        for k in range(len(i_idx)):
            i = i_idx[k]
            j = j_idx[k]
            diff_ij = pts[i] - pts[j]
            dist_sq_ij = np.dot(diff_ij, diff_ij)
            if dist_sq_ij < 1e-15:
                continue
            
            # Gradient of log(d_ij) w.r.t. pts[i] = (pts[i] - pts[j]) / d_ij^2
            # Gradient of log(d_ij) w.r.t. pts[j] = (pts[j] - pts[i]) / d_ij^2
            
            # For soft min: we want to INCREASE d_min, so add positive gradient
            grad[i] += w_min[k] * diff_ij / dist_sq_ij
            grad[j] += w_min[k] * (-diff_ij) / dist_sq_ij
            
            # For soft max: we want to DECREASE d_max, so subtract gradient
            grad[i] -= w_max[k] * diff_ij / dist_sq_ij
            grad[j] -= w_max[k] * (-diff_ij) / dist_sq_ij
        
        # Project gradient to tangent space of sphere (for each point)
        for i in range(n):
            grad[i] -= np.dot(grad[i], pts[i]) * pts[i]
        
        # Value: actual log ratio
        val = np.log(d_min) - np.log(d_max)
        
        return -val, -grad.flatten()  # Negate for minimization
    
    def objective_and_grad(flat_params):
        """Same as above but returns (value, gradient) for minimization."""
        return log_ratio_objective_and_grad(flat_params)
    
    def lbfgs_optimize(points, max_iter=500, beta_min=100.0, beta_max=20.0):
        """Use L-BFGS-B to optimize the log-ratio objective."""
        flat = points.flatten()
        
        def obj_with_beta(flat_params):
            return log_ratio_objective_and_grad(flat_params, beta_min, beta_max)
        
        result = minimize(
            obj_with_beta,
            flat,
            method='L-BFGS-B',
            jac=True,
            options={'maxiter': max_iter, 'ftol': 1e-14, 'gtol': 1e-10}
        )
        
        optimized = result.x.reshape(n, d)
        optimized = normalize_sphere(optimized)
        return optimized
    
    def slsqp_polish(points, max_iter=200, beta_min=100.0, beta_max=20.0):
        """Use SLSQP to polish the solution with explicit sphere constraints."""
        flat = points.flatten()
        
        # Define sphere constraints: ||p_i||^2 - 1 = 0 for each point
        def constraints(flat_params):
            pts = flat_params.reshape(n, d)
            return np.sum(pts**2, axis=1) - 1.0
        
        def obj_with_beta(flat_params):
            return log_ratio_objective_and_grad(flat_params, beta_min, beta_max)
        
        result = minimize(
            obj_with_beta,
            flat,
            method='SLSQP',
            jac=True,
            constraints={'type': 'eq', 'fun': constraints},
            options={'maxiter': max_iter, 'ftol': 1e-12}
        )
        
        if result.success:
            optimized = result.x.reshape(n, d)
            optimized = normalize_sphere(optimized)
            return optimized
        else:
            return points
    
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
        # Add two points: try (1,1,1)/norm and (-1,-1,-1)/norm
        p1 = normalize_sphere(np.array([[1.0, 1.0, 1.0]]))[0]
        p2 = normalize_sphere(np.array([[-1.0, -1.0, -1.0]]))[0]
        return np.vstack([icos, [p1, p2]])
    
    def tetrahedral_based():
        """Start from tetrahedral symmetry and add points."""
        tetra = np.array([
            [1, 1, 1], [1, -1, -1], [-1, 1, -1], [-1, -1, 1]
        ], dtype=float)
        tetra = normalize_sphere(tetra)
        pts = list(tetra)
        # Add 10 more points in structured positions
        extra_positions = [
            [1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1],
            [1, 1, 0], [-1, 1, 0], [1, -1, 0], [-1, -1, 0]
        ]
        for pos in extra_positions[:10]:
            p = np.array(pos, dtype=float)
            p = p / np.linalg.norm(p)
            pts.append(p)
        return np.array(pts)
    
    def static_good_initial():
        """A pre-computed good configuration."""
        # Based on known good packings
        pts = []
        # Icosahedron vertices
        phi = (1 + 5**0.5) / 2
        icos = np.array([
            [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
            [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
            [phi, 0, 1], [-phi, 0, 1], [phi, 0, -1], [-phi, 0, -1]
        ], dtype=float)
        icos = normalize_sphere(icos)
        pts.extend(icos.tolist())
        # Add 2 more points
        p1 = np.array([1.0, 1.0, 1.0])
        p1 = p1 / np.linalg.norm(p1)
        p2 = np.array([-1.0, -1.0, -1.0])
        p2 = p2 / np.linalg.norm(p2)
        pts.extend([p1.tolist(), p2.tolist()])
        return np.array(pts)
    
    # Generate initial configurations
    inits = []
    inits.append(fibonacci_sphere(n))
    inits.append(icosahedron_plus_antipodal())
    inits.append(tetrahedral_based())
    inits.append(static_good_initial())
    
    # Many random starts
    for seed in range(30):
        inits.append(random_sphere(n, seed))
    
    best_points = None
    best_score = -1.0
    
    # Phase 1: L-BFGS optimization from all inits
    for idx, init in enumerate(inits):
        if check_deadline():
            break
        
        # L-BFGS optimization with high-precision soft parameters
        optimized = lbfgs_optimize(init, max_iter=400, beta_min=100.0, beta_max=20.0)
        s = score(optimized)
        if s > best_score:
            best_score = s
            best_points = optimized.copy()
        
        # SLSQP polish with high precision
        if not check_deadline():
            polished = slsqp_polish(optimized, max_iter=150, beta_min=100.0, beta_max=20.0)
            s2 = score(polished)
            if s2 > best_score:
                best_score = s2
                best_points = polished.copy()
    
    # Phase 2: Refine the best with more iterations
    if best_points is not None and not check_deadline():
        # Longer L-BFGS with higher precision
        refined = lbfgs_optimize(best_points, max_iter=600, beta_min=150.0, beta_max=25.0)
        s = score(refined)
        if s > best_score:
            best_score = s
            best_points = refined.copy()
        
        # SLSQP polish with high precision
        if not check_deadline():
            polished = slsqp_polish(best_points, max_iter=250, beta_min=150.0, beta_max=25.0)
            s2 = score(polished)
            if s2 > best_score:
                best_score = s2
                best_points = polished.copy()
    
    # Phase 3: Multi-restart with perturbations
    if best_points is not None and not check_deadline():
        rng = np.random.default_rng(42)
        for trial in range(20):
            if check_deadline():
                break
            
            # Perturb the best points
            perturbation_scale = 0.08
            perturbed = best_points + rng.standard_normal((n, d)) * perturbation_scale
            perturbed = normalize_sphere(perturbed)
            
            # L-BFGS optimization with high precision
            optimized = lbfgs_optimize(perturbed, max_iter=250, beta_min=120.0, beta_max=22.0)
            s = score(optimized)
            if s > best_score:
                best_score = s
                best_points = optimized.copy()
            
            # SLSQP polish with high precision
            if not check_deadline():
                polished = slsqp_polish(optimized, max_iter=120, beta_min=120.0, beta_max=22.0)
                s2 = score(polished)
                if s2 > best_score:
                    best_score = s2
                    best_points = polished.copy()
    
    # Phase 4: Final fine-tuning with highest precision
    if best_points is not None and not check_deadline():
        # Very careful L-BFGS with highest precision parameters
        refined = lbfgs_optimize(best_points, max_iter=500, beta_min=200.0, beta_max=30.0)
        s = score(refined)
        if s > best_score:
            best_score = s
            best_points = refined.copy()
        
        # SLSQP final polish with highest precision
        if not check_deadline():
            polished = slsqp_polish(best_points, max_iter=200, beta_min=200.0, beta_max=30.0)
            s2 = score(polished)
            if s2 > best_score:
                best_score = s2
                best_points = polished.copy()
    
    # Fallback
    if best_points is None:
        best_points = fibonacci_sphere(n)
    
    # Validate
    if not np.all(np.isfinite(best_points)) or best_points.shape != (n, d):
        best_points = fibonacci_sphere(n)
    
    return best_points

# EVOLVE-BLOCK-END
