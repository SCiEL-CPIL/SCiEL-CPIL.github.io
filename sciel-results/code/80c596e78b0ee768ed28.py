"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 130

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial import ConvexHull

def _normalize(pts):
    """Normalize points to unit sphere."""
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-15)
    return pts / norms

def _covering_radius_exact(pts):
    """
    Compute the exact covering radius in radians using convex hull.
    The covering radius is the maximum circumradius of the facets of the convex hull.
    """
    pts = np.array(pts, dtype=np.float64)
    pts = _normalize(pts)
    
    n_pts = len(pts)
    if n_pts < 4:
        return np.pi
    
    # Check for coincident points
    for i in range(n_pts):
        for j in range(i + 1, n_pts):
            if np.linalg.norm(pts[i] - pts[j]) < 1e-12:
                return np.pi
    
    # Compute convex hull
    try:
        hull = ConvexHull(pts)
    except Exception:
        return np.pi
    
    # Check if origin is inside the convex hull (points not in a closed hemisphere)
    # If all facet normals point in roughly the same direction, points are in a hemisphere
    facet_normals = []
    for simplex in hull.simplices:
        if len(simplex) == 3:
            a, b, c = pts[simplex[0]], pts[simplex[1]], pts[simplex[2]]
            n = np.cross(b - a, c - a)
            n_norm = np.linalg.norm(n)
            if n_norm > 1e-15:
                n = n / n_norm
                centroid = (a + b + c) / 3.0
                if np.dot(n, centroid) < 0:
                    n = -n
                facet_normals.append(n)
    
    if len(facet_normals) == 0:
        return np.pi
    
    facet_normals = np.array(facet_normals)
    mean_n = np.mean(facet_normals, axis=0)
    mean_norm = np.linalg.norm(mean_n)
    if mean_norm > 0.3:
        # Likely in a hemisphere - return large value
        return np.pi
    
    # Compute covering radius from facets
    max_circumradius = 0.0
    
    for simplex in hull.simplices:
        facet_pts = pts[simplex]
        k = len(simplex)
        
        if k == 3:
            a, b, c = facet_pts
            
            # Circumcenter on the sphere
            n = np.cross(b - a, c - a)
            n_norm = np.linalg.norm(n)
            if n_norm < 1e-15:
                continue
            x = n / n_norm
            
            centroid = (a + b + c) / 3.0
            if np.dot(x, centroid) < 0:
                x = -x
            
            dot_val = np.dot(x, a)
            dot_val = np.clip(dot_val, -1.0, 1.0)
            angular_r = np.arccos(dot_val)
            max_circumradius = max(max_circumradius, angular_r)
    
    return max_circumradius

def _covering_radius_grid(pts, n_grid=5000):
    """Compute covering radius using a grid-based approach."""
    pts = np.array(pts, dtype=np.float64)
    pts = _normalize(pts)
    
    # Fibonacci sphere grid
    i = np.arange(n_grid) + 0.5
    z = 1.0 - 2.0 * i / n_grid
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.clip(1.0 - z * z, 0, None))
    grid = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    # Compute max dot product for each grid point
    dots = grid @ pts.T  # shape (n_grid, N)
    min_dots = np.max(dots, axis=1)
    max_angular_dist = np.arccos(np.clip(min_dots, -1.0, 1.0))
    
    return np.max(max_angular_dist)

def _lse_objective_and_grad(x, beta=50.0, n_grid=3000):
    """
    Log-sum-exp surrogate objective and its gradient for sphere covering.
    x is a flat array of 3*N coordinates.
    """
    pts = x.reshape(-1, 3)
    pts = _normalize(pts)
    
    # Fibonacci sphere grid
    i = np.arange(n_grid) + 0.5
    z = 1.0 - 2.0 * i / n_grid
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.clip(1.0 - z * z, 0, None))
    grid = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    # Compute dots: (n_grid, N)
    dots = grid @ pts.T
    
    # For each grid point, find the max dot (nearest point)
    # We want to minimize the max angular distance = maximize the min dot
    # Use log-sum-exp of dots to get a smooth approximation of max
    
    # max_dots[i] = max_j dots[i,j]
    # We want to minimize -min_dots = maximize min_dots
    # Actually, we want to minimize the covering radius = maximize the minimum dot product
    
    # Let's use: f = -log(sum(exp(beta * dots))) which approximates -max(dots)
    # Minimizing f = maximizing max(dots) for each grid point
    
    # For each grid point i, max_dot_i = max_j dots[i,j]
    # We want to minimize max_i (-max_dot_i) = minimize max_i (1 - max_dot_i) approximately
    
    # Better: use the covering radius directly via grid
    # min_dots[i] = max_j dots[i,j]
    # covering_radius = max_i arccos(min_dots[i])
    
    # For smooth optimization, use:
    # f = log(sum_i exp(beta * (1 - min_dots[i])))
    # This approximates max_i (1 - min_dots[i])
    
    max_dots = np.max(dots, axis=1)  # (n_grid,)
    one_minus = 1.0 - max_dots  # (n_grid,)
    
    # Log-sum-exp
    max_val = np.max(one_minus)
    if max_val < -1e10:
        return 0.0, np.zeros_like(x)
    
    exp_terms = np.exp(beta * (one_minus - max_val))
    log_sum_exp = np.log(np.sum(exp_terms)) / beta + max_val
    
    # Gradient: d/dx log_sum_exp
    # d/dx_j log(sum_i exp(beta*(1-max_dots_i)))
    # = beta * sum_i exp(beta*(1-max_dots_i)) * d(1-max_dots_i)/dx_j / sum_i exp(beta*(1-max_dots_i))
    # = beta * sum_i p_i * d(-max_dots_i)/dx_j
    # where p_i = exp(beta*(1-max_dots_i)) / sum(exp(...))
    
    # d(max_dots_i)/dx_j: max_dots_i = max_k dots[i,k]
    # Let j* be the argmax for grid point i
    # d(dots[i,k])/dx = grid[i] for component k
    # So d(max_dots_i)/dx = grid[i] for the nearest point index
    
    # For each grid point, find the nearest point index
    nearest_idx = np.argmax(dots, axis=1)  # (n_grid,)
    
    # Gradient with respect to pts
    grad_pts = np.zeros_like(pts)  # (N, 3)
    for i in range(n_grid):
        j = nearest_idx[i]
        grad_pts[j] += beta * exp_terms[i] * grid[i]
    
    grad_pts /= np.sum(exp_terms)
    
    # Chain rule: x -> pts (normalization)
    # pts = x / ||x||
    # d pts / d x = (I - pts*pts^T) / ||x||
    
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-15)
    
    # grad_x = (I - pts*pts^T) @ grad_pts / norms
    # For each point j:
    # grad_x[j] = (grad_pts[j] - pts[j] * dot(grad_pts[j], pts[j])) / norms[j]
    
    grad_x = np.zeros_like(x)
    for j in range(N):
        proj = np.dot(grad_pts[j], pts[j])
        grad_x[3*j:3*j+3] = (grad_pts[j] - pts[j] * proj) / norms[j]
    
    return log_sum_exp, grad_x

def construct():
    start_time = time.time()
    deadline = start_time + 135.0  # 15s safety margin before 150s limit
    
    def _elapsed():
        return time.time() - start_time
    
    def _remaining():
        return deadline - time.time()
    
    # Initialize with Fibonacci spiral
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.clip(1.0 - z * z, 0, None))
    pts = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    best_pts = pts.copy()
    best_radius = _covering_radius_grid(best_pts, 10000)
    
    # Optimize using L-BFGS-B with the log-sum-exp surrogate
    x0 = pts.flatten()
    
    # Multiple restarts with different beta values
    beta_values = [30.0, 50.0, 80.0]
    
    for beta in beta_values:
        if _remaining() < 20.0:
            break
        
        # Try from the current best and from a random start
        for start_type in ['best', 'random']:
            if _remaining() < 15.0:
                break
            
            if start_type == 'best':
                x_init = best_pts.flatten()
            else:
                x_init = np.random.randn(N * 3)
                x_init = _normalize(x_init.reshape(-1, 3)).flatten()
            
            # Use L-BFGS-B
            result = minimize(
                _lse_objective_and_grad,
                x_init,
                args=(beta, 5000),
                jac=True,
                method='L-BFGS-B',
                options={'maxiter': 50, 'ftol': 1e-12, 'gtol': 1e-8, 'maxls': 50}
            )
            
            if _remaining() < 10.0:
                break
            
            # Evaluate the result
            x_opt = result.x
            pts_opt = _normalize(x_opt.reshape(-1, 3))
            
            # Check for validity
            valid = True
            for i in range(N):
                for j in range(i + 1, N):
                    if np.linalg.norm(pts_opt[i] - pts_opt[j]) < 1e-10:
                        valid = False
                        break
                if not valid:
                    break
            
            if valid:
                radius_grid = _covering_radius_grid(pts_opt, 10000)
                if radius_grid < best_radius:
                    best_radius = radius_grid
                    best_pts = pts_opt.copy()
    
    # Fine-tuning phase: local search with small perturbations
    if _remaining() > 15.0:
        p_current = best_pts.copy()
        
        for trial in range(50):
            if _remaining() < 10.0:
                break
            
            # Perturb a random subset of points
            n_perturb = np.random.randint(1, 4)
            indices = np.random.choice(N, n_perturb, replace=False)
            
            p_trial = p_current.copy()
            for idx in indices:
                perturbation = np.random.randn(3)
                perturbation /= np.linalg.norm(perturbation)
                step_size = 0.005
                p_trial[idx] += step_size * perturbation
                p_trial[idx] /= np.linalg.norm(p_trial[idx])
            
            trial_r = _covering_radius_grid(p_trial, 10000)
            if trial_r < best_radius:
                best_radius = trial_r
                best_pts = p_trial.copy()
                p_current = p_trial.copy()
    
    # Final evaluation with a denser grid
    final_radius = _covering_radius_grid(best_pts, 20000)
    if final_radius < best_radius:
        best_radius = final_radius
    
    return best_pts
# EVOLVE-BLOCK-END
