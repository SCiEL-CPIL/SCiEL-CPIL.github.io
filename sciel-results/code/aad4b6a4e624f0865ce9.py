"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 300

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial.distance import pdist

def _compute_radius(points):
    """Compute the feasible radius for a set of points."""
    if points is None or len(points) == 0:
        return 0.0
    # Distance to boundary
    dist_boundary = 1.0 - np.linalg.norm(points, axis=1)
    min_boundary = np.min(dist_boundary)
    
    # Pairwise distances
    d = pdist(points)
    min_dist = np.min(d)
    min_pair = min_dist / 2.0
    
    return min(min_boundary, min_pair)

def _generate_hexagonal_lattice():
    """
    Generate a hexagonal lattice packing.
    This provides a good initial guess with high symmetry.
    """
    best_r = 0.0
    best_points = None
    
    # Spacing a relates to radius r by r = a/2 in ideal hex packing.
    # Theoretical max r ~ sqrt(0.9069 / 300) ~ 0.055
    # So a ~ 0.11
    for a in np.linspace(0.108, 0.112, 10):
        h = a * np.sqrt(3) / 2
        
        points = []
        # Generate enough points to fill the disk
        for j in range(-20, 21):
            for i in range(-20, 21):
                x = (i + 0.5 * (j % 2)) * a
                y = j * h
                # Quick filter
                if x*x + y*y < (1.0 - a/2)**2:
                    points.append([x, y])
        
        if len(points) < N:
            continue
            
        points = np.array(points)
        
        # Exact filter
        norms = np.linalg.norm(points, axis=1)
        mask = (1.0 - norms) >= (a/2) - 1e-9
        points = points[mask]
        
        if len(points) < N:
            continue
        
        # Select N points closest to center
        dists = np.linalg.norm(points, axis=1)
        idx = np.argsort(dists)[:N]
        selected = points[idx]
        
        r = _compute_radius(selected)
        if r > best_r:
            best_r = r
            best_points = selected.copy()
            
    return best_points, best_r

def _soft_min_objective(points, beta=50.0):
    """
    Differentiable approximation of min(d_ij/2, 1-d_i).
    We maximize this, so we minimize the negative.
    """
    pts = points.reshape(N, 2)
    
    # 1. Boundary constraints: 1 - ||p_i||
    dists = np.linalg.norm(pts, axis=1)
    boundary_terms = 1.0 - dists
    
    # 2. Pairwise constraints: ||p_i - p_j|| / 2
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dist_matrix = np.linalg.norm(diff, axis=2)
    np.fill_diagonal(dist_matrix, np.inf)
    pair_terms = dist_matrix / 2.0
    
    # Combine all terms into a single vector
    all_terms = np.concatenate([boundary_terms, pair_terms.flatten()])
    
    # Scale for numerical stability
    min_term = np.min(all_terms)
    exp_terms = np.exp(-beta * (all_terms - min_term))
    
    log_sum = np.log(np.sum(exp_terms))
    soft_min_val = min_term - np.log(np.sum(exp_terms)) / beta
    
    return -soft_min_val

def _optimize_gradient(points, deadline):
    """
    Optimize the packing using L-BFGS-B on the soft-min objective.
    Returns the best points found before the deadline or convergence.
    """
    x0 = points.flatten()
    
    # Objective function wrapper
    def objective(x):
        return _soft_min_objective(x, beta=50.0)
        
    # We use L-BFGS-B which is efficient for large problems.
    # We monitor time and stop if we exceed deadline.
    # Since minimize() doesn't easily support time-based stopping in a callback that halts cleanly 
    # without complex state management, we will run it with a reasonable maxiter 
    # and rely on the fact that convergence is usually fast for this smooth objective.
    # However, to be safe, we can run it in a way that we can interrupt if needed, 
    # but standard minimize blocks. 
    # Given the 150s limit, and L-BFGS-B speed, one call should be safe.
    
    res = minimize(objective, x0, method='L-BFGS-B', options={'maxiter': 100, 'ftol': 1e-12, 'gtol': 1e-8})
    
    if res.success:
        return res.x.reshape(N, 2)
    else:
        return res.x.reshape(N, 2)

def construct():
    """Construct the optimal packing."""
    start_time = time.time()
    deadline = start_time + 140.0  # 10 second safety margin
    
    # 1. Generate initial hexagonal lattice
    points, init_r = _generate_hexagonal_lattice()
    
    if points is None:
        # Fallback: simple grid or random
        points = np.random.randn(N, 2) * 0.5
        init_r = 0.0
    
    # 2. Gradient-based optimization
    # The soft-min objective is smooth and differentiable.
    # L-BFGS-B is a good choice for this.
    
    if time.time() < deadline - 5:
        opt_points = _optimize_gradient(points, deadline)
        opt_r = _compute_radius(opt_points)
        
        if opt_r > init_r:
            points = opt_points
            init_r = opt_r
    
    # 3. If we have time left, we could try a second pass with a different beta or start,
    # but the current strategy is a single pass of gradient optimization from a good start.
    # The key improvement here is the use of a smooth objective function that allows
    # the optimizer to navigate the space more effectively than discrete swaps or
    # non-smooth heuristics.
    
    return points
# EVOLVE-BLOCK-END
