# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import pdist
import time

def _compute_score(points):
    """Compute R = (d_min/d_max)^2 for a set of points."""
    if len(points) < 2:
        return 0.0
    d = pdist(points)
    d_min = np.min(d)
    d_max = np.max(d)
    if d_max == 0:
        return 0.0
    if d_min == 0:
        return 0.0
    return (d_min / d_max) ** 2

def _min_dist(points_flat):
    """Compute minimum pairwise distance from flat array."""
    n = 16
    points = points_flat.reshape(n, 2)
    d = pdist(points)
    return np.min(d)

def _max_dist(points_flat):
    """Compute maximum pairwise distance from flat array."""
    n = 16
    points = points_flat.reshape(n, 2)
    d = pdist(points)
    return np.max(d)

def _objective_minimize_neg_min_dist(points_flat, r_max):
    """Objective: maximize min distance (minimize negative min distance).
    Constraint: all pairwise distances <= r_max."""
    n = 16
    points = points_flat.reshape(n, 2)
    d = pdist(points)
    d_min = np.min(d)
    d_max = np.max(d)
    
    # Penalty if max distance exceeds bound
    if d_max > r_max:
        penalty = 1e6 * (d_max - r_max)
    else:
        penalty = 0.0
    
    return -d_min + penalty

def _constraint_max_dist(points_flat, r_max):
    """Constraint: max distance <= r_max (returns >= 0 for feasibility)."""
    d_max = _max_dist(points_flat)
    return r_max - d_max

def _generate_hexagonal_grid(n):
    """Generate a hexagonal grid of approximately n points."""
    points = []
    for j in range(4):
        for i in range(4):
            x = i + 0.5 * j
            y = j * np.sqrt(3) / 2
            points.append([x, y])
    points = np.array(points)
    points -= np.mean(points, axis=0)
    return points

def _generate_circle(n):
    """Generate points on a circle."""
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    points = np.column_stack([np.cos(angles), np.sin(angles)])
    return points

def _generate_random(n, seed):
    """Generate random points."""
    np.random.seed(seed)
    points = np.random.randn(n, 2)
    points -= np.mean(points, axis=0)
    return points

def _generate_square_grid(n):
    """Generate a square grid."""
    side = int(np.ceil(np.sqrt(n)))
    points = []
    count = 0
    for i in range(side):
        for j in range(side):
            if count < n:
                points.append([i, j])
                count += 1
    points = np.array(points[:n])
    points -= np.mean(points, axis=0)
    return points

def _optimize_from_init(init_pts, r_max, start_time, deadline):
    """Run SLSQP optimization from a given initial point with max distance constraint."""
    n = 16
    x0 = init_pts.flatten()
    
    # Define constraints
    constraints = [
        {'type': 'ineq', 'fun': lambda x: _constraint_max_dist(x, r_max)}
    ]
    
    try:
        result = minimize(
            _objective_minimize_neg_min_dist,
            x0,
            args=(r_max,),
            method='SLSQP',
            constraints=constraints,
            options={'maxiter': 500, 'ftol': 1e-12}
        )
        
        if time.time() > deadline:
            return None, -1.0
        
        if result.fun < 1e5:  # Not heavily penalized
            opt_pts = result.x.reshape(n, 2)
            score = _compute_score(opt_pts)
            return opt_pts, score
        else:
            return None, -1.0
    except Exception:
        return None, -1.0

def min_max_dist_dim2_16() -> np.ndarray:
    """
    Creates 16 points in 2 dimensions in order to maximize the ratio of minimum to maximum distance.

    Returns
        points: np.ndarray of shape (16,2) containing the (x,y) coordinates of the points.

    """
    n = 16
    start_time = time.time()
    deadline = start_time + 580  # Safety margin before 630s kill

    # Generate diverse initial candidates
    candidates = []
    
    try:
        hex_pts = _generate_hexagonal_grid(n)
        candidates.append(hex_pts)
    except:
        pass

    try:
        circ_pts = _generate_circle(n)
        candidates.append(circ_pts)
    except:
        pass

    try:
        sq_pts = _generate_square_grid(n)
        candidates.append(sq_pts)
    except:
        pass

    for seed in [42, 123, 456, 789, 101, 202, 303, 404]:
        try:
            rand_pts = _generate_random(n, seed)
            candidates.append(rand_pts)
        except:
            pass

    best_points = None
    best_score = -1.0

    # First pass: evaluate initial candidates
    for init_pts in candidates:
        score = _compute_score(init_pts)
        if score > best_score:
            best_score = score
            best_points = init_pts

    # Second pass: try SLSQP optimization with different r_max values
    # Start with a reasonable r_max estimate
    for init_pts in candidates:
        if time.time() > deadline - 30:
            break
            
        d_max_init = _max_dist(init_pts.flatten())
        d_min_init = _min_dist(init_pts.flatten())
        
        # Try different r_max values around the initial max distance
        for r_mult in [0.8, 0.9, 1.0, 1.1, 1.2]:
            if time.time() > deadline - 15:
                break
                
            r_max = d_max_init * r_mult
            opt_pts, score = _optimize_from_init(init_pts, r_max, start_time, deadline)
            
            if opt_pts is not None and score > best_score:
                best_score = score
                best_points = opt_pts

    # Third pass: if we have a good candidate, try to refine it further
    if best_points is not None and time.time() < deadline - 20:
        d_max_best = _max_dist(best_points.flatten())
        d_min_best = _min_dist(best_points.flatten())
        
        # Try tighter constraints
        for r_mult in [0.95, 0.98, 1.0, 1.02, 1.05]:
            if time.time() > deadline - 10:
                break
                
            r_max = d_max_best * r_mult
            opt_pts, score = _optimize_from_init(best_points, r_max, start_time, deadline)
            
            if opt_pts is not None and score > best_score:
                best_score = score
                best_points = opt_pts

    # Fourth pass: try to improve by scaling
    if best_points is not None and time.time() < deadline - 5:
        # The score is scale-invariant, but let's try slight perturbations
        for _ in range(3):
            if time.time() > deadline - 5:
                break
            np.random.seed(int(time.time() * 1000) % 10000)
            perturbed = best_points + np.random.randn(n, 2) * 0.01
            score = _compute_score(perturbed)
            if score > best_score:
                best_score = score
                best_points = perturbed

    if best_points is None:
        # Absolute fallback
        return np.random.randn(16, 2)

    return best_points

# EVOLVE-BLOCK-END
