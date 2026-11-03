import numpy as np
import time
from scipy.optimize import minimize
import random

N = 100
DEADLINE_MARGIN = 5.0  # seconds before the 150 s kill

def _score(points):
    """Return the maximal radius for the given set of centres."""
    if len(points) != N:
        return 0.0
    pts = np.asarray(points, dtype=np.float64)
    if not np.all(np.isfinite(pts)):
        return 0.0
    margin = 0.5 - np.abs(pts).max(axis=1)
    r_margin = margin.min()
    if r_margin <= 0:
        return 0.0
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dists = np.sqrt((diff ** 2).sum(axis=2))
    np.fill_diagonal(dists, np.inf)
    min_dist = dists.min()
    r_pair = min_dist / 2.0
    return min(r_pair, r_margin)

def _compute_min_radius(pts, fixed_pts):
    """
    Compute the minimum radius for the new point `pts` (shape 3,) 
    relative to `fixed_pts` (shape M, 3) and the boundary.
    """
    # Boundary constraint
    r_boundary = 0.5 - np.abs(pts).max()
    
    if len(fixed_pts) == 0:
        return r_boundary

    # Distance to fixed points
    diffs = pts - fixed_pts
    dists = np.sqrt(np.sum(diffs**2, axis=1))
    r_pairs = dists.min() / 2.0
    
    return min(r_boundary, r_pairs)

def _objective_and_grad(x, fixed_pts):
    """
    Objective: minimize the negative of the minimum radius.
    We want to maximize min(r_boundary, min_dist/2).
    This is equivalent to minimizing -min(r_boundary, min_dist/2).
    Since min is non-smooth, we use a soft-min or just rely on the gradient 
    of the active constraint. However, L-BFGS-B handles box constraints.
    
    Actually, a better formulation for optimization is:
    Minimize max(0, 2*max_dist_to_fixed - 2*r) ... no.
    
    Let's stick to maximizing the "safe radius".
    r(x) = min(0.5 - max|x|, min_j |x - fixed_j|/2)
    
    Gradient of r(x) is tricky because of the min.
    We can approximate the gradient by finding the "bottleneck" constraint.
    
    Alternatively, we can minimize an energy function:
    E(x) = sum_j (1/|x - fixed_j|^2) + boundary_penalty
    This pushes x away from all fixed points and the boundary.
    The minimum of this energy corresponds to the point that is "most central" 
    relative to the fixed points, which usually maximizes the minimum distance.
    """
    diffs = x - fixed_pts
    dists_sq = np.sum(diffs**2, axis=1)
    dists_sq = np.maximum(dists_sq, 1e-12) # Avoid div by zero
    
    # Energy: sum of inverse squares
    # This is a standard repulsion potential
    E = np.sum(1.0 / dists_sq)
    
    # Boundary penalty: we want to stay away from boundaries too.
    # We can add a term for distance to boundary.
    # Let d_b = 0.5 - max|x|. We want to maximize d_b.
    # So minimize -d_b? Or add penalty if close to boundary.
    # Let's add a term: (0.5 - |x|)^-2 for each coordinate?
    # Or simpler: just use the repulsion from fixed points and let the 
    # optimizer find the spot. But if fixed points are far, it might go to center.
    # The boundary is a constraint.
    
    # Let's use a combined energy that includes boundary repulsion.
    # Boundary repulsion: for each coordinate i, if |x_i| is close to 0.5, penalty increases.
    # Let m_i = 0.5 - |x_i|.
    # Penalty = sum_i 1/m_i^2 ? No, that explodes at boundary.
    # We want to maximize min(m_i, min_dist/2).
    
    # Let's try a different objective:
    # Maximize r such that |x - fixed_j| >= 2r and |x| <= 0.5 - r.
    # This is a feasibility problem.
    
    # Back to energy minimization:
    # E(x) = sum_j (1/|x - fixed_j|^2) + sum_k (1/(0.5 - |x_k|)^2)
    # This will find a local maximum of the minimum distance.
    
    m = 0.5 - np.abs(x)
    m = np.maximum(m, 1e-6)
    boundary_energy = np.sum(1.0 / (m**2))
    
    total_E = E + 10.0 * boundary_energy
    
    # Gradient
    # dE/dx_i = sum_j -2 * (x - fixed_j) / |x - fixed_j|^4
    grad_E = np.zeros(3)
    for j in range(len(fixed_pts)):
        d = x - fixed_pts[j]
        d_sq = np.sum(d**2)
        d_sq = max(d_sq, 1e-12)
        grad_E += -2.0 * d / (d_sq**2)
        
    # d(boundary_energy)/dx_k
    # b_k = 1/(0.5 - |x_k|)^2
    # db_k/dx_k = -2 * (0.5 - |x_k|)^-3 * (-sign(x_k)) = 2 * sign(x_k) / (0.5 - |x_k|)^3
    grad_b = np.zeros(3)
    for k in range(3):
        m_k = 0.5 - abs(x[k])
        m_k = max(m_k, 1e-6)
        sign = np.sign(x[k])
        if sign == 0: sign = 1
        grad_b[k] = 10.0 * 2.0 * sign / (m_k**3)
        
    total_grad = grad_E + grad_b
    
    return total_E, total_grad

def _place_sphere(fixed_pts, n_tries=5, maxiter=50):
    """
    Place a new sphere by optimizing its position.
    """
    best_pos = None
    best_r = -1.0
    
    # Try multiple random starts
    for _ in range(n_tries):
        # Random start inside cube
        x0 = np.random.uniform(-0.4, 0.4, 3)
        
        try:
            res = minimize(_objective_and_grad, x0, args=(fixed_pts,), 
                           method='L-BFGS-B', jac=True,
                           options={'maxiter': maxiter, 'gtol': 1e-6})
            x_opt = res.x
            
            # Check validity
            if not np.all(np.isfinite(x_opt)):
                continue
            if np.abs(x_opt).max() >= 0.5:
                continue
                
            r = _compute_min_radius(x_opt, fixed_pts)
            
            if r > best_r:
                best_r = r
                best_pos = x_opt
                
        except Exception:
            continue
            
    if best_pos is None:
        # Fallback: random
        best_pos = np.random.uniform(-0.4, 0.4, 3)
        best_r = _compute_min_radius(best_pos, fixed_pts)
        
    return best_pos, best_r

def _local_search(points, maxiter=100):
    """
    Perform a local search to improve the packing.
    We use a simple iterative improvement:
    For each point, try to move it slightly to increase the global minimum radius.
    """
    current_points = points.copy()
    current_score = _score(current_points)
    
    for _ in range(10): # Limit iterations for time
        improved = False
        for i in range(N):
            # Hold all other points fixed
            other_pts = np.delete(current_points, i, axis=0)
            x0 = current_points[i]
            
            # Optimize point i
            def obj_i(x):
                # We want to maximize the minimum distance to other_pts and boundary
                # But we also care about the global score.
                # A simple proxy: maximize min(dist to others, dist to boundary)
                # This is what _objective_and_grad does (minimizing energy).
                E, g = _objective_and_grad(x, other_pts)
                return E, g
            
            try:
                res = minimize(obj_i, x0, method='L-BFGS-B', jac=True,
                               options={'maxiter': 20, 'gtol': 1e-5})
                new_x = res.x
                if np.all(np.isfinite(new_x)) and np.abs(new_x).max() < 0.5:
                    new_points = current_points.copy()
                    new_points[i] = new_x
                    new_score = _score(new_points)
                    if new_score > current_score:
                        current_points = new_points
                        current_score = new_score
                        improved = True
            except:
                pass
        if not improved:
            break
            
    return current_points

def construct():
    start_time = time.time()
    deadline = start_time + 150.0 - DEADLINE_MARGIN
    
    np.random.seed(42)
    random.seed(42)
    
    # Phase 1: Greedy Sequential Insertion
    points = []
    
    # First sphere: center
    points.append(np.array([0.0, 0.0, 0.0]))
    
    for i in range(1, N):
        if time.time() > deadline - 20.0:
            break
            
        fixed_pts = np.array(points)
        
        # Adaptive number of tries based on remaining time
        remaining = deadline - time.time()
        if remaining > 60:
            n_tries = 10
            maxiter = 50
        elif remaining > 30:
            n_tries = 5
            maxiter = 30
        else:
            n_tries = 2
            maxiter = 10
            
        pos, r = _place_sphere(fixed_pts, n_tries=n_tries, maxiter=maxiter)
        points.append(pos)
        
    # If we didn't place all spheres, fill with random valid points
    while len(points) < N:
        x = np.random.uniform(-0.45, 0.45, 3)
        # Check if it's valid (non-overlapping)
        valid = True
        for p in points:
            if np.linalg.norm(x - p) < 0.1: # Rough check, will be fixed in local search
                valid = False
                break
        if valid:
            points.append(x)
        else:
            # Just add it, local search will fix
            points.append(x)
            
    points = np.array(points)
    
    # Phase 2: Local Search Improvement
    if time.time() < deadline - 10.0:
        # Run local search
        # We need to be careful with the time here.
        # _local_search is O(N * N * iter).
        # Let's run it for a few seconds.
        local_start = time.time()
        points = _local_search(points)
        
    # Final check and clip
    points = np.clip(points, -0.4999, 0.4999)
    
    return points

# EVOLVE-BLOCK-END
