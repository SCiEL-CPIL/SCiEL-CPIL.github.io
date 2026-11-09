"""
Spherical Tammes/Covering optimization for N=130 points.
Strategy:
1. High-quality initialization (Fibonacci sphere).
2. Alternating Lloyd's Algorithm (Voronoi centroid relaxation) and
   projected gradient ascent with soft-max repulsion.
   Lloyd's moves points to Voronoi cell centroids (structural), while
   gradient ascent with soft-max penalty on closest neighbors enforces
   uniform repulsive pressure targeting the covering radius.
3. Final polishing with L-BFGS-B using a smooth log-sum-exp surrogate
   for the nonsmooth max-circumradius objective.
"""
import numpy as np
from scipy.optimize import minimize
from scipy.spatial import ConvexHull
import time

N = 130

def _normalize(points):
    """Normalize points to unit sphere."""
    norms = np.linalg.norm(points, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-10)
    return points / norms

def _covering_radius(points):
    """
    Compute the covering radius in radians.
    The covering radius is the maximum circumradius of the facets
    of the convex hull of the points.
    """
    try:
        hull = ConvexHull(points)
        max_circumradius = 0.0
        for simplex in hull.simplices:
            tri = points[simplex]
            a, b, c = tri[0], tri[1], tri[2]
            
            # Circumcenter x satisfies x·a = x·b = x·c and |x|=1
            # x is in the intersection of planes x·(a-b)=0 and x·(b-c)=0
            v1 = a - b
            v2 = b - c
            x_dir = np.cross(v1, v2)
            x_norm = np.linalg.norm(x_dir)
            if x_norm < 1e-10:
                continue
            x = x_dir / x_norm
            
            # Ensure x is on the same side as the triangle centroid (outward normal)
            centroid = (a + b + c) / 3.0
            centroid_norm = np.linalg.norm(centroid)
            if centroid_norm > 1e-10:
                centroid = centroid / centroid_norm
                if np.dot(x, centroid) < 0:
                    x = -x
            
            cos_R = np.clip(np.dot(x, a), -1.0, 1.0)
            R = np.arccos(cos_R)
            
            max_circumradius = max(max_circumradius, R)
        
        return max_circumradius
    except Exception:
        # Fallback: grid-based approximation
        theta = np.linspace(0, 2*np.pi, 100)
        phi = np.linspace(0, np.pi, 100)
        TH, PH = np.meshgrid(theta, phi)
        X = np.sin(PH) * np.cos(TH)
        Y = np.sin(PH) * np.sin(TH)
        Z = np.cos(PH)
        grid_points = np.column_stack([X.ravel(), Y.ravel(), Z.ravel()])
        
        dots = grid_points @ points.T
        cos_angles = np.clip(dots, -1.0, 1.0)
        angles = np.arccos(cos_angles)
        min_angles = np.min(angles, axis=1)
        return np.max(min_angles)

def _fibonacci_sphere(n):
    """Generate Fibonacci sphere points."""
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])

def _lloyd_iteration(points, grid):
    """
    One iteration of Lloyd's algorithm.
    Move each point to the centroid of its Voronoi cell (approximated by grid).
    """
    n = len(points)
    new_points = np.zeros_like(points)
    
    # Assign each grid point to its nearest configuration point
    # dot product is maximized when angle is minimized
    dots = grid @ points.T  # (num_grid, N)
    nearest = np.argmax(dots, axis=1)
    
    # Compute centroid of Voronoi cell for each point
    for i in range(n):
        mask = (nearest == i)
        count = np.sum(mask)
        if count > 0:
            cell_points = grid[mask]
            centroid = np.mean(cell_points, axis=0)
            centroid_norm = np.linalg.norm(centroid)
            if centroid_norm > 1e-10:
                new_points[i] = centroid / centroid_norm
            else:
                new_points[i] = points[i]
        else:
            new_points[i] = points[i]
            
    return new_points

def _lloyd_optimize(points, time_limit):
    """
    Run alternating Lloyd's algorithm and projected gradient ascent
    with soft-max repulsion for a specified time limit.
    """
    start_time = time.time()
    
    # Create a fine grid for Voronoi approximation
    theta = np.linspace(0, 2*np.pi, 200)
    phi = np.linspace(0, np.pi, 200)
    TH, PH = np.meshgrid(theta, phi)
    X = np.sin(PH) * np.cos(TH)
    Y = np.sin(PH) * np.sin(TH)
    Z = np.cos(PH)
    grid = np.column_stack([X.ravel(), Y.ravel(), Z.ravel()])
    
    current = points.copy()
    best = current.copy()
    best_val = _covering_radius(best)
    
    for it in range(1000):
        if time.time() - start_time > time_limit:
            break
        
        # Alternate between Lloyd's and gradient ascent
        if it % 3 == 0:
            # Lloyd's iteration (structural)
            new_pts = _lloyd_iteration(current, grid)
        else:
            # Projected gradient ascent with soft-max repulsion
            new_pts = _projected_gradient_step(current, lr=0.05)
        
        # Check convergence
        if np.max(np.linalg.norm(new_pts - current, axis=1)) < 1e-9:
            break
            
        current = new_pts
        
        # Evaluate
        val = _covering_radius(current)
        if val < best_val:
            best_val = val
            best = current.copy()
            
    return best, best_val

def _projected_gradient_step(points, lr=0.05):
    """
    One step of projected gradient ascent with soft-max repulsion.
    Each point is pushed away from its closest neighbors with weights
    based on a soft-max of the inverse angular distances.
    """
    n = len(points)
    
    # Compute pairwise dot products (cosines of angles)
    dots = points @ points.T  # (N, N)
    np.fill_diagonal(dots, 2.0)  # Set diagonal to avoid self-interaction
    
    # Angular distances
    cos_angles = np.clip(dots, -1.0, 1.0)
    angles = np.arccos(cos_angles)
    
    # For each point i, compute soft-max weights on other points j
    # We want to push away from closest neighbors
    # Use soft-max of negative angles (so closer points get higher weight)
    beta = 50.0
    neg_angles = -beta * angles
    # Mask diagonal
    neg_angles = np.where(np.eye(n, dtype=bool), -1e10, neg_angles)
    
    # Soft-max weights
    max_neg = np.max(neg_angles, axis=1, keepdims=True)
    exp_neg = np.exp(neg_angles - max_neg)
    weights = exp_neg / np.sum(exp_neg, axis=1, keepdims=True)
    
    # Vectorized computation of weighted sum of tangent vectors
    away_vectors = points[:, np.newaxis, :] - points[np.newaxis, :, :]  # (N, N, 3)
    
    # Project onto tangent plane at point_i
    dots_away = np.einsum('ijk,ik->ij', away_vectors, points)
    tangent_away = away_vectors - dots_away[:, :, np.newaxis] * points[:, np.newaxis, :]
    
    # weighted_sum[i] = sum_j weights[i, j] * tangent_away[i, j]
    weighted_sum = np.einsum('ij,ijk->ik', weights, tangent_away)
    
    # Update
    new_points = points + lr * weighted_sum
    
    # Normalize
    norms = np.linalg.norm(new_points, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-10)
    new_points = new_points / norms
    
    return new_points

def _pairwise_soft_max_obj(points, beta=100.0):
    """
    Smooth surrogate for maximizing the minimum angular separation.
    Minimizes the log-sum-exp of the negative dot products.
    """
    dots = points @ points.T
    mask = ~np.eye(len(points), dtype=bool)
    dots = dots[mask]
    
    shifted = -beta * dots
    max_shifted = np.max(shifted)
    log_sum_exp = max_shifted / beta + np.log(np.sum(np.exp(shifted - max_shifted))) / beta
    return log_sum_exp

def _polish(points, time_limit):
    """
    Final polish using L-BFGS-B with a smooth pairwise log-sum-exp surrogate.
    """
    start_time = time.time()
    
    def flat_objective_smooth(flat):
        pts = _normalize(flat)
        return _pairwise_soft_max_obj(pts, beta=100.0)
    
    def flat_objective_exact(flat):
        pts = _normalize(flat)
        return _covering_radius(pts)
    
    best = points.copy()
    best_val = _covering_radius(best)
    
    # Phase 1: Smooth optimization with pairwise surrogate
    try:
        result = minimize(
            flat_objective_smooth,
            points.ravel(),
            method='L-BFGS-B',
            options={'maxiter': 500, 'maxfun': 1000, 'ftol': 1e-12, 'gtol': 1e-10}
        )
        if result.fun < best_val + 0.01:
            candidate = _normalize(result.x)
            cand_val = _covering_radius(candidate)
            if cand_val < best_val:
                best = candidate
                best_val = cand_val
    except Exception:
        pass
    
    # Phase 2: Tight polish with exact objective if time permits
    if time.time() - start_time < time_limit * 0.5:
        try:
            result2 = minimize(
                flat_objective_exact,
                best.ravel(),
                method='L-BFGS-B',
                options={'maxiter': 100, 'maxfun': 200, 'ftol': 1e-14, 'gtol': 1e-12}
            )
            if result2.fun < best_val:
                best = _normalize(result2.x)
                best_val = result2.fun
        except Exception:
            pass
    
    return best, best_val

def construct():
    """Construct 130 points on the unit sphere to minimize covering radius."""
    start_time = time.time()
    deadline = start_time + 140.0  # 10s buffer for 150s limit
    
    # 1. Initialize
    points = _fibonacci_sphere(N)
    
    # 2. Alternating Lloyd's + Gradient Ascent
    main_time = 100.0
    points, val = _lloyd_optimize(points, main_time)
    
    # 3. Polish with smooth surrogate then exact objective
    if time.time() < deadline - 10:
        points, val = _polish(points, 30.0)
        
    # 4. Final sanity check and normalization
    points = _normalize(points)
    
    # Ensure no coincident points
    for i in range(N):
        for j in range(i+1, N):
            if np.linalg.norm(points[i] - points[j]) < 1e-8:
                points[j] = _normalize(points[j] + np.random.normal(0, 0.01, 3))
                
    return points
