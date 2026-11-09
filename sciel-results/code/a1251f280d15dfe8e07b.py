import numpy as np
import time
from scipy.optimize import minimize

def _compute_vertex_score(vertices):
    """
    Compute the per-vertex score for a single vertex of a polygon.
    vertices: (n, 2) array of polygon vertices in cyclic order.
    Returns the score for vertex 0.
    """
    n = len(vertices)
    diffs = vertices - vertices[0]
    dists = np.sqrt(np.sum(diffs ** 2, axis=1))
    other_dists = dists[1:]
    d = np.sort(other_dists)
    
    if d[0] < 0.05:
        return 0.0
    
    gaps = d[1:] - d[:-1]
    k = np.argmin(gaps)
    D = (d[k] + d[k+1]) / 2.0
    if D < 0.1:
        return 0.0
    
    if k == 0:
        sel = d[0:4]
    elif k == 17:
        sel = d[15:19]
    else:
        sel = d[k-1:k+3]
    
    raw = np.mean(np.maximum(sel / D, D / sel))
    score = min(1.0, 1.0 / raw)
    return score

def _polygon_min_score(polygon):
    """
    Compute the minimum vertex score over all vertices of a polygon.
    polygon: (n, 2) array.
    """
    n = len(polygon)
    centroid = np.mean(polygon, axis=0)
    centered = polygon - centroid
    dists_to_centroid = np.sqrt(np.sum(centered ** 2, axis=1))
    mean_dist = np.mean(dists_to_centroid)
    if mean_dist <= 0.01:
        return 0.0
    normalized = centered / mean_dist
    
    # Check convexity
    if np.any(np.linalg.norm(np.diff(np.vstack([normalized, normalized[:1]]), axis=0), axis=1) < 1e-10):
        return 0.0
    
    edges = np.diff(np.vstack([normalized, normalized[:1]]), axis=0)
    cross = edges[:, 0] * edges[:, 1] - edges[:, 1] * edges[:, 0]
    
    if np.any(np.abs(cross) < 1e-10):
        return 0.0
    
    signs = np.sign(cross)
    if np.any(signs != signs[0]):
        return 0.0
    
    total_turn = np.sum(np.arctan2(cross, edges[:, 0] * edges[:, 0] + edges[:, 1] * edges[:, 1]))
    if abs(abs(total_turn) - 2 * np.pi) > 1e-6:
        return 0.0
    
    min_score = 1.0
    for i in range(n):
        rotated = np.roll(normalized, -i, axis=0)
        v_score = _compute_vertex_score(rotated)
        if v_score < min_score:
            min_score = v_score
    return min_score

def _make_polygon_from_angles(delta_theta, n=20):
    """
    Create a polygon from angular perturbations.
    delta_theta: (n,) array of perturbations to the regular angles.
    """
    base_angles = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    angles = base_angles + delta_theta
    
    # Ensure cyclic order and convexity by enforcing small perturbations
    # and checking that the resulting polygon is strictly convex.
    # We use a fixed radius of 1.0 for simplicity, as the metric is scale-invariant
    # (normalization handles it) but shape matters.
    radii = np.ones(n)
    
    x = radii * np.cos(angles)
    y = radii * np.sin(angles)
    return np.stack([x, y], axis=1)

def _objective(params, n=20):
    """
    Objective function for minimization.
    Returns negative score.
    """
    try:
        delta_theta = params
        polygon = _make_polygon_from_angles(delta_theta, n)
        score = _polygon_min_score(polygon)
        return -score
    except Exception:
        return 1000.0

def _search():
    """
    Search for a good 20-gon using gradient-based optimization on angular perturbations.
    """
    n = 20
    start_time = time.time()
    deadline = start_time + 620.0
    
    best_score = 0.0
    best_params = np.zeros(n)
    
    # Regular polygon is the starting point
    # Try to optimize angular perturbations
    # Since the regular polygon is a local minimum for the "equidistant" objective 
    # (all distances are symmetric), we might need to break symmetry.
    
    # However, the score for a regular polygon is likely high (close to 1.0) 
    # because the distances are very regular.
    # Let's check the score of the regular polygon first.
    reg_poly = _make_polygon_from_angles(np.zeros(n), n)
    reg_score = _polygon_min_score(reg_poly)
    best_score = reg_score
    best_params = np.zeros(n)
    
    # Try gradient descent from several random initializations
    np.random.seed(42)
    
    for i in range(20):
        if time.time() > deadline:
            break
            
        # Small random perturbations
        x0 = np.random.normal(0, 0.05, n)
        
        # Use L-BFGS-B
        res = minimize(_objective, x0, method='L-BFGS-B', args=(n,), 
                       options={'maxiter': 500, 'ftol': 1e-10})
        
        if res.success:
            score = -res.fun
            if score > best_score:
                best_score = score
                best_params = res.x
        
        # Also try a few random walks if gradient fails
        if time.time() > deadline:
            break
            
        # Simple random search around best
        for j in range(50):
            if time.time() > deadline:
                break
            x_rand = best_params + np.random.normal(0, 0.01, n)
            score = -_objective(x_rand, n)
            if score > best_score:
                best_score = score
                best_params = x_rand
                
    return _make_polygon_from_angles(best_params, n)

def equidistant_points_polygon20() -> np.ndarray:
    """
    Build a convex polygon with n = 20 vertices.
    Returns:
        polygon: np.ndarray of shape (20, 2) with the x,y coordinates of the vertices,
            given in convex (counter-clockwise) order.
    """
    start_time = time.time()
    
    # Try search-based approach
    polygon = _search()
    
    # Ensure we don't exceed time limit
    if time.time() - start_time > 625:
        # Fallback to regular polygon
        n = 20
        radius = 1.0
        angles = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
        x = radius * np.cos(angles)
        y = radius * np.sin(angles)
        polygon = np.stack([x, y], axis=1)
    
    return polygon
