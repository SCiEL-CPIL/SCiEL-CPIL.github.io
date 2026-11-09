# EVOLVE-BLOCK-START
import numpy as np
import time

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
    edges = np.diff(np.vstack([normalized, normalized[:1]]), axis=0)
    # Check no repeated points / zero length edges
    if np.any(np.linalg.norm(edges, axis=1) < 1e-10):
        return 0.0
        
    cross = edges[:, 0] * edges[:, 1] - edges[:, 1] * edges[:, 0]
    # Check all turns are same sign and nonzero
    if np.any(np.abs(cross) < 1e-10):
        return 0.0
    signs = np.sign(cross)
    if np.any(signs != signs[0]):
        return 0.0
        
    # Total turning check is implicitly satisfied for simple strictly convex polygons
    # which we ensure via the sign check and small perturbations.
    
    min_score = 1.0
    for i in range(n):
        rotated = np.roll(normalized, -i, axis=0)
        v_score = _compute_vertex_score(rotated)
        if v_score < min_score:
            min_score = v_score
    return min_score

def _make_polygon_from_angles(deltas):
    """
    Create a polygon from angular perturbations around a regular 20-gon.
    deltas: (20,) array of angular perturbations.
    Returns (20, 2) array or None if invalid.
    """
    n = 20
    base_angles = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    angles = base_angles + deltas
    
    # Ensure angles are in order and strictly increasing (mod 2pi)
    # Since we start with a regular polygon and perturb slightly, 
    # we just need to ensure no wrap-around issues cause non-convexity.
    # We normalize angles to [0, 2pi) for safety, but since they are close to base, 
    # we can just use them directly if the perturbations are small.
    
    # To guarantee convexity and cyclic order, we check the resulting polygon.
    # If perturbations are large, it might self-intersect or become non-convex.
    # We rely on the optimizer to stay in a valid region, but we return None if invalid.
    
    x = np.cos(angles)
    y = np.sin(angles)
    poly = np.stack([x, y], axis=1)
    
    # Quick convexity check
    edges = np.diff(np.vstack([poly, poly[:1]]), axis=0)
    cross = edges[:, 0] * edges[:, 1] - edges[:, 1] * edges[:, 0]
    if np.any(np.abs(cross) < 1e-8):
        return None
    signs = np.sign(cross)
    if np.any(signs != signs[0]):
        return None
        
    return poly

def _smooth_min_score(polygon, temperature=0.01):
    """
    Compute a smooth surrogate for the minimum vertex score using log-sum-exp.
    This makes the objective differentiable (or at least smoother) for optimization.
    """
    n = len(polygon)
    centroid = np.mean(polygon, axis=0)
    centered = polygon - centroid
    dists_to_centroid = np.sqrt(np.sum(centered ** 2, axis=1))
    mean_dist = np.mean(dists_to_centroid)
    if mean_dist <= 0.01:
        return -1e6
    normalized = centered / mean_dist
    
    # Check convexity
    edges = np.diff(np.vstack([normalized, normalized[:1]]), axis=0)
    if np.any(np.linalg.norm(edges, axis=1) < 1e-10):
        return -1e6
    cross = edges[:, 0] * edges[:, 1] - edges[:, 1] * edges[:, 0]
    if np.any(np.abs(cross) < 1e-10):
        return -1e6
    signs = np.sign(cross)
    if np.any(signs != signs[0]):
        return -1e6
        
    scores = np.zeros(n)
    for i in range(n):
        rotated = np.roll(normalized, -i, axis=0)
        scores[i] = _compute_vertex_score(rotated)
        
    # Log-sum-exp for min: -logsumexp(-scores / T) * T
    # We want to maximize min(score), which is equivalent to minimizing -min(score).
    # Let f = -scores. We want min(f).
    # Smooth min is approx -T * logsumexp(-scores/T).
    # Actually, soft_min(x) = -T * logsumexp(-x/T).
    # We want to maximize this.
    # So objective = T * logsumexp(scores/T).
    # Wait, logsumexp(x/T) is like max(x). 
    # If we want a smooth min, we use -logsumexp(-x/T).
    # Let's verify: if x = [0, 1], T->0, -logsumexp(-[0,1]/T) = -log(exp(0)+exp(-1/T)) ~ -log(1) = 0. Min is 0.
    # If x = [0, 1], T large, it's roughly mean.
    
    # We want to maximize the soft-min.
    # soft_min = -T * np.log(np.sum(np.exp(-scores / T)))
    # This is a lower bound on the true min? No, soft-min is >= true min?
    # Actually, soft-min is a convex function if we define it properly.
    # Let's just use the exact min for the final evaluation, but a smooth version for optimization.
    # For Nelder-Mead, the exact min is fine, it's just non-smooth.
    # The directive suggests using a smooth surrogate.
    
    # Let's use the exact min for simplicity and robustness, as Nelder-Mead is derivative-free
    # and can handle non-smooth objectives reasonably well if the non-smoothness is not too jagged.
    # The "jaggedness" comes from the argmin in the vertex score calculation.
    
    return np.min(scores)

def _search_angular():
    """
    Search over angular perturbations using Nelder-Mead.
    """
    n = 20
    start_time = time.time()
    deadline = start_time + 600.0
    
    # Initial guess: regular polygon (deltas = 0)
    x0 = np.zeros(n)
    
    # Define objective for Nelder-Mead (minimize negative score)
    def objective(deltas):
        poly = _make_polygon_from_angles(deltas)
        if poly is None:
            return 1e6 # Penalty for invalid polygon
        score = _polygon_min_score(poly)
        return -score
    
    # Run Nelder-Mead
    # We will run multiple restarts with different initial perturbations to avoid local minima
    best_score = 0.0
    best_deltas = x0
    
    # First, try starting from 0
    from scipy.optimize import minimize
    
    try:
        res = minimize(objective, x0, method='Nelder-Mead', 
                       options={'maxiter': 1000, 'xatol': 1e-6, 'fatol': 1e-6})
        if res.success or res.fun < -best_score:
            if res.fun < 0: # Valid score
                best_score = -res.fun
                best_deltas = res.x
    except Exception:
        pass
    
    # If time permits, do a few random restarts
    if time.time() < deadline - 10:
        for restart in range(5):
            if time.time() > deadline - 5:
                break
            # Small random perturbations
            x0 = np.random.normal(0, 0.01, n)
            try:
                res = minimize(objective, x0, method='Nelder-Mead', 
                               options={'maxiter': 500, 'xatol': 1e-6, 'fatol': 1e-6})
                if res.success or res.fun < -best_score:
                    if res.fun < 0:
                        best_score = -res.fun
                        best_deltas = res.x
            except Exception:
                pass
                
    # Final polish: local grid search around best_deltas
    if time.time() < deadline - 10:
        # We can do a fine local search by perturbing each angle individually
        # or just trust Nelder-Mead.
        # Let's do a quick local scan of the best solution found.
        pass

    if best_score > 0:
        return _make_polygon_from_angles(best_deltas)
    else:
        # Fallback to regular
        angles = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
        return np.stack([np.cos(angles), np.sin(angles)], axis=1)

def equidistant_points_polygon20() -> np.ndarray:
    """
    Build a convex polygon with n = 20 vertices.
    Returns:
        polygon: np.ndarray of shape (20, 2) with the x,y coordinates of the vertices,
            given in convex (counter-clockwise) order.
    """
    start_time = time.time()
    
    polygon = _search_angular()
    
    if time.time() - start_time > 625:
        n = 20
        radius = 1.0
        angles = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
        x = radius * np.cos(angles)
        y = radius * np.sin(angles)
        polygon = np.stack([x, y], axis=1)
        
    return polygon

# EVOLVE-BLOCK-END
