# EVOLVE-BLOCK-START
import numpy as np
import time
import scipy.optimize as opt

def _compute_vertex_score(vertices):
    """
    Compute the per-vertex score for a single vertex of a polygon.
    vertices: (n, 2) array of polygon vertices in cyclic order.
    Returns the score for vertex 0 (we rotate so the target vertex is index 0).
    """
    n = len(vertices)
    # Distances from vertex 0 to all others
    diffs = vertices - vertices[0]
    dists = np.sqrt(np.sum(diffs ** 2, axis=1))
    # Exclude self
    other_dists = dists[1:]
    # Sort
    d = np.sort(other_dists)
    # Check min distance
    if d[0] < 0.05:
        return 0.0
    # Compute gaps
    gaps = d[1:] - d[:-1]
    # Find first index minimizing the gap d_{k+1} - d_k
    # k is index in d, so we look at gaps[k] = d[k+1] - d[k]
    k = np.argmin(gaps)
    D = (d[k] + d[k+1]) / 2.0
    if D < 0.1:
        return 0.0
    # Select four distances
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
    # Normalize: subtract centroid, divide by mean distance to centroid
    centroid = np.mean(polygon, axis=0)
    centered = polygon - centroid
    dists_to_centroid = np.sqrt(np.sum(centered ** 2, axis=1))
    mean_dist = np.mean(dists_to_centroid)
    if mean_dist <= 0.01:
        return 0.0
    normalized = centered / mean_dist
    
    # Check convexity
    # No repeated points
    if np.any(np.linalg.norm(np.diff(np.vstack([normalized, normalized[:1]]), axis=0), axis=1) < 1e-10):
        return 0.0
    # Compute cross products for turns
    edges = np.diff(np.vstack([normalized, normalized[:1]]), axis=0)
    cross = edges[:, 0] * edges[:, 1] - edges[:, 1] * edges[:, 0]
    # Check all turns are same sign and nonzero
    if np.any(np.abs(cross) < 1e-10):
        return 0.0
    # Check all same sign
    signs = np.sign(cross)
    if np.any(signs != signs[0]):
        return 0.0
    # Total turning should be +-2*pi
    total_turn = np.sum(np.arctan2(cross, edges[:, 0] * edges[:, 0] + edges[:, 1] * edges[:, 1]))
    if abs(abs(total_turn) - 2 * np.pi) > 1e-6:
        return 0.0
    
    # Compute min vertex score
    min_score = 1.0
    for i in range(n):
        # Rotate so vertex i is at index 0
        rotated = np.roll(normalized, -i, axis=0)
        v_score = _compute_vertex_score(rotated)
        if v_score < min_score:
            min_score = v_score
    return min_score

def _objective_cartesian(params):
    """
    Objective function for Cartesian coordinate optimization.
    params: (40,) array of 20 (x,y) pairs.
    Returns a cost (lower is better). We want to maximize min vertex score,
    so we minimize -min_score, with heavy penalty for invalid polygons.
    """
    n = 20
    polygon = params.reshape(n, 2)
    
    # Quick convexity check before expensive score computation
    # Compute cross products
    edges = np.diff(np.vstack([polygon, polygon[:1]]), axis=0)
    cross = edges[:, 0] * edges[:, 1] - edges[:, 1] * edges[:, 0]
    
    # Check if all cross products are nonzero and same sign
    if np.any(np.abs(cross) < 1e-8):
        return 100.0
    signs = np.sign(cross)
    if np.any(signs != signs[0]):
        return 100.0
    
    # Check total turning
    total_turn = np.sum(np.arctan2(cross, edges[:, 0] * edges[:, 0] + edges[:, 1] * edges[:, 1]))
    if abs(abs(total_turn) - 2 * np.pi) > 1e-5:
        return 100.0
    
    # Check no repeated points
    if np.any(np.linalg.norm(np.diff(np.vstack([polygon, polygon[:1]]), axis=0), axis=1) < 1e-8):
        return 100.0
    
    # Compute actual score
    score = _polygon_min_score(polygon)
    if score == 0.0:
        return 100.0
    
    return -score

def equidistant_points_polygon20() -> np.ndarray:
    """
    Build a convex polygon with n = 20 vertices.
    Returns:
        polygon: np.ndarray of shape (20, 2) with the x,y coordinates of the vertices,
            given in convex (counter-clockwise) order.
    """
    start_time = time.time()
    deadline = start_time + 620.0
    
    n = 20
    # Start with a regular polygon as initial guess
    angles = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    regular_poly = np.stack([np.cos(angles), np.sin(angles)], axis=1)
    best_polygon = regular_poly
    best_score = _polygon_min_score(regular_poly)
    
    # Stage 1: Differential Evolution in Cartesian space
    # Bounds: coordinates in [-2, 2]
    bounds = [(-2.0, 2.0)] * (2 * n)
    
    # Use DE with moderate population
    try:
        de_result = opt.differential_evolution(
            _objective_cartesian,
            bounds,
            seed=42,
            maxiter=30,
            popsize=15,
            tol=1e-4,
            polish=False,
            init='sobol'
        )
        
        if time.time() < deadline:
            candidate = de_result.x.reshape(n, 2)
            cand_score = _polygon_min_score(candidate)
            if cand_score > best_score:
                best_score = cand_score
                best_polygon = candidate
    except Exception:
        pass
    
    # Stage 2: Nelder-Mead refinement from the best candidate found
    if time.time() < deadline - 30:
        try:
            nm_result = opt.minimize(
                _objective_cartesian,
                best_polygon.flatten(),
                method='Nelder-Mead',
                options={'xatol': 1e-6, 'fatol': 1e-6, 'maxiter': 500}
            )
            
            if time.time() < deadline:
                candidate = nm_result.x.reshape(n, 2)
                cand_score = _polygon_min_score(candidate)
                if cand_score > best_score:
                    best_score = cand_score
                    best_polygon = candidate
        except Exception:
            pass
    
    # Stage 3: Try a few random restarts with Nelder-Mead if time permits
    rng = np.random.default_rng(123)
    attempts = 0
    while time.time() < deadline - 60 and attempts < 5:
        # Generate a random convex polygon
        # Random angles sorted, random radii
        theta = np.sort(rng.uniform(0, 2 * np.pi, n))
        r = rng.uniform(0.5, 1.5, n)
        poly = np.stack([r * np.cos(theta), r * np.sin(theta)], axis=1)
        
        # Ensure convexity by checking and adjusting if needed
        # For now, just try NM from this point
        try:
            nm_result = opt.minimize(
                _objective_cartesian,
                poly.flatten(),
                method='Nelder-Mead',
                options={'xatol': 1e-6, 'fatol': 1e-6, 'maxiter': 200}
            )
            
            if time.time() < deadline:
                candidate = nm_result.x.reshape(n, 2)
                cand_score = _polygon_min_score(candidate)
                if cand_score > best_score:
                    best_score = cand_score
                    best_polygon = candidate
        except Exception:
            pass
        attempts += 1
    
    # Final safety check: ensure the returned polygon is valid
    final_score = _polygon_min_score(best_polygon)
    if final_score <= 0.0:
        # Fallback to regular polygon
        best_polygon = regular_poly
    
    return best_polygon

# EVOLVE-BLOCK-END
