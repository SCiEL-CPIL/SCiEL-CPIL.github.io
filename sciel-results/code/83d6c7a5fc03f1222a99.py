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

def _make_symmetric_polygon(epsilon, freq=1, n=20):
    """
    Create a polygon with radial perturbation r_i = 1 + epsilon * cos(freq * theta_i).
    This maintains rotational symmetry (if freq divides n) or specific symmetries.
    """
    angles = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    radii = 1.0 + epsilon * np.cos(freq * angles)
    x = radii * np.cos(angles)
    y = radii * np.sin(angles)
    return np.stack([x, y], axis=1)

def _search_analytic():
    """
    Search for optimal epsilon and frequency using deterministic optimization.
    """
    start_time = time.time()
    deadline = start_time + 620.0
    
    best_score = 0.0
    best_polygon = None
    
    # Regular polygon as baseline
    regular_poly = _make_symmetric_polygon(0.0, 1)
    best_score = _polygon_min_score(regular_poly)
    best_polygon = regular_poly
    
    # Try different frequencies (harmonics)
    # Frequencies that preserve convexity and symmetry are often divisors of n or small integers
    # For n=20, freq=1, 2, 3, 4, 5, 10 are candidates.
    # freq=1 is a "dipole" (breaks rotational symmetry, keeps reflection).
    # freq=2 is a "quadrupole" etc.
    
    candidate_freqs = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    
    for freq in candidate_freqs:
        if time.time() > deadline:
            break
            
        def objective(eps):
            poly = _make_symmetric_polygon(eps, freq)
            score = _polygon_min_score(poly)
            # We want to maximize score, so minimize -score
            # Add penalty if score is 0 (invalid or bad)
            if score == 0.0:
                return 10.0
            return -score
        
        # Search range for epsilon
        # Too small: regular polygon (score ~1.0)
        # Too large: convexity might break or distances become bad
        # Let's try -0.5 to 0.5
        bounds = (-0.5, 0.5)
        
        try:
            # Use bounded minimization
            res = opt.minimize_scalar(objective, bounds=bounds, method='bounded', 
                                      options={'xatol': 1e-6, 'maxiter': 100})
            
            if res.success:
                eps_opt = res.x
                poly_opt = _make_symmetric_polygon(eps_opt, freq)
                score_opt = _polygon_min_score(poly_opt)
                
                if score_opt > best_score:
                    best_score = score_opt
                    best_polygon = poly_opt
                    
        except Exception:
            pass
            
    # Also try a 2D search for freq=1 and freq=2 with a second harmonic?
    # r_i = 1 + eps1*cos(1*theta) + eps2*cos(2*theta)
    # This is more complex but might find better local optima.
    
    def objective_2d(params):
        eps1, eps2 = params
        angles = np.linspace(0.0, 2.0 * np.pi, 20, endpoint=False)
        radii = 1.0 + eps1 * np.cos(angles) + eps2 * np.cos(2 * angles)
        # Ensure convexity: radii must be positive and not vary too wildly
        if np.any(radii < 0.1):
            return 10.0
        x = radii * np.cos(angles)
        y = radii * np.sin(angles)
        poly = np.stack([x, y], axis=1)
        score = _polygon_min_score(poly)
        if score == 0.0:
            return 10.0
        return -score

    # Nelder-Mead for 2D search
    initial_guesses = [
        [0.1, 0.1],
        [0.1, -0.1],
        [-0.1, 0.1],
        [-0.1, -0.1],
        [0.05, 0.05],
        [0.2, 0.0],
        [0.0, 0.2]
    ]
    
    for guess in initial_guesses:
        if time.time() > deadline:
            break
        try:
            res = opt.minimize(objective_2d, guess, method='Nelder-Mead', 
                               options={'xatol': 1e-5, 'fatol': 1e-5, 'maxiter': 200})
            if res.success:
                eps1_opt, eps2_opt = res.x
                angles = np.linspace(0.0, 2.0 * np.pi, 20, endpoint=False)
                radii = 1.0 + eps1_opt * np.cos(angles) + eps2_opt * np.cos(2 * angles)
                if np.any(radii < 0.1):
                    continue
                x = radii * np.cos(angles)
                y = radii * np.sin(angles)
                poly_opt = np.stack([x, y], axis=1)
                score_opt = _polygon_min_score(poly_opt)
                if score_opt > best_score:
                    best_score = score_opt
                    best_polygon = poly_opt
        except Exception:
            pass

    return best_polygon

def equidistant_points_polygon20() -> np.ndarray:
    """
    Build a convex polygon with n = 20 vertices.
    Returns:
        polygon: np.ndarray of shape (20, 2) with the x,y coordinates of the vertices,
            given in convex (counter-clockwise) order.
    """
    start_time = time.time()
    
    # Use analytic search
    polygon = _search_analytic()
    
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

# EVOLVE-BLOCK-END
