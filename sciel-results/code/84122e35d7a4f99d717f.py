import numpy as np
import time
from scipy.optimize import differential_evolution, minimize

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
    if np.any(np.linalg.norm(edges, axis=1) < 1e-10):
        return 0.0
        
    cross = edges[:, 0] * edges[:, 1] - edges[:, 1] * edges[:, 0]
    if np.any(np.abs(cross) < 1e-10):
        return 0.0
    signs = np.sign(cross)
    if np.any(signs != signs[0]):
        return 0.0
        
    min_score = 1.0
    for i in range(n):
        rotated = np.roll(normalized, -i, axis=0)
        v_score = _compute_vertex_score(rotated)
        if v_score < min_score:
            min_score = v_score
    return min_score

def _make_polygon_from_radii(radii):
    """
    Generate a polygon from 20 radii.
    radii: (20,) array of radii for angles 0, 2pi/20, ...
    """
    n = 20
    theta = np.linspace(0, 2 * np.pi, n, endpoint=False)
    x = radii * np.cos(theta)
    y = radii * np.sin(theta)
    return np.stack([x, y], axis=1)

def _objective_radii(radii):
    """
    Objective function for radius optimization.
    Maximizes min vertex score.
    Penalties for non-convexity.
    """
    try:
        # Ensure radii are positive and reasonable
        r = np.clip(radii, 0.5, 2.0)
        poly = _make_polygon_from_radii(r)
        
        # Check convexity explicitly for penalty
        centroid = np.mean(poly, axis=0)
        centered = poly - centroid
        edges = np.diff(np.vstack([centered, centered[:1]]), axis=0)
        cross = edges[:, 0] * edges[:, 1] - edges[:, 1] * edges[:, 0]
        
        # Penalty if not strictly convex
        if np.any(np.abs(cross) < 1e-8):
            return 1.0 + np.sum(np.abs(cross)) # Large penalty
        signs = np.sign(cross)
        if np.any(signs != signs[0]):
            return 1.0 + np.sum(np.abs(cross))
            
        score = _polygon_min_score(poly)
        return -score
    except Exception:
        return 1.0

def equidistant_points_polygon20() -> np.ndarray:
    """
    Build a convex polygon with n = 20 vertices.
    Returns:
        polygon: np.ndarray of shape (20, 2) with the x,y coordinates of the vertices,
            given in convex (counter-clockwise) order.
    """
    start_time = time.time()
    deadline = start_time + 600.0
    n = 20
    
    # Initial best: Regular polygon (radii all 1)
    best_radii = np.ones(n)
    best_poly = _make_polygon_from_radii(best_radii)
    best_score = _polygon_min_score(best_poly)
    
    # Phase 1: Global Search using Differential Evolution on 20D radii space
    # Bounds: [0.5, 2.0] to allow significant shape variation while keeping center of mass reasonable
    bounds = [(0.5, 2.0)] * n
    
    try:
        result_de = differential_evolution(
            _objective_radii, 
            bounds, 
            maxiter=150, 
            popsize=40, 
            tol=1e-8,
            seed=42,
            polish=False,
            init='sobol'
        )
        
        if result_de.fun < -best_score:
            candidate_radii = np.clip(result_de.x, 0.5, 2.0)
            candidate_poly = _make_polygon_from_radii(candidate_radii)
            score = _polygon_min_score(candidate_poly)
            if score > best_score:
                best_score = score
                best_poly = candidate_poly
                best_radii = candidate_radii
                
    except Exception as e:
        print(f"DE failed: {e}")
        
    if time.time() > deadline - 120.0:
        return best_poly

    # Phase 2: Local Optimization using Nelder-Mead on radii
    x0 = best_radii.copy()
    
    try:
        options = {
            'maxiter': 3000,
            'xatol': 1e-10,
            'fatol': 1e-12,
            'adaptive': True
        }
        
        result_nm = minimize(
            _objective_radii, 
            x0, 
            method='Nelder-Mead', 
            options=options
        )
        
        if result_nm.fun < -best_score:
            candidate_radii = np.clip(result_nm.x, 0.5, 2.0)
            candidate_poly = _make_polygon_from_radii(candidate_radii)
            score = _polygon_min_score(candidate_poly)
            if score > best_score:
                best_score = score
                best_poly = candidate_poly
                best_radii = candidate_radii
                
    except Exception as e:
        print(f"NM failed: {e}")

    # Phase 3: Multi-start local refinement
    if time.time() < deadline - 60.0:
        for restart in range(10):
            if time.time() > deadline - 60.0:
                break
            
            # Perturb best_radii
            perturbation_scale = 0.1 * (1.0 / (restart + 1))
            x0_perturbed = best_radii + np.random.normal(0, perturbation_scale, size=n)
            x0_perturbed = np.clip(x0_perturbed, 0.5, 2.0)
            
            try:
                options = {
                    'maxiter': 1000,
                    'xatol': 1e-10,
                    'fatol': 1e-12,
                    'adaptive': True
                }
                result_nm = minimize(
                    _objective_radii, 
                    x0_perturbed, 
                    method='Nelder-Mead', 
                    options=options
                )
                
                if result_nm.fun < -best_score:
                    candidate_radii = np.clip(result_nm.x, 0.5, 2.0)
                    candidate_poly = _make_polygon_from_radii(candidate_radii)
                    score = _polygon_min_score(candidate_poly)
                    if score > best_score:
                        best_score = score
                        best_poly = candidate_poly
                        best_radii = candidate_radii
                        
            except Exception:
                pass

    return best_poly
