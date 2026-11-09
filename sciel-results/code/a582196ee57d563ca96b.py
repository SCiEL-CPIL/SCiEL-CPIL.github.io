import numpy as np
import time
from scipy.spatial.distance import pdist
from scipy.optimize import minimize
import itertools


def _evaluate(points: np.ndarray) -> float:
    """Compute the metric (d_min / d_max)^2 for a set of points."""
    if len(points) < 2:
        return 0.0
    dists = pdist(points)
    d_min = np.min(dists)
    d_max = np.max(dists)
    if d_max == 0 or not np.isfinite(d_min) or not np.isfinite(d_max):
        return 0.0
    ratio = d_min / d_max
    return ratio * ratio


def _icosahedron_vertices() -> np.ndarray:
    """Return the 12 vertices of a regular icosahedron on the unit sphere."""
    phi = (1 + np.sqrt(5)) / 2
    vertices = np.array([
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [-phi, 0, 1], [phi, 0, -1], [-phi, 0, -1]
    ], dtype=float)
    norms = np.linalg.norm(vertices, axis=1, keepdims=True)
    return vertices / norms


def _normalize(points: np.ndarray) -> np.ndarray:
    """Normalize points to unit sphere."""
    norms = np.linalg.norm(points, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return points / norms


def _nelder_mead_exact(points: np.ndarray, max_iter=1000, tol=1e-10) -> np.ndarray:
    """
    Refine points using Nelder-Mead simplex on the exact metric.
    We minimize the negative metric.
    """
    def objective(params):
        pts = params.reshape(14, 3)
        # Optional: keep on sphere to avoid scale drift? 
        # The metric is scale invariant, but NM might drift.
        # Let's project to sphere inside objective to constrain search space effectively.
        # Actually, projecting inside objective makes it non-smooth. 
        # Better: optimize in R^42 and just evaluate. The scale invariance means 
        # the objective is flat along the radial direction. NM might struggle.
        # Let's fix the scale by normalizing in the evaluation, but optimizing in R^42.
        # Wait, if I normalize in the objective, the gradient is weird.
        
        # Alternative: Optimize only 2 parameters for the 2 extra points, 
        # and fix the icosahedron? No, the icosahedron might deform.
        
        # Let's try optimizing all 42 coordinates.
        # To handle scale invariance, we can add a small penalty for deviation from unit norm.
        pts_norm = _normalize(pts)
        metric = _evaluate(pts_norm)
        # Penalty to keep points near unit sphere (helps NM stay in a good basin)
        norms = np.linalg.norm(pts, axis=1)
        penalty = 0.1 * np.sum((norms - 1.0)**2)
        return -metric + penalty

    try:
        result = minimize(
            objective,
            points.flatten(),
            method='Nelder-Mead',
            options={'maxiter': max_iter, 'xatol': tol, 'fatol': tol, 'adaptive': True}
        )
        return result.x.reshape(14, 3)
    except Exception:
        return points


def min_max_dist_dim3_14() -> np.ndarray:
    start_time = time.time()
    deadline = start_time + 580  # Safety margin

    ico = _icosahedron_vertices()
    
    # Strategy:
    # 1. Start with Icosahedron + 2 poles (standard high-symmetry config).
    # 2. Use Nelder-Mead to refine this exact configuration.
    # 3. Generate multiple symmetric variants (rotations) of the 2 poles and refine each.
    # 4. Also try random restarts from the best found so far.
    
    best_points = None
    best_metric = 0.0
    
    # Initial candidate: Ico + (0,0,1) + (0,0,-1)
    p1 = np.array([0, 0, 1])
    p2 = np.array([0, 0, -1])
    cand0 = np.vstack([ico, p1.reshape(1, -1), p2.reshape(1, -1)])
    cand0 = _normalize(cand0)
    
    # Refine the initial candidate
    refined = _nelder_mead_exact(cand0, max_iter=500)
    refined = _normalize(refined)
    metric = _evaluate(refined)
    
    if metric > best_metric:
        best_metric = metric
        best_points = refined.copy()
    
    # Generate symmetric variants of the pole positions
    # The icosahedron has 60 rotational symmetries. 
    # Instead of all 60, let's try a few key rotations of the poles.
    # Actually, the poles are on the 5-fold axis. Rotating by other axes moves them to other 5-fold axes.
    # There are 6 5-fold axes. Let's try poles on each of the 6 axes.
    
    # Define the 6 5-fold axes (normalized)
    axes = []
    # The icosahedron vertices come in opposite pairs. The axes through opposite vertices are 5-fold.
    # There are 6 such axes.
    # We can extract them from the icosahedron vertices.
    # Vertices are:
    # (0, 1, phi), (0, -1, phi), (0, 1, -phi), (0, -1, -phi) -> 4 on y-z plane? No.
    # Let's just use the normalized icosahedron vertices as potential pole positions.
    # If we pick a vertex v, we can place the two new points at v and -v.
    # This preserves the icosahedral symmetry if v is a vertex? 
    # No, the icosahedron itself has 12 vertices. If we add v and -v, and v is already a vertex, we have a duplicate.
    # So we should NOT use existing vertices as the new poles.
    
    # The 5-fold axes pass through vertices. So placing poles at vertices creates duplicates.
    # The 3-fold axes pass through face centers.
    # The 2-fold axes pass through edge midpoints.
    
    # Let's try placing the 2 new points at:
    # 1. (0,0,1) and (0,0,-1) [Current]
    # 2. Face centers?
    # Let's generate face centers of the icosahedron.
    # An icosahedron has 20 faces.
    # We can compute face centers by averaging 3 vertices of a face.
    # We need to identify the faces.
    
    # Simpler: Use random restarts from the best configuration found so far.
    rng = np.random.RandomState(42)
    
    # Try a few random rotations of the poles
    for i in range(20):
        if time.time() > deadline - 30:
            break
            
        # Random rotation matrix
        # Generate random unit vector for axis
        axis = rng.randn(3)
        axis = axis / np.linalg.norm(axis)
        angle = rng.uniform(0, 2 * np.pi)
        
        # Rodrigues' rotation formula
        c = np.cos(angle)
        s = np.sin(angle)
        k = axis
        K = np.array([
            [0, -k[2], k[1]],
            [k[2], 0, -k[0]],
            [-k[1], k[0], 0]
        ])
        R = np.eye(3) + s * K + (1 - c) * (K @ K)
        
        # Rotate the poles
        p1_rot = R @ p1
        p2_rot = R @ p2
        
        cand = np.vstack([ico, p1_rot.reshape(1, -1), p2_rot.reshape(1, -1)])
        cand = _normalize(cand)
        
        # Refine
        refined = _nelder_mead_exact(cand, max_iter=300)
        refined = _normalize(refined)
        metric = _evaluate(refined)
        
        if metric > best_metric:
            best_metric = metric
            best_points = refined.copy()
            
    # More aggressive random restarts from the best found
    if best_points is not None:
        for i in range(10):
            if time.time() > deadline - 10:
                break
            
            noise = rng.randn(14, 3) * 0.1
            cand = best_points + noise
            cand = _normalize(cand)
            
            refined = _nelder_mead_exact(cand, max_iter=300)
            refined = _normalize(refined)
            metric = _evaluate(refined)
            
            if metric > best_metric:
                best_metric = metric
                best_points = refined.copy()
                
    # Final polish with higher iteration count
    if best_points is not None and time.time() < deadline - 5:
        refined = _nelder_mead_exact(best_points, max_iter=1000, tol=1e-12)
        refined = _normalize(refined)
        metric = _evaluate(refined)
        if metric > best_metric:
            best_points = refined
            best_metric = metric

    # Fallback
    if best_points is None:
        best_points = cand0
        
    # Ensure finite
    if not np.all(np.isfinite(best_points)):
        best_points = cand0
        
    return best_points
