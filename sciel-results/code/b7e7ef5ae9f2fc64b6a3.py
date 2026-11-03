"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 13

# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial import ConvexHull
import time
import itertools

def construct():
    """
    Place 13 points to maximize the minimum triangle area divided by convex hull area.
    Uses a 1+12 ring symmetry parameterization with differential evolution.
    """
    deadline = time.time() + 110  # safety margin before 120s limit
    best_score = -1.0
    best_points = None
    
    # Precompute all triangle indices
    tri_indices = list(itertools.combinations(range(N), 3))
    tri_indices = np.array(tri_indices)
    
    def compute_score(points):
        """Compute min triangle area / convex hull area."""
        if points.shape[0] != N:
            return 0.0
        # Check for degenerate points
        if np.any(np.isnan(points)) or np.any(np.isinf(points)):
            return 0.0
        
        # Compute convex hull area
        try:
            hull = ConvexHull(points)
            hull_area = hull.volume  # In 2D, .volume is area
        except Exception:
            return 0.0
        
        if hull_area <= 1e-12:
            return 0.0
        
        # Compute all triangle areas
        # For each triangle (i, j, k), area = 0.5 * |cross(p_j - p_i, p_k - p_i)|
        pi = points[tri_indices[:, 0]]
        pj = points[tri_indices[:, 1]]
        pk = points[tri_indices[:, 2]]
        
        v1 = pj - pi
        v2 = pk - pi
        cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
        areas = 0.5 * np.abs(cross)
        
        min_area = np.min(areas)
        return min_area / hull_area
    
    def params_to_points(params):
        """
        Convert 13-dimensional parameter vector to 13 points.
        params[0] = center x, params[1] = center y
        params[2:14] = radial distances for 12 ring points at fixed angles
        
        The 12 ring points are at angles 0, pi/6, pi/3, ..., 11*pi/6
        (i.e., 30-degree spacing), with radii given by params[2:14].
        """
        cx, cy = params[0], params[1]
        radii = params[2:14]
        
        # Ensure positive radii
        radii = np.maximum(radii, 0.01)
        
        angles = np.linspace(0, 2 * np.pi, 12, endpoint=False)
        
        ring_x = radii * np.cos(angles)
        ring_y = radii * np.sin(angles)
        
        points = np.column_stack((cx + ring_x, cy + ring_y))
        return points
    
    # Initial best: regular 13-gon
    angles_init = np.linspace(0, 2 * np.pi, N, endpoint=False)
    points_init = np.column_stack((np.cos(angles_init), np.sin(angles_init)))
    best_score = compute_score(points_init)
    best_points = points_init.copy()
    
    # Try a few more initializations
    inits = []
    
    # Regular 13-gon
    inits.append(points_init.copy())
    
    # Fermat spiral
    i = np.arange(N)
    r = np.sqrt((i + 0.5) / N)
    theta = i * 2.399963229728653
    pts_fermat = np.column_stack((r * np.cos(theta), r * np.sin(theta)))
    pts_fermat -= np.mean(pts_fermat, axis=0)
    inits.append(pts_fermat)
    
    # Circle with center
    angles_c = np.linspace(0, 2 * np.pi, 12, endpoint=False)
    ring_c = np.column_stack((np.cos(angles_c), np.sin(angles_c)))
    pts_circle = np.vstack([np.array([[0.0, 0.0]]), ring_c])
    inits.append(pts_circle)
    
    for init_pts in inits:
        s = compute_score(init_pts)
        if s > best_score:
            best_score = s
            best_points = init_pts.copy()
    
    # Differential evolution on the symmetric parameter space
    # 13 parameters: cx, cy, r0..r11
    # Bounds: cx, cy in [-1, 1], radii in [0.05, 2.0]
    bounds = [(-1.0, 1.0), (-1.0, 1.0)] + [(0.05, 2.0)] * 12
    
    from scipy.optimize import differential_evolution
    
    def objective(params):
        points = params_to_points(params)
        return -compute_score(points)  # minimize negative score
    
    # Run differential evolution
    try:
        result = differential_evolution(
            objective,
            bounds=bounds,
            seed=42,
            maxiter=200,
            popsize=15,
            tol=1e-8,
            polish=False,
            callback=None
        )
        
        if result.success or result.fun < 0:
            pts_de = params_to_points(result.x)
            s_de = compute_score(pts_de)
            if s_de > best_score:
                best_score = s_de
                best_points = pts_de.copy()
    except Exception:
        pass
    
    # Check time
    if time.time() > deadline - 30:
        return best_points
    
    # Local refinement using Nelder-Mead on the best parameters
    # Find params that correspond to best_points
    # For the symmetric parameterization, we need to invert
    # Let's just do a local search in full 26D space from best_points
    
    from scipy.optimize import minimize
    
    def neg_score_full(pts_flat):
        pts = pts_flat.reshape(N, 2)
        return -compute_score(pts)
    
    x0 = best_points.flatten()
    
    try:
        result_nm = minimize(
            neg_score_full,
            x0,
            method='Nelder-Mead',
            options={'maxiter': 5000, 'xatol': 1e-10, 'fatol': 1e-12}
        )
        
        if result_nm.fun < 0:
            pts_nm = result_nm.x.reshape(N, 2)
            s_nm = compute_score(pts_nm)
            if s_nm > best_score:
                best_score = s_nm
                best_points = pts_nm.copy()
    except Exception:
        pass
    
    # Check time again
    if time.time() > deadline - 15:
        return best_points
    
    # Try a few more local refinements from different starting points
    for init_pts in inits:
        if time.time() > deadline - 10:
            break
        try:
            x0 = init_pts.flatten()
            result_nm2 = minimize(
                neg_score_full,
                x0,
                method='Nelder-Mead',
                options={'maxiter': 3000, 'xatol': 1e-10, 'fatol': 1e-12}
            )
            if result_nm2.fun < 0:
                pts_nm2 = result_nm2.x.reshape(N, 2)
                s_nm2 = compute_score(pts_nm2)
                if s_nm2 > best_score:
                    best_score = s_nm2
                    best_points = pts_nm2.copy()
        except Exception:
            pass
    
    return best_points
# EVOLVE-BLOCK-END
