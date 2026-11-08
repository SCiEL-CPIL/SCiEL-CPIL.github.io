"""Heilbronn convex regions n=14: maximize min triangle area / hull area."""
import numpy as np
import time
import itertools

N = 14

def _min_triangle_area(pts):
    """Compute minimum triangle area among all triples of points."""
    n = len(pts)
    min_area = float('inf')
    for i in range(n):
        for j in range(i + 1, n):
            v1 = pts[j] - pts[i]
            for k in range(j + 1, n):
                v2 = pts[k] - pts[i]
                area = 0.5 * abs(v1[0] * v2[1] - v1[1] * v2[0])
                if area < min_area:
                    min_area = area
    return min_area

def _convex_hull_area(pts):
    """Compute area of convex hull using scipy."""
    from scipy.spatial import ConvexHull
    try:
        hull = ConvexHull(pts)
        return hull.volume  # In 2D, volume is area
    except Exception:
        # Fallback: use shapely-like computation or just return 0
        return 0.0

def _score(pts):
    """Score = min triangle area / convex hull area."""
    hull_area = _convex_hull_area(pts)
    if hull_area < 1e-12:
        return 0.0
    min_tri = _min_triangle_area(pts)
    return min_tri / hull_area

def _is_strictly_convex(pts):
    """Check if points form a strictly convex polygon (no three collinear, all turn same direction)."""
    n = len(pts)
    if n < 3:
        return False
    # Check all cross products have same sign
    signs = []
    for i in range(n):
        p0 = pts[i]
        p1 = pts[(i + 1) % n]
        p2 = pts[(i + 2) % n]
        cross = (p1[0] - p0[0]) * (p2[1] - p0[1]) - (p1[1] - p0[1]) * (p2[0] - p0[0])
        if abs(cross) < 1e-12:
            return False
        signs.append(cross > 0)
    # All must be same sign
    return all(signs) or all(not s for s in signs)

def _regular_polygon(n, radius=1.0):
    """Generate regular n-gon centered at origin."""
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    return np.column_stack([radius * np.cos(angles), radius * np.sin(angles)])

def _perturb_polygon(base_pts, angles_pert, radii_pert):
    """Apply angular and radial perturbations to a regular polygon."""
    n = len(base_pts)
    base_angles = np.arctan2(base_pts[:, 1], base_pts[:, 0])
    base_radii = np.sqrt(base_pts[:, 0]**2 + base_pts[:, 1]**2)
    
    new_angles = base_angles + angles_pert
    new_radii = base_radii + radii_pert
    
    # Ensure radii are positive
    new_radii = np.maximum(new_radii, 0.01)
    
    pts = np.column_stack([new_radii * np.cos(new_angles), new_radii * np.sin(new_angles)])
    return pts

def construct():
    start_time = time.time()
    time_limit = 115.0  # Leave 5s buffer for safety
    
    # Initialize with regular 14-gon
    best_pts = _regular_polygon(N, radius=1.0)
    best_score = _score(best_pts)
    
    # Nelder-Mead style optimization on perturbation parameters
    # We parameterize as angular perturbations + radial perturbations
    # Dimension: 2*N = 28
    
    def objective(params):
        """Negative score (we minimize)."""
        angles_pert = params[:N]
        radii_pert = params[N:]
        
        # Constraint: keep radii positive
        radii_pert = np.maximum(radii_pert, -0.5)  # Allow some inward movement
        
        pts = _perturb_polygon(best_pts, angles_pert, radii_pert)
        
        # Check convexity
        if not _is_strictly_convex(pts):
            return 1e10  # Penalty
        
        s = _score(pts)
        return -s
    
    # Use scipy.optimize.minimize with Nelder-Mead
    from scipy.optimize import minimize
    
    # Initial perturbation: zero
    x0 = np.zeros(2 * N)
    
    # Multiple restarts with different initial perturbations
    rng = np.random.default_rng(42)
    
    n_restarts = 5
    for restart in range(n_restarts):
        if time.time() - start_time > time_limit - 5:
            break
        
        if restart == 0:
            x_init = np.zeros(2 * N)
        else:
            # Small random perturbations
            x_init = rng.normal(0, 0.1, size=2 * N)
        
        try:
            result = minimize(
                objective,
                x_init,
                method='Nelder-Mead',
                options={
                    'maxiter': 5000,
                    'maxfev': 10000,
                    'xatol': 1e-8,
                    'fatol': 1e-10,
                    'adaptive': True
                }
            )
            
            if result.fun < 1e9:  # Valid result
                angles_pert = result.x[:N]
                radii_pert = result.x[N:]
                radii_pert = np.maximum(radii_pert, -0.5)
                
                candidate_pts = _perturb_polygon(best_pts, angles_pert, radii_pert)
                
                if _is_strictly_convex(candidate_pts):
                    candidate_score = _score(candidate_pts)
                    if candidate_score > best_score:
                        best_score = candidate_score
                        best_pts = candidate_pts
        except Exception:
            continue
    
    # Final polish: try to improve by moving points inward while maintaining min triangle area
    # This is a heuristic: scale points toward centroid if it improves the ratio
    
    # Centroid
    centroid = np.mean(best_pts, axis=0)
    
    # Try different scaling factors
    for scale in [0.95, 0.9, 0.85, 0.8, 0.75, 0.7, 0.65, 0.6, 0.55, 0.5]:
        if time.time() - start_time > time_limit - 2:
            break
        
        scaled_pts = centroid + scale * (best_pts - centroid)
        
        if _is_strictly_convex(scaled_pts):
            s = _score(scaled_pts)
            if s > best_score:
                best_score = s
                best_pts = scaled_pts
    
    return best_pts
