import numpy as np
import time
from scipy.spatial import ConvexHull
from scipy.optimize import minimize

def _triangle_areas(points):
    n = len(points)
    indices = []
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                indices.append((i, j, k))
    indices = np.array(indices)
    p1 = points[indices[:, 0]]
    p2 = points[indices[:, 1]]
    p3 = points[indices[:, 2]]
    areas = 0.5 * np.abs(
        p1[:, 0] * (p2[:, 1] - p3[:, 1]) +
        p2[:, 0] * (p3[:, 1] - p1[:, 1]) +
        p3[:, 0] * (p1[:, 1] - p2[:, 1])
    )
    return areas

def _score(points):
    try:
        if np.any(~np.isfinite(points)):
            return 0.0
        if len(points) != 13 or points.shape != (13, 2):
            return 0.0
        hull = ConvexHull(points)
        a_hull = hull.volume
        if a_hull <= 0:
            return 0.0
        areas = _triangle_areas(points)
        a_min = np.min(areas)
        if a_min <= 0:
            return 0.0
        return a_min / a_hull
    except Exception:
        return 0.0

def _generate_hexagonal_lattice():
    """Generate a hexagonal lattice configuration with 13 points.
    
    This creates a central point surrounded by two hexagonal rings,
    which is known to be a strong starting configuration for the
    Heilbronn problem.
    """
    pts = np.zeros((13, 2))
    pts[0] = [0, 0]
    
    # Inner hexagon (6 points)
    r1 = 1.0
    for i in range(6):
        angle = i * np.pi / 3.0
        pts[i+1] = [r1 * np.cos(angle), r1 * np.sin(angle)]
    
    # Outer hexagon (6 points), rotated by 30 degrees
    r2 = 1.8
    theta_offset = np.pi / 6.0
    for i in range(6):
        angle = i * np.pi / 3.0 + theta_offset
        pts[i+7] = [r2 * np.cos(angle), r2 * np.sin(angle)]
    
    return pts

def _generate_regular_polygon(n):
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    return np.column_stack((np.cos(angles), np.sin(angles)))

def _direct_triangle_optimization(points, deadline, max_iter=2000):
    """
    Direct optimization targeting the minimum-area triangle.
    
    Strategy:
    1. Find the minimum-area triangle
    2. Move its vertices to increase that triangle's area
    3. Pull convex hull points inward to reduce hull area
    4. Accept moves that improve the overall score
    """
    pts = points.copy()
    best_score = _score(pts)
    if best_score <= 0:
        return pts, best_score
    
    # Precompute triangle indices
    tri_indices = []
    for i in range(13):
        for j in range(i + 1, 13):
            for k in range(j + 1, 13):
                tri_indices.append((i, j, k))
    
    # Adaptive step size
    base_step = 0.02
    min_step = 0.001
    step_scale = 1.0
    
    for iteration in range(max_iter):
        if time.time() > deadline:
            break
        
        # Find the minimum area triangle
        areas = _triangle_areas(pts)
        min_idx = np.argmin(areas)
        i, j, k = tri_indices[min_idx]
        
        # Get the three points
        p_i = pts[i]
        p_j = pts[j]
        p_k = pts[k]
        
        # Calculate vectors
        v1 = p_j - p_i
        v2 = p_k - p_i
        cross = v1[0] * v2[1] - v1[1] * v2[0]
        
        if abs(cross) < 1e-10:
            # Nearly collinear, use random perturbation
            np.random.seed(iteration)
            for idx in [i, j, k]:
                pts[idx] += 0.01 * np.random.normal(0, 1, 2)
            continue
        
        # Direction to increase area: perpendicular to the triangle
        # For triangle (p_i, p_j, p_k), moving p_i opposite to the cross product
        # direction increases area, as do p_j and p_k
        norm = np.linalg.norm(v2)
        if norm > 1e-10:
            perp = np.array([v2[1], -v2[0]]) / norm
        else:
            perp = np.array([0, 1])
        
        step = base_step * step_scale
        
        # Try moving the three vertices to increase the triangle area
        # p_i moves opposite to perp, p_j and p_k move along perp
        trial_pts = pts.copy()
        trial_pts[i] = p_i - step * perp
        trial_pts[j] = p_j + step * perp
        trial_pts[k] = p_k - step * perp
        
        # Also pull convex hull points inward
        try:
            hull = ConvexHull(trial_pts)
            hull_indices = set(hull.vertices)
        except:
            hull_indices = set(range(13))
        
        all_centroid = np.mean(trial_pts, axis=0)
        for idx in hull_indices:
            if idx not in [i, j, k]:
                # Move slightly toward center to reduce hull area
                trial_pts[idx] = 0.99 * trial_pts[idx] + 0.01 * all_centroid
        
        new_score = _score(trial_pts)
        
        if new_score > best_score:
            pts = trial_pts
            best_score = new_score
            # Increase step slightly on success
            step_scale = min(1.5, step_scale * 1.1)
        else:
            # Reduce step size on failure
            step_scale = max(0.7, step_scale * 0.9)
            if step_scale < 0.1:
                break
    
    return pts, best_score

def _nelder_mead_refinement(points, deadline):
    """
    Refine using Nelder-Mead with a focus on the critical triangle.
    """
    def _obj(x):
        pts = x.reshape(13, 2)
        return -_score(pts)
    
    x_start = points.flatten()
    
    try:
        res = minimize(_obj, x_start, method='Nelder-Mead',
                       options={'xatol': 1e-7, 'fatol': 1e-7, 
                               'maxiter': 500, 'maxfev': 1000})
        
        if res.fun < 0:
            pts_opt = res.x.reshape(13, 2)
            score_opt = _score(pts_opt)
            if score_opt > 0:
                return pts_opt, score_opt
    except Exception:
        pass
    
    return points, _score(points)

def heilbronn_convex13() -> np.ndarray:
    deadline = time.time() + 590  # Safety margin: 630s limit, 40s buffer
    
    # Generate multiple candidate initial configurations
    candidates = []
    
    # Primary: Hexagonal lattice (known strong configuration)
    candidates.append(_generate_hexagonal_lattice())
    
    # Secondary: Regular 13-gon
    candidates.append(_generate_regular_polygon(13))
    
    # Tertiary: Hexagonal lattice with different parameters
    pts_alt = np.zeros((13, 2))
    pts_alt[0] = [0, 0]
    r1 = 0.9
    for i in range(6):
        angle = i * np.pi / 3.0
        pts_alt[i+1] = [r1 * np.cos(angle), r1 * np.sin(angle)]
    r2 = 1.7
    theta_offset = np.pi / 5.0
    for i in range(6):
        angle = i * np.pi / 3.0 + theta_offset
        pts_alt[i+7] = [r2 * np.cos(angle), r2 * np.sin(angle)]
    candidates.append(pts_alt)
    
    # Find best initial candidate
    best_pts = candidates[0]
    best_score = _score(best_pts)
    for cand in candidates[1:]:
        s = _score(cand)
        if s > best_score:
            best_score = s
            best_pts = cand
    
    # Phase 1: Direct triangle optimization on the best candidate
    if time.time() < deadline - 100:
        pts_opt, score_opt = _direct_triangle_optimization(best_pts, deadline - 100, max_iter=3000)
        if score_opt > best_score:
            best_score = score_opt
            best_pts = pts_opt
    
    # Phase 2: Nelder-Mead refinement
    if time.time() < deadline - 50:
        pts_nm, score_nm = _nelder_mead_refinement(best_pts, deadline - 50)
        if score_nm > best_score:
            best_score = score_nm
            best_pts = pts_nm
    
    # Phase 3: Final direct optimization pass with smaller steps
    if time.time() < deadline - 20:
        pts_final, score_final = _direct_triangle_optimization(best_pts, deadline - 20, max_iter=2000)
        if score_final > best_score:
            best_score = score_final
            best_pts = pts_final
    
    # Phase 4: One more Nelder-Mead pass if time permits
    if time.time() < deadline - 10:
        pts_final2, score_final2 = _nelder_mead_refinement(best_pts, deadline - 10)
        if score_final2 > best_score:
            best_score = score_final2
            best_pts = pts_final2
    
    # Fallback: if something went wrong, return a valid configuration
    if _score(best_pts) <= 0:
        best_pts = _generate_hexagonal_lattice()
    
    return best_pts
