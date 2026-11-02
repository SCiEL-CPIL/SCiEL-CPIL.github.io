"""
Heilbronn convex regions n=14: maximize min triangle area / convex hull area.

Strategy:
  1. Structural representation: closed cubic B-spline with 7 control points (14D).
     Points are sampled at equal arc-length intervals along the curve.
     This ensures a smooth, strictly convex boundary and allows non-uniform point distribution.
  2. Vectorized triangle-area computation for speed.
  3. Multi-phase optimization:
     a. Differential evolution on B-spline control points for global exploration.
     b. Local refinement using Nelder-Mead on Cartesian coordinates.
     c. "Hull-pull" structural move to enlarge convex hull area if points are interior.
  4. Time-guarded to finish well within 150s.
"""
import time
import numpy as np
from scipy.optimize import minimize, differential_evolution
from scipy.spatial import ConvexHull
from scipy.interpolate import CubicSpline
import itertools

N = 14

_START_TIME = time.time()
TIME_LIMIT = 140.0  # safety margin before 150s kill

def _time_remaining():
    return TIME_LIMIT - (time.time() - _START_TIME)

# Pre-compute all triangle index combinations
tri_indices = np.array(list(itertools.combinations(range(N), 3)))  # shape (364, 3)


def _min_triangle_area(points):
    """Minimum triangle area over all triples, vectorized."""
    p = points
    pi = p[tri_indices[:, 0]]
    pj = p[tri_indices[:, 1]]
    pk = p[tri_indices[:, 2]]
    cross = (pj[:, 0] - pi[:, 0]) * (pk[:, 1] - pi[:, 1]) - (pj[:, 1] - pi[:, 1]) * (pk[:, 0] - pi[:, 0])
    areas = 0.5 * np.abs(cross)
    return np.min(areas)


def _hull_area(points):
    try:
        hull = ConvexHull(points)
        return hull.volume
    except Exception:
        return 0.0


def _score(points):
    hull_area = _hull_area(points)
    if hull_area < 1e-10:
        return 0.0
    min_area = _min_triangle_area(points)
    if min_area < 1e-10:
        return 0.0
    return min_area / hull_area


def _objective_hard(flat):
    pts = flat.reshape(N, 2)
    if np.any(~np.isfinite(pts)):
        return 1e10
    min_area = _min_triangle_area(pts)
    if min_area < 1e-12:
        return 1e10
    hull_area = _hull_area(pts)
    if hull_area < 1e-10:
        return 1e10
    return hull_area / min_area


# ---------- B-spline convex boundary parameterization ----------
def _decode_bspline(ctrl_params):
    """
    Decode 14-dim control point parameters to 14 points on a B-spline curve.
    ctrl_params: shape (14,) = [x0,y0, ..., x6,y6]
    """
    ctrl = ctrl_params.reshape(7, 2)
    
    # Center the control points to improve numerical stability
    ctrl_centered = ctrl - ctrl.mean(axis=0)
    
    try:
        # Create a closed cubic B-spline
        # For a closed curve with 7 control points, we need to handle periodicity
        # We'll sample many points and then pick 14 at equal arc-length
        
        # Create spline for each coordinate
        # For a closed B-spline, we repeat the first control point at the end
        ctrl_closed = np.vstack([ctrl_centered, ctrl_centered[0]])
        
        # Use parameter values 0..7 for 8 points (7 unique + repeat)
        t = np.arange(8)
        
        # Create cubic splines for x and y
        cs_x = CubicSpline(t, ctrl_closed[:, 0], bc_type='periodic')
        cs_y = CubicSpline(t, ctrl_closed[:, 1], bc_type='periodic')
        
        # Sample many points along the curve
        t_fine = np.linspace(0, 7, 2000)
        x_fine = cs_x(t_fine)
        y_fine = cs_y(t_fine)
        
        # Compute arc length
        dx = np.diff(x_fine)
        dy = np.diff(y_fine)
        ds = np.sqrt(dx**2 + dy**2)
        s = np.concatenate([[0], np.cumsum(ds)])
        
        # Total arc length
        total_s = s[-1]
        
        # Sample 14 points at equal arc-length intervals
        s_targets = np.linspace(0, total_s, N, endpoint=False)
        
        # Interpolate to get points at target arc lengths
        x_pts = np.interp(s_targets, s, x_fine)
        y_pts = np.interp(s_targets, s, y_fine)
        
        pts = np.column_stack([x_pts, y_pts])
        
        return pts
        
    except Exception:
        # Fallback: just use the control points plus their antipodes
        # This gives 14 points (7 + 7)
        pts = np.vstack([ctrl_centered, -ctrl_centered[::-1]])
        return pts


def _objective_bspline(ctrl_params):
    pts = _decode_bspline(ctrl_params)
    return _objective_hard(pts.flatten())


def _generate_bspline_inits():
    """Generate diverse initial control point configurations for B-spline."""
    inits = []
    rng = np.random.default_rng(42)
    
    # 1. Regular heptagon control points
    angles = np.linspace(0, 2 * np.pi, 7, endpoint=False)
    ctrl = np.column_stack([np.cos(angles), np.sin(angles)])
    inits.append(ctrl.flatten().copy())
    
    # 2. Elliptical control points
    for rx, ry in [(1.2, 0.8), (0.8, 1.2), (1.3, 0.7), (0.7, 1.3), (1.1, 0.9)]:
        angles = np.linspace(0, 2 * np.pi, 7, endpoint=False)
        ctrl = np.column_stack([rx * np.cos(angles), ry * np.sin(angles)])
        inits.append(ctrl.flatten().copy())
    
    # 3. Perturbed regular heptagon
    for js in [0.05, 0.1, 0.15, 0.2]:
        angles = np.linspace(0, 2 * np.pi, 7, endpoint=False)
        ctrl = np.column_stack([np.cos(angles), np.sin(angles)])
        ctrl += rng.normal(0, js, size=(7, 2))
        inits.append(ctrl.flatten().copy())
    
    # 4. Random convex-ish polygons
    for seed in range(15):
        rs = np.random.default_rng(seed * 13 + 7)
        angles = np.sort(rs.uniform(0, 2 * np.pi, 7))
        radii = rs.uniform(0.6, 1.4, 7)
        ctrl = np.column_stack([radii * np.cos(angles), radii * np.sin(angles)])
        inits.append(ctrl.flatten().copy())
    
    # 5. Larger perturbations
    for seed in range(10):
        rs = np.random.default_rng(seed * 31 + 17)
        angles = np.sort(rs.uniform(0, 2 * np.pi, 7))
        radii = rs.uniform(0.4, 1.6, 7)
        ctrl = np.column_stack([radii * np.cos(angles), radii * np.sin(angles)])
        inits.append(ctrl.flatten().copy())
        
    return inits


def _refine_cartesian(pts):
    """Refine points in Cartesian space using Nelder-Mead."""
    if _time_remaining() < 10.0:
        return pts
    
    def obj(flat):
        p = flat.reshape(N, 2)
        if np.any(~np.isfinite(p)):
            return 1e10
        return _objective_hard(flat)
    
    try:
        res = minimize(obj, pts.flatten(), method='Nelder-Mead', 
                       options={'maxiter': 500, 'xatol': 1e-8, 'fatol': 1e-8})
        if res.fun < _objective_hard(pts.flatten()):
            return res.x.reshape(N, 2)
    except Exception:
        pass
    return pts


def _hull_pull(pts):
    """
    Structural move: pull interior points towards the convex hull boundary
    to potentially increase the hull area relative to min triangle area.
    This is a heuristic to escape local minima where points are too clustered.
    """
    if _time_remaining() < 10.0:
        return pts
    
    try:
        hull = ConvexHull(pts)
        hull_verts = pts[hull.vertices]
        
        # For each point, if it's not on the hull, pull it slightly towards
        # the nearest hull vertex or centroid of hull
        centroid = hull_verts.mean(axis=0)
        
        new_pts = pts.copy()
        for i in range(N):
            if i not in hull.vertices:
                # Pull towards centroid with small step
                step = 0.05
                new_pts[i] = pts[i] + step * (centroid - pts[i])
        
        # Check if this improved the score
        if _score(new_pts) > _score(pts):
            return new_pts
    except Exception:
        pass
    
    return pts


def construct():
    """Construct 14 points maximizing min triangle area / hull area."""
    
    # Phase 1: Differential Evolution on B-spline control points
    # Bounds for control points: roughly in [-2, 2]
    bounds = [(-2.0, 2.0)] * 14
    
    # Initial guess: regular heptagon
    angles = np.linspace(0, 2 * np.pi, 7, endpoint=False)
    best_params = np.column_stack([np.cos(angles), np.sin(angles)]).flatten()
    best_pts = _decode_bspline(best_params)
    best_score = _score(best_pts)
    
    # Run differential evolution
    if _time_remaining() > 30.0:
        try:
            result = differential_evolution(
                _objective_bspline,
                bounds=bounds,
                seed=42,
                maxiter=100,
                popsize=15,
                tol=1e-4,
                polish=False,  # We'll do our own polish
                init='sobol'
            )
            
            if result.fun < _objective_hard(best_params):
                best_params = result.x
                best_pts = _decode_bspline(best_params)
                best_score = _score(best_pts)
        except Exception:
            pass
    
    # Phase 2: Local refinement using multi-start from diverse inits
    inits = _generate_bspline_inits()
    
    for init in inits:
        if _time_remaining() < 15.0:
            break
        
        # Refine this init using Nelder-Mead on control points
        try:
            res = minimize(_objective_bspline, init, method='Nelder-Mead',
                          options={'maxiter': 200, 'xatol': 1e-6, 'fatol': 1e-6})
            
            if res.fun < _objective_hard(best_params):
                best_params = res.x
                best_pts = _decode_bspline(best_params)
                best_score = _score(best_pts)
        except Exception:
            pass
    
    # Phase 3: Cartesian refinement
    if _time_remaining() > 10.0:
        best_pts = _refine_cartesian(best_pts)
        new_score = _score(best_pts)
        if new_score > best_score:
            best_score = new_score
    
    # Phase 4: Hull pull
    if _time_remaining() > 10.0:
        best_pts = _hull_pull(best_pts)
        new_score = _score(best_pts)
        if new_score > best_score:
            best_score = new_score
    
    # Final: ensure points are finite and valid
    if not np.all(np.isfinite(best_pts)):
        # Fallback: regular 14-gon
        angles = 2.0 * np.pi * np.arange(N) / N
        r = 1.0 / np.sqrt(np.pi)
        best_pts = np.column_stack([r * np.cos(angles), r * np.sin(angles)])
    
    return best_pts
