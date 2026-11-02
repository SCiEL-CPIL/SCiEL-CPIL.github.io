"""
Combined approach: B-spline convex boundary parameterization with edge-vector
convexity constraints, soft-min objective, and multi-phase optimization.

Strategy:
  1. Primary representation: 14 points as vertices of a convex polygon defined
     by 14 edge vectors with monotonically increasing directions and positive
     lengths. This allows non-radial, asymmetric deformation of the hull.
  2. Fallback: B-spline control point parameterization (7 control points).
  3. Soft-min objective for smoother optimization landscape.
  4. Multi-phase: global exploration (DE) -> local NM/Powell -> final polish.
  5. Time-guarded to finish well within 150s.
"""
import time
import numpy as np
from scipy.optimize import minimize, differential_evolution
from scipy.spatial import ConvexHull
from scipy.interpolate import CubicSpline
import itertools

N = 14
DEADLINE_MARGIN = 12.0  # safety margin before 150s kill

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


def _all_triangle_areas(points):
    pi = points[tri_indices[:, 0]]
    pj = points[tri_indices[:, 1]]
    pk = points[tri_indices[:, 2]]
    cross = (pj[:, 0] - pi[:, 0]) * (pk[:, 1] - pi[:, 1]) - (pj[:, 1] - pi[:, 1]) * (pk[:, 0] - pi[:, 0])
    return 0.5 * np.abs(cross)


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


def _objective_soft(flat, beta=80.0):
    pts = flat.reshape(N, 2)
    if np.any(~np.isfinite(pts)):
        return 1e10
    areas = _all_triangle_areas(pts)
    if np.min(areas) < 1e-12:
        return 1e10
    z = -beta * areas
    max_z = np.max(z)
    if max_z < -500:
        log_sum = max_z
    else:
        log_sum = max_z + np.log(np.sum(np.exp(z - max_z)))
    soft_min = -log_sum / beta
    if soft_min < 1e-12:
        return 1e10
    hull_area = _hull_area(pts)
    if hull_area < 1e-10:
        return 1e10
    return hull_area / soft_min


# ---------- Edge-vector convex polygon parameterization ----------
# 14 points as vertices of a convex polygon defined by 14 edge vectors.
# Each edge has a direction angle (monotonically increasing) and a length.
# The polygon is closed, so the sum of edge vectors must be zero.

def _edge_vector_decode(params):
    """
    Decode edge-vector parameters to N points.
    
    params: shape (2*N,) = [log_len_0, angle_0, log_len_1, angle_1, ...]
    - log_len_k: log of edge length (exp to get actual length)
    - angle_k: direction angle of edge k
    
    The angles are constrained to be monotonically increasing with total
    turn of 2*pi. We use a parameterization where:
    - base_angle_k = 2*pi*k/N (regular polygon)
    - perturbation_k is small and constrained
    
    Actually, for a closed convex polygon, the edge directions must satisfy:
    sum of edge vectors = 0, and the directions are in [0, 2*pi) in order.
    
    Simpler approach: parameterize by the vertex angles and radii directly,
    but enforce convexity via the constraint that all cross products of
    consecutive edges are positive.
    
    For this implementation, we use a simpler parameterization:
    params: [r_0, theta_0, r_1, theta_1, ..., r_13, theta_13]
    where r_k > 0 and theta_k are angles in [0, 2*pi) sorted.
    The points are (r_k * cos(theta_k), r_k * sin(theta_k)).
    Convexity is enforced by the optimizer via penalty.
    """
    r = np.exp(params[0::2])  # positive radii
    theta = params[1::2]  # angles
    
    # Ensure angles are in [0, 2*pi) and sorted
    theta = np.mod(theta, 2 * np.pi)
    theta = np.sort(theta)
    
    # If angles are too close, spread them out
    # This is a heuristic to maintain some convexity
    
    pts = np.column_stack([r * np.cos(theta), r * np.sin(theta)])
    
    # Center the points
    pts -= pts.mean(axis=0)
    
    return pts


def _is_convex(pts):
    """Check if points form a convex polygon (in order)."""
    n = len(pts)
    if n < 3:
        return False
    sign = 0
    for i in range(n):
        p0 = pts[i]
        p1 = pts[(i + 1) % n]
        p2 = pts[(i + 2) % n]
        cross = (p1[0] - p0[0]) * (p2[1] - p0[1]) - (p1[1] - p0[1]) * (p2[0] - p0[0])
        if abs(cross) < 1e-12:
            continue
        if sign == 0:
            sign = 1 if cross > 0 else -1
        elif (cross > 0 and sign < 0) or (cross < 0 and sign > 0):
            return False
    return True


def _objective_edge(flat, convex_penalty=100.0):
    """Objective with convexity penalty for edge-vector parameterization."""
    pts = _edge_vector_decode(flat)
    if np.any(~np.isfinite(pts)):
        return 1e10
    
    # Check convexity and add penalty if not convex
    if not _is_convex(pts):
        # Compute how non-convex: sum of negative cross products
        n = len(pts)
        penalty = 0.0
        for i in range(n):
            p0 = pts[i]
            p1 = pts[(i + 1) % n]
            p2 = pts[(i + 2) % n]
            cross = (p1[0] - p0[0]) * (p2[1] - p0[1]) - (p1[1] - p0[1]) * (p2[0] - p0[0])
            if cross < 0:
                penalty += -cross
        return 1e10 + convex_penalty * penalty
    
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
    
    ctrl_params: shape (14,) = [x0,y0, x1,y1, ..., x6,y6]
    """
    ctrl = ctrl_params.reshape(7, 2)
    
    try:
        ctrl_closed = np.vstack([ctrl, ctrl[0]])
        t = np.arange(8)
        
        cs_x = CubicSpline(t, ctrl_closed[:, 0], bc_type='periodic')
        cs_y = CubicSpline(t, ctrl_closed[:, 1], bc_type='periodic')
        
        t_fine = np.linspace(0, 7, 2000)
        x_fine = cs_x(t_fine)
        y_fine = cs_y(t_fine)
        
        dx = np.diff(x_fine)
        dy = np.diff(y_fine)
        ds = np.sqrt(dx**2 + dy**2)
        s = np.concatenate([[0], np.cumsum(ds)])
        
        total_s = s[-1]
        s_targets = np.linspace(0, total_s, N, endpoint=False)
        
        x_pts = np.interp(s_targets, s, x_fine)
        y_pts = np.interp(s_targets, s, y_fine)
        
        pts = np.column_stack([x_pts, y_pts])
        pts -= pts.mean(axis=0)
        
        return pts
        
    except Exception:
        ctrl_centered = ctrl - ctrl.mean(axis=0)
        pts = np.vstack([ctrl_centered, -ctrl_centered[::-1]])
        return pts


def _objective_bspline(ctrl_params):
    pts = _decode_bspline(ctrl_params)
    return _objective_hard(pts.flatten())


def _regular_polygon(n, radius=1.0, phase=0.0):
    """Regular n-gon."""
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False) + phase
    return np.column_stack([radius * np.cos(angles), radius * np.sin(angles)])


def _two_ring_config(r_inner, n_inner, n_outer, phase_offset=0.0):
    """Two concentric rings."""
    a1 = np.linspace(0, 2 * np.pi, n_inner, endpoint=False)
    a2 = np.linspace(0, 2 * np.pi, n_outer, endpoint=False) + phase_offset
    p1 = r_inner * np.column_stack([np.cos(a1), np.sin(a1)])
    p2 = np.column_stack([np.cos(a2), np.sin(a2)])
    return np.vstack([p1, p2])


def construct():
    start_time = time.time()
    deadline = start_time + 150.0 - DEADLINE_MARGIN  # 138s effective
    
    best_score = 0.0
    best_pts = None
    best_obj = float('inf')
    
    def update_best(pts):
        nonlocal best_score, best_pts, best_obj
        if pts is None:
            return
        pts = np.asarray(pts, dtype=float)
        if pts.shape != (N, 2) or not np.all(np.isfinite(pts)):
            return
        s = _score(pts)
        if s > best_score:
            best_score = s
            best_pts = pts.copy()
            best_obj = _objective_hard(pts.flatten())
    
    def time_left():
        return deadline - time.time()
    
    # === Phase 0: Quick evaluation of structured initial configs ===
    inits = []
    inits.append(_regular_polygon(N))
    for r in [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]:
        inits.append(_two_ring_config(r, 7, 7, np.pi / 7))
    for n1, n2 in [(5, 9), (4, 10), (6, 8), (9, 5), (10, 4), (8, 6)]:
        for r in [0.4, 0.6, 0.7]:
            inits.append(_two_ring_config(r, n1, n2, np.pi / max(n1, 1)))
    for r1, r2 in [(0.3, 0.6), (0.4, 0.7), (0.5, 0.8)]:
        a1 = np.linspace(0, 2*np.pi, 5, endpoint=False)
        a2 = np.linspace(0, 2*np.pi, 4, endpoint=False) + np.pi/4
        a3 = np.linspace(0, 2*np.pi, 5, endpoint=False) + np.pi/5
        p1 = r1 * np.column_stack([np.cos(a1), np.sin(a1)])
        p2 = r2 * np.column_stack([np.cos(a2), np.sin(a2)])
        p3 = np.column_stack([np.cos(a3), np.sin(a3)])
        inits.append(np.vstack([p1, p2, p3]))
    
    for pts0 in inits:
        if time_left() < 20:
            break
        update_best(pts0)
    
    # === Phase 1: B-spline parameter optimization ===
    if time_left() > 30:
        # Generate diverse initial control point configurations
        rng = np.random.default_rng(42)
        bspline_starts = []
        
        # Regular heptagon
        angles = np.linspace(0, 2 * np.pi, 7, endpoint=False)
        ctrl = np.column_stack([np.cos(angles), np.sin(angles)])
        bspline_starts.append(ctrl.flatten().copy())
        
        # Elliptical
        for rx, ry in [(1.2, 0.8), (0.8, 1.2), (1.1, 0.9)]:
            angles = np.linspace(0, 2 * np.pi, 7, endpoint=False)
            ctrl = np.column_stack([rx * np.cos(angles), ry * np.sin(angles)])
            bspline_starts.append(ctrl.flatten().copy())
        
        # Perturbed regular
        for js in [0.05, 0.1, 0.15]:
            angles = np.linspace(0, 2 * np.pi, 7, endpoint=False)
            ctrl = np.column_stack([np.cos(angles), np.sin(angles)])
            ctrl += rng.normal(0, js, size=(7, 2))
            bspline_starts.append(ctrl.flatten().copy())
        
        # Random convex-ish
        for seed in range(10):
            rs = np.random.default_rng(seed * 13 + 7)
            angles = np.sort(rs.uniform(0, 2 * np.pi, 7))
            radii = rs.uniform(0.6, 1.4, 7)
            ctrl = np.column_stack([radii * np.cos(angles), radii * np.sin(angles)])
            bspline_starts.append(ctrl.flatten().copy())
        
        for bstart in bspline_starts:
            if time_left() < 15:
                break
            try:
                res = minimize(_objective_bspline, bstart, method='Nelder-Mead',
                               options={'maxiter': 2000, 'xatol': 1e-9, 'fatol': 1e-11, 'adaptive': True})
                if res.fun < best_obj:
                    pts_b = _decode_bspline(res.x)
                    update_best(pts_b)
            except Exception:
                pass
    
    # === Phase 2: Edge-vector parameter optimization ===
    if time_left() > 25:
        # Generate initial edge-vector parameters from best points
        if best_pts is not None:
            # Convert best_pts to edge-vector representation
            # Sort points by angle around centroid
            centroid = best_pts.mean(axis=0)
            angles = np.arctan2(best_pts[:, 1] - centroid[1], best_pts[:, 0] - centroid[0])
            sort_idx = np.argsort(angles)
            sorted_pts = best_pts[sort_idx]
            
            # Compute radii and angles
            radii = np.linalg.norm(sorted_pts - centroid, axis=1)
            angles_sorted = np.sort(angles)
            
            # Initialize params: [log(r_0), theta_0, log(r_1), theta_1, ...]
            edge_params = np.zeros(2 * N)
            edge_params[0::2] = np.log(np.maximum(radii, 0.1))
            edge_params[1::2] = angles_sorted
            
            if time_left() > 20:
                try:
                    res = minimize(_objective_edge, edge_params, method='Nelder-Mead',
                                   options={'maxiter': 2000, 'xatol': 1e-9, 'fatol': 1e-11, 'adaptive': True})
                    if res.fun < best_obj:
                        pts_e = _edge_vector_decode(res.x)
                        update_best(pts_e)
                except Exception:
                    pass
        
        # Try random edge-vector starts
        for seed in range(5):
            if time_left() < 10:
                break
            rs = np.random.default_rng(seed * 17 + 3)
            radii = rs.uniform(0.5, 1.5, N)
            angles = np.sort(rs.uniform(0, 2 * np.pi, N))
            edge_params = np.zeros(2 * N)
            edge_params[0::2] = np.log(radii)
            edge_params[1::2] = angles
            
            try:
                res = minimize(_objective_edge, edge_params, method='Nelder-Mead',
                               options={'maxiter': 1000, 'xatol': 1e-8, 'fatol': 1e-10, 'adaptive': True})
                if res.fun < best_obj:
                    pts_e = _edge_vector_decode(res.x)
                    update_best(pts_e)
            except Exception:
                pass
    
    # === Phase 3: Multi-start Nelder-Mead on full 28D space ===
    if best_pts is not None and time_left() > 20:
        try:
            res = minimize(_objective_hard, best_pts.flatten(), method='Nelder-Mead',
                           options={'maxiter': 3000, 'xatol': 1e-9, 'fatol': 1e-11, 'adaptive': True})
            if res.fun < best_obj:
                update_best(res.x.reshape(N, 2))
        except Exception:
            pass
        
        # Jittered versions
        rng = np.random.default_rng(123)
        for std in [0.05, 0.1, 0.15]:
            if time_left() < 10:
                break
            jittered = best_pts + rng.normal(0, std, (N, 2))
            try:
                res = minimize(_objective_hard, jittered.flatten(), method='Nelder-Mead',
                               options={'maxiter': 1500, 'xatol': 1e-9, 'fatol': 1e-11, 'adaptive': True})
                if res.fun < best_obj:
                    update_best(res.x.reshape(N, 2))
            except Exception:
                pass
    
    # === Phase 4: Differential evolution for global exploration ===
    if time_left() > 40:
        try:
            bounds = [(-2, 2)] * (2 * N)
            de_res = differential_evolution(_objective_hard, bounds, maxiter=50,
                                            popsize=15, tol=1e-8, seed=42,
                                            polish=True, init='sobol')
            if de_res.fun < best_obj:
                update_best(de_res.x.reshape(N, 2))
        except Exception:
            pass
    
    # === Phase 5: Powell method for final polish ===
    if best_pts is not None and time_left() > 15:
        try:
            res_powell = minimize(_objective_hard, best_pts.flatten(), method='Powell',
                                  options={'maxiter': 2000, 'xtol': 1e-12, 'ftol': 1e-14})
            if res_powell.fun < best_obj:
                update_best(res_powell.x.reshape(N, 2))
        except Exception:
            pass
    
    # === Phase 6: Final tight polish ===
    if best_pts is not None and time_left() > 5:
        try:
            res = minimize(_objective_hard, best_pts.flatten(), method='Nelder-Mead',
                           options={'maxiter': 5000, 'xatol': 1e-11, 'fatol': 1e-13, 'adaptive': True})
            if res.fun < best_obj:
                update_best(res.x.reshape(N, 2))
        except Exception:
            pass
    
    # Fallback
    if best_pts is None:
        best_pts = _regular_polygon(N)
    
    # Normalize to unit area
    try:
        hull = ConvexHull(best_pts)
        area = hull.volume
        if area > 1e-10:
            scale = np.sqrt(1.0 / area)
            best_pts = best_pts * scale
    except Exception:
        pass
    
    return best_pts
