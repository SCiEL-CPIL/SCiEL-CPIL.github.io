"""Pack 12 unit hexagons in minimal outer hexagon."""
import numpy as np
from scipy.optimize import minimize
import time

N = 12
START_TIME = time.time()
TIME_LIMIT = 140.0  # safety margin before 150s kill

def hex_vertices(cx, cy, side, angle_deg):
    """Vertices of a regular hexagon centered at (cx,cy) with given side and rotation."""
    a = np.radians(angle_deg)
    verts = []
    for k in range(6):
        theta = a + k * np.pi / 3
        verts.append([cx + side * np.cos(theta), cy + side * np.sin(theta)])
    return np.array(verts)

def hex_apothem(side):
    return side * np.sqrt(3) / 2.0

def segment_interior_intersection(p1, p2, p3, p4):
    """Check if segments p1-p2 and p3-p4 have a proper interior intersection."""
    d1 = p2 - p1
    d2 = p4 - p3
    cross = d1[0] * d2[1] - d1[1] * d2[0]
    if abs(cross) < 1e-12:
        return False
    t = ((p3[0] - p1[0]) * d2[1] - (p3[1] - p1[1]) * d2[0]) / cross
    u = ((p3[0] - p1[0]) * d1[1] - (p3[1] - p1[1]) * d1[0]) / cross
    return 1e-9 < t < 1 - 1e-9 and 1e-9 < u < 1 - 1e-9

def polygons_disjoint(v1, v2, eps=1e-9):
    """Check if two convex polygons are disjoint (no interior overlap)."""
    # Check if any edge of v1 properly intersects any edge of v2
    n1, n2 = len(v1), len(v2)
    for i in range(n1):
        p1, p2 = v1[i], v1[(i + 1) % n1]
        for j in range(n2):
            p3, p4 = v2[j], v2[(j + 1) % n2]
            if segment_interior_intersection(p1, p2, p3, p4):
                return False
    # Check if any vertex of v1 is strictly inside v2
    for vi in v1:
        if point_in_polygon(vi, v2, strict=True):
            return False
    # Check if any vertex of v2 is strictly inside v1
    for vi in v2:
        if point_in_polygon(vi, v1, strict=True):
            return False
    return True

def point_in_polygon(p, poly, strict=False):
    """Ray casting for point in convex polygon."""
    n = len(poly)
    inside = False
    j = n - 1
    for i in range(n):
        pi, pj = poly[i], poly[j]
        if ((pi[1] > p[1]) != (pj[1] > p[1])):
            x_intersect = (pj[0] - pi[0]) * (p[1] - pi[1]) / (pj[1] - pi[1]) + pi[0]
            if p[0] < x_intersect:
                inside = not inside
        j = i
    if strict:
        # Check if on boundary
        for i in range(n):
            vi, vj = poly[i], poly[(i + 1) % n]
            d = vj - vi
            w = p - vi
            cross = d[0] * w[1] - d[1] * w[0]
            if abs(cross) < 1e-9:
                t = np.dot(w, d) / np.dot(d, d)
                if -1e-9 <= t <= 1 + 1e-9:
                    return False
    return inside

def all_inner_disjoint(inner_verts, min_gap=0.0):
    """Check all pairs of inner hexagons are disjoint."""
    n = len(inner_verts)
    for i in range(n):
        for j in range(i + 1, n):
            if not polygons_disjoint(inner_verts[i], inner_verts[j]):
                return False
    return True

def all_inside_outer(inner_verts, outer_verts):
    """Check all inner hexagons are inside outer hexagon."""
    for iv in inner_verts:
        for vi in iv:
            if not point_in_polygon(vi, outer_verts, strict=False):
                return False
    return True

def compute_outer_side(inner_verts):
    """Compute minimum outer hexagon side that contains all inner hexagons (axis-aligned outer)."""
    # For an outer hexagon with rotation 0 (flat-top), the constraint is:
    # For each point (x,y), we need max of projections onto 3 axes <= apothem_outer
    # The 3 axis directions for flat-top hex: 0, 60, 120 degrees
    # apothem = side * sqrt(3)/2, so side = 2*apothem/sqrt(3)
    
    all_pts = np.vstack(inner_verts)
    # For flat-top hexagon centered at origin, the half-widths:
    # x direction: |x| <= side
    # 60 deg: |x*cos60 + y*sin60| <= side * sqrt(3)/2 ... 
    # Actually for a regular hexagon with flat top, vertices at angles 0,60,...,300
    # The supporting lines are at angles 30, 90, 150, 210, 270, 330
    # Let me use the 3 axis pairs: directions 0°, 60°, 120°
    
    # For a regular hexagon with side s, flat-top, centered at origin:
    # It's the intersection of 3 strips: |x*cos(theta_k) + y*sin(theta_k)| <= s*sqrt(3)/2
    # where theta_k = 30°, 90°, 150° (normals to the 3 pairs of parallel sides)
    
    angles = [np.radians(30), np.radians(90), np.radians(150)]
    max_proj = 0
    for a in angles:
        projs = np.abs(all_pts[:, 0] * np.cos(a) + all_pts[:, 1] * np.sin(a))
        max_proj = max(max_proj, np.max(projs))
    
    apothem = max_proj
    side = 2 * apothem / np.sqrt(3)
    return side

def objective(params):
    """Objective: outer hexagon side length + penalty for violations."""
    # params: 12*3 (x, y, angle) for inner + 2 (cx, cy) for outer center = 38
    inner_params = params[:36].reshape(12, 3)
    outer_cx, outer_cy = params[36], params[37]
    
    inner_verts = []
    for k in range(12):
        x, y, ang = inner_params[k]
        v = hex_vertices(x, y, 1.0, ang)
        inner_verts.append(v)
    
    # Compute outer side needed
    all_pts = np.vstack(inner_verts)
    angles = [np.radians(30), np.radians(90), np.radians(150)]
    max_proj = 0
    for a in angles:
        projs = np.abs((all_pts[:, 0] - outer_cx) * np.cos(a) + (all_pts[:, 1] - outer_cy) * np.sin(a))
        max_proj = max(max_proj, np.max(projs))
    
    side = 2 * max_proj / np.sqrt(3)
    
    # Penalty for overlap
    penalty = 0.0
    for i in range(12):
        for j in range(i + 1, 12):
            v1, v2 = inner_verts[i], inner_verts[j]
            # Check vertex-in-polygon for overlap detection
            for vi in v1:
                if point_in_polygon(vi, v2, strict=True):
                    penalty += 100.0
            for vi in v2:
                if point_in_polygon(vi, v1, strict=True):
                    penalty += 100.0
            # Edge crossings
            for i1 in range(6):
                p1, p2 = v1[i1], v1[(i1 + 1) % 6]
                for j1 in range(6):
                    p3, p4 = v2[j1], v2[(j1 + 1) % 6]
                    if segment_interior_intersection(p1, p2, p3, p4):
                        penalty += 100.0
                        break
                else:
                    continue
                break
    
    return side + penalty

def make_initial_guess():
    """Create initial guess: 1 center + 6 ring + 5 outer."""
    params = np.zeros(38)
    # Center hexagon
    params[0] = 0.0
    params[1] = 0.0
    params[2] = 0.0
    
    # 6 hexagons in a ring at distance sqrt(3)
    r = np.sqrt(3)
    for k in range(6):
        theta = k * np.pi / 3
        idx = (k + 1) * 3
        params[idx] = r * np.cos(theta)
        params[idx + 1] = r * np.sin(theta)
        params[idx + 2] = 0.0
    
    # 5 more hexagons: place them in the 5 gaps between ring hexagons, further out
    # The gaps are at angles 30°, 90°, 150°, 210°, 270° (between the 6 ring positions)
    # Actually, place them at the vertices of a pentagon-like arrangement
    # Better: place at distance ~2*sqrt(3) at angles offset
    r2 = 2.0 * np.sqrt(3)
    for k in range(5):
        theta = np.radians(30) + k * 2 * np.pi / 5  # spread around
        idx = (k + 7) * 3
        params[idx] = r2 * np.cos(theta)
        params[idx + 1] = r2 * np.sin(theta)
        params[idx + 2] = 0.0
    
    # Outer center
    params[36] = 0.0
    params[37] = 0.0
    
    return params

def construct():
    global START_TIME
    best_result = None
    best_score = np.inf
    
    x0 = make_initial_guess()
    
    # Try multiple starting points and optimization methods
    starts = [x0]
    
    # Add some perturbed starts
    rng = np.random.RandomState(42)
    for _ in range(3):
        x_pert = x0 + rng.randn(38) * 0.3
        starts.append(x_pert)
    
    for x_start in starts:
        if time.time() - START_TIME > TIME_LIMIT - 10:
            break
        
        try:
            res = minimize(objective, x_start, method='Nelder-Mead',
                          options={'maxiter': 2000, 'xatol': 1e-8, 'fatol': 1e-10,
                                   'adaptive': True})
            if res.fun < best_score and res.fun < 20:  # sanity check
                best_score = res.fun
                best_result = res.x
        except Exception:
            pass
    
    if best_result is None:
        # Fallback: return the initial guess
        best_result = x0
    
    inner_params = best_result[:36].reshape(12, 3)
    outer_cx, outer_cy = best_result[36], best_result[37]
    
    inner = [[float(x), float(y), float(ang)] for x, y, ang in inner_params]
    
    # Compute final outer side
    inner_verts = [hex_vertices(x, y, 1.0, ang) for x, y, ang in inner]
    all_pts = np.vstack(inner_verts)
    angles = [np.radians(30), np.radians(90), np.radians(150)]
    max_proj = 0
    for a in angles:
        projs = np.abs((all_pts[:, 0] - outer_cx) * np.cos(a) + (all_pts[:, 1] - outer_cy) * np.sin(a))
        max_proj = max(max_proj, np.max(projs))
    side = 2 * max_proj / np.sqrt(3)
    
    return inner, (float(outer_cx), float(outer_cy)), float(side), 0.0
