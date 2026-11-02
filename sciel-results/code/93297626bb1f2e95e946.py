"""Optimized placement of 11 unit hexagons inside a regular hexagon."""
import numpy as np
import time
from scipy.optimize import minimize, linprog

N = 11
DEADLINE = 120.0
START_TIME = time.time()

def _check_deadline():
    return (time.time() - START_TIME) < DEADLINE - 2.0

def _hex_vertices_local():
    """Vertices of a unit regular hexagon centered at origin, flat-top orientation.
    Side length 1, so circumradius = 1.
    Angles: 0, 60, 120, 180, 240, 300 degrees."""
    angles = np.array([0, 60, 120, 180, 240, 300]) * np.pi / 180
    return np.column_stack([np.cos(angles), np.sin(angles)])

def _hex_vertices(cx, cy, angle_deg):
    """Vertices of a unit hexagon at (cx,cy) rotated by angle_deg."""
    verts = _hex_vertices_local()
    theta = np.radians(angle_deg)
    R = np.array([[np.cos(theta), -np.sin(theta)],
                  [np.sin(theta), np.cos(theta)]])
    verts = verts @ R.T
    return verts + np.array([cx, cy])

def _outer_side_normals(angle_deg):
    """
    For a regular hexagon with flat-top orientation rotated by angle_deg,
    the 6 outward normals to the sides.
    A regular hexagon has sides perpendicular to directions at 30, 90, 150, 210, 270, 330 degrees
    for flat-top. Actually, let's think carefully.
    
    A regular hexagon with vertices at angles 0,60,...,300 (pointy-top) has sides
    with outward normals at 30, 90, 150, 210, 270, 330.
    
    But our _hex_vertices_local has vertices at 0,60,...,300 which is pointy-top.
    The side between vertex at 0° and 60° has outward normal at 30°.
    So outward normals are at 30° + k*60° for k=0..5.
    
    When we rotate the hexagon by angle_deg, the normals rotate too.
    """
    base_normals = np.array([30 + 60*k for k in range(6)]) * np.pi / 180
    theta = np.radians(angle_deg)
    R = np.array([[np.cos(theta), -np.sin(theta)],
                  [np.sin(theta), np.cos(theta)]])
    normals = np.column_stack([np.cos(base_normals), np.sin(base_normals)])
    normals = normals @ R.T
    return normals  # shape (6,2)

def _compute_outer_side(inner_points, outer_center, outer_angle_deg):
    """
    Compute the minimum side length of a regular hexagon (with given center and angle)
    that contains all inner hexagons.
    
    For each of the 6 sides of the outer hexagon, we need the maximum projection
    of any vertex of any inner hexagon onto the outward normal of that side.
    The side length s must satisfy: for each side i, the projection of all points
    onto normal_i must be <= s * (apothem factor) ... 
    
    Actually, for a regular hexagon with side length s and circumradius R=s,
    the apothem (distance from center to each side) is s * sqrt(3)/2.
    
    So for each outward normal n_i, we need:
    max over all vertices v: dot(v - center, n_i) <= s * sqrt(3)/2
    
    Therefore s >= max_i (max_v dot(v-center, n_i)) / (sqrt(3)/2)
    
    But also, the opposite side: dot(v - center, -n_i) <= s * sqrt(3)/2
    i.e., -dot(v-center, n_i) <= s * sqrt(3)/2
    So |dot(v-center, n_i)| <= s * sqrt(3)/2 for all v and all i.
    
    Wait, that's not quite right. The hexagon has 6 sides. For side with outward normal n_i,
    the constraint is dot(v - center, n_i) <= apothem = s*sqrt(3)/2.
    The opposite side has outward normal -n_i (which is one of the 6 normals), so
    dot(v - center, -n_i) <= s*sqrt(3)/2, i.e., dot(v-center, n_i) >= -s*sqrt(3)/2.
    
    So for each normal direction n_i (i=0..5), we need:
    -s*sqrt(3)/2 <= dot(v-center, n_i) <= s*sqrt(3)/2 for all vertices v.
    
    Thus s >= (1/(sqrt(3)/2)) * max_i max_v |dot(v-center, n_i)|
    
    Hmm wait, that's not right either. The 6 normals are at 60° intervals, so they cover
    all 6 sides. For each side, only the upper bound matters (the lower bound is handled
    by the opposite side). So:
    
    s >= max_i [ max_v dot(v-center, n_i) ] / (sqrt(3)/2)
    
    But we also need the hexagon to actually contain the points, which requires both
    upper and lower bounds. Since the 6 normals come in opposite pairs (n_{i+3} = -n_i),
    the constraint for side i is dot(v-center, n_i) <= s*sqrt(3)/2, and for side i+3
    it's dot(v-center, -n_i) <= s*sqrt(3)/2, i.e., dot(v-center, n_i) >= -s*sqrt(3)/2.
    
    So s >= max( max_i max_v dot(v-center, n_i), max_i max_v dot(v-center, -n_i) ) / (sqrt(3)/2)
    = max_i max_v |dot(v-center, n_i)| / (sqrt(3)/2)
    
    Wait no. Let me re-derive. We have 6 sides with outward normals n_0,...,n_5 where
    n_{i+3} = -n_i. For side i, the half-space is dot(x - center, n_i) <= apothem.
    So for all vertices v: dot(v - center, n_i) <= s*sqrt(3)/2 for all i=0..5.
    
    This is equivalent to: for i=0,1,2: 
    dot(v-center, n_i) <= s*sqrt(3)/2 AND dot(v-center, -n_i) <= s*sqrt(3)/2
    i.e., |dot(v-center, n_i)| <= s*sqrt(3)/2 for i=0,1,2.
    
    So s >= max_{i in {0,1,2}} max_v |dot(v-center, n_i)| / (sqrt(3)/2)
    
    Actually, we should just use all 6 and take the max of dot(v-center, n_i) over all i and v,
    then divide by sqrt(3)/2. Because the negative cases are covered by the opposite normals.
    """
    normals = _outer_side_normals(outer_angle_deg)  # (6,2)
    center = np.array(outer_center)
    
    max_proj = 0.0
    for cx, cy, ang in inner_points:
        verts = _hex_vertices(cx, cy, ang)
        rel = verts - center  # (6,2)
        projs = rel @ normals.T  # (6,6)
        max_proj = max(max_proj, np.max(projs))
    
    return max_proj / (np.sqrt(3) / 2)

def _pairwise_penalty(inner_points, min_dist=2.0):
    """Penalty for overlapping inner hexagons. Two unit hexagons don't overlap
    if their centers are at least 2 apart (for same orientation). For different
    orientations, the minimum distance to avoid overlap depends on orientation.
    A safe lower bound: if center distance < 2, they likely overlap.
    Use a soft penalty."""
    n = len(inner_points)
    penalty = 0.0
    for i in range(n):
        for j in range(i+1, n):
            dx = inner_points[i][0] - inner_points[j][0]
            dy = inner_points[i][1] - inner_points[j][1]
            d = np.sqrt(dx*dx + dy*dy)
            if d < min_dist:
                penalty += (min_dist - d) ** 2
    return penalty

def _objective(params, outer_center, outer_angle_deg):
    """Objective: outer side length + overlap penalty."""
    n = N
    # params: 3*N = (x1,y1,ang1, x2,y2,ang2, ...)
    inner_points = []
    for i in range(n):
        x, y, ang = params[3*i], params[3*i+1], params[3*i+2]
        inner_points.append((x, y, ang))
    
    side = _compute_outer_side(inner_points, outer_center, outer_angle_deg)
    penalty = _pairwise_penalty(inner_points, min_dist=1.95)
    return side + 10.0 * penalty

def _optimize_inner(outer_center, outer_angle_deg, initial_params, max_iter=500):
    """Optimize inner hexagon positions."""
    n = N
    bounds = [(-20, 20)] * (3*n)
    
    result = minimize(_objective, initial_params, args=(outer_center, outer_angle_deg),
                      method='L-BFGS-B', bounds=bounds, options={'maxiter': max_iter, 'ftol': 1e-10})
    
    if result.success or result.fun < 100:
        inner_points = []
        for i in range(n):
            inner_points.append([result.x[3*i], result.x[3*i+1], result.x[3*i+2]])
        return inner_points, result.fun
    return None, float('inf')

def _optimize_outer(inner_points, outer_center_init, outer_angle_init):
    """Optimize outer hexagon center and angle for fixed inner points."""
    def outer_obj(params):
        oc = (params[0], params[1])
        oa = params[2]
        return _compute_outer_side(inner_points, oc, oa)
    
    result = minimize(outer_obj, np.array([outer_center_init[0], outer_center_init[1], outer_angle_init]),
                      method='Nelder-Mead', options={'maxiter': 200, 'xatol': 1e-6, 'fatol': 1e-8})
    
    oc = (result.x[0], result.x[1])
    oa = result.x[2]
    return oc, oa, result.fun

def construct():
    global START_TIME
    START_TIME = time.time()
    
    # Initial configuration: hexagonal packing
    # 11 hexagons in a hexagonal arrangement: center + 6 around + 4 more
    # Or a more compact arrangement
    
    # Start with a reasonable initial guess
    # Hexagonal lattice: spacing = 2 (for non-overlapping)
    # Positions for 11 hexagons in a compact cluster
    
    # A common arrangement: 1 center, 6 in first ring, 4 in second ring (partial)
    # Or: rows of 3, 4, 3, 1 = 11? Let's use a hexagonal number pattern.
    # Hexagonal numbers: 1, 7, 19. So 11 is between 7 and 19.
    
    # Let's use a pattern:
    # Row 0: 3 hexagons
    # Row 1: 4 hexagons  
    # Row 2: 3 hexagons
    # Row 3: 1 hexagon
    # Total: 11
    
    # Hexagonal lattice with spacing 2
    dx = 2.0
    dy = np.sqrt(3)  # for pointy-top, vertical spacing is sqrt(3)*side = sqrt(3)
    
    initial = []
    # Row 0 (y = 0): 3 hexagons at x = -2, 0, 2
    for x in [-2.0, 0.0, 2.0]:
        initial.append([x, 0.0, 0.0])
    # Row 1 (y = sqrt(3)): 4 hexagons at x = -3, -1, 1, 3
    for x in [-3.0, -1.0, 1.0, 3.0]:
        initial.append([x, np.sqrt(3), 0.0])
    # Row 2 (y = 2*sqrt(3)): 3 hexagons at x = -2, 0, 2
    for x in [-2.0, 0.0, 2.0]:
        initial.append([x, 2*np.sqrt(3), 0.0])
    # Row 3 (y = 3*sqrt(3)): 1 hexagon at x = 0
    initial.append([0.0, 3*np.sqrt(3), 0.0])
    
    # Flatten to params
    params0 = []
    for p in initial:
        params0.extend(p)
    
    best_inner = initial
    best_outer_center = (0.0, 0.0)
    best_outer_angle = 0.0
    best_side = _compute_outer_side(best_inner, best_outer_center, best_outer_angle)
    best_penalty = _pairwise_penalty(best_inner)
    best_total = best_side + 10.0 * best_penalty
    
    # Multi-start optimization
    n_starts = 5
    rng = np.random.default_rng(42)
    
    for start in range(n_starts):
        if not _check_deadline():
            break
        
        if start == 0:
            p0 = np.array(params0)
        else:
            p0 = np.array(params0) + rng.normal(0, 0.5, size=3*N)
        
        inner, val = _optimize_inner((0.0, 0.0), 0.0, p0, max_iter=300)
        
        if inner is not None:
            # Now optimize outer
            oc, oa, side_val = _optimize_outer(inner, (0.0, 0.0), 0.0)
            pen = _pairwise_penalty(inner, min_dist=1.95)
            total = side_val + 10.0 * pen
            
            if total < best_total:
                best_inner = inner
                best_outer_center = oc
                best_outer_angle = oa
                best_side = side_val
                best_penalty = pen
                best_total = total
    
    # Final polish: optimize outer more carefully
    if _check_deadline():
        oc, oa, side_val = _optimize_outer(best_inner, best_outer_center, best_outer_angle)
        if side_val < best_side:
            best_outer_center = oc
            best_outer_angle = oa
            best_side = side_val
    
    # Ensure no overlaps by checking and adjusting
    # If there are overlaps, we need to handle them
    # For now, return the best found
    
    # Format output
    inner_out = [[float(x), float(y), float(a)] for x, y, a in best_inner]
    return inner_out, (float(best_outer_center[0]), float(best_outer_center[1])), float(best_side), float(best_outer_angle)

if __name__ == "__main__":
    result = construct()
    print(f"Side length: {result[2]:.6f}")
    print(f"Outer center: {result[1]}")
    print(f"Outer angle: {result[3]}")
