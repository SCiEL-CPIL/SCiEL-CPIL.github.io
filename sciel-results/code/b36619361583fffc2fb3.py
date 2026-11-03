import numpy as np
from scipy.optimize import minimize
import time

N = 12
DEADLINE = 145.0  # safety margin before 150s kill

def hex_vertices(cx, cy, side, angle_deg):
    """Return 6 vertices of a regular hexagon with given center, side, rotation."""
    theta = np.radians(angle_deg)
    angles = np.pi / 3 * np.arange(6) + theta
    return np.column_stack([cx + side * np.cos(angles),
                            cy + side * np.sin(angles)])

def poly_overlap_sat(v1, v2):
    """SAT test: return True if two convex polygons overlap (including touching)."""
    n1, n2 = len(v1), len(v2)
    for i in range(n1):
        # Edge normal for polygon 1
        p1 = v1[i]
        p2 = v1[(i + 1) % n1]
        edge = p2 - p1
        normal = np.array([-edge[1], edge[0]])
        # Project both polygons onto this normal
        proj1 = v1 @ normal
        proj2 = v2 @ normal
        if proj1.max() < proj2.min() or proj2.max() < proj1.min():
            return False  # Separating axis found
    for i in range(n2):
        p1 = v2[i]
        p2 = v2[(i + 1) % n2]
        edge = p2 - p1
        normal = np.array([-edge[1], edge[0]])
        proj1 = v1 @ normal
        proj2 = v2 @ normal
        if proj1.max() < proj2.min() or proj2.max() < proj1.min():
            return False
    return True

def all_disjoint(inner_verts):
    """Check all pairs of inner hexagons are disjoint."""
    n = len(inner_verts)
    for i in range(n):
        for j in range(i + 1, n):
            if poly_overlap_sat(inner_verts[i], inner_verts[j]):
                return False
    return True

def enclosing_hexagon_size(verts_list, angle_deg):
    """
    Given list of polygon vertices, find the minimum side length of a regular hexagon
    (with given rotation) centered at some point that contains all of them.
    We try center = centroid of all points, and optimize.
    """
    # Collect all points
    all_pts = np.vstack(verts_list)
    cx0, cy0 = all_pts.mean(axis=0)
    
    # For a hexagon with side s, rotation a, center (cx,cy), a point (x,y) is inside
    # if for all 6 edge normals, the projection is within [-s*sqrt(3)/2, s*sqrt(3)/2]
    # Actually, the support function: max over points of (point - center) . n_k <= s * sqrt(3)/2
    # where n_k are the 6 outward normals of the hexagon edges.
    
    theta = np.radians(angle_deg)
    # Outward normals of hexagon edges: directions at 0, 60, 120, 180, 240, 300 degrees
    normals = np.column_stack([np.cos(np.pi/3 * np.arange(6) + theta),
                               np.sin(np.pi/3 * np.arange(6) + theta)])
    
    def required_side(cx, cy):
        rel = all_pts - np.array([cx, cy])
        # For each normal direction, the max projection must be <= s * sqrt(3)/2
        # The hexagon has width sqrt(3)*s in each normal direction
        # So s >= max_k (max_p (p-center).n_k) / (sqrt(3)/2)
        # But we need both sides: max and min projections
        # Actually for a regular hexagon centered at c with side s and rotation theta:
        # A point p is inside iff for all k, | (p-c) . n_k | <= s * sqrt(3) / 2
        # Wait, that's not quite right. Let me reconsider.
        # The hexagon vertices are at c + s * (cos(60k+theta), sin(60k+theta))
        # The edges are between consecutive vertices. The outward normal of edge k
        # bisects the angle between vertex k and vertex k+1.
        # Edge k goes from vertex k to vertex k+1. Midpoint direction = 60k+30+theta
        # So outward normals are at angles 30+60k+theta, k=0..5
        pass
    
    # Let me use a cleaner formulation:
    # Regular hexagon with center c, side s, rotation theta.
    # Vertices: c + s * (cos(60k + theta), sin(60k + theta)), k=0..5
    # The hexagon is the intersection of 6 half-planes.
    # Edge between vertex k and vertex k+1: the outward normal points in direction 60k+30+theta
    # The distance from center to each edge is s * cos(30°) = s * sqrt(3)/2
    # So constraint: (p - c) . n_k <= s * sqrt(3)/2 for all k and all points p
    # where n_k = (cos(60k+30+theta), sin(60k+30+theta))
    
    norm_angles = np.pi/6 + np.pi/3 * np.arange(6) + theta
    normals = np.column_stack([np.cos(norm_angles), np.sin(norm_angles)])
    
    def compute_s(cx, cy):
        rel = all_pts - np.array([cx, cy])
        projs = rel @ normals.T  # shape (npts, 6)
        max_projs = projs.max(axis=0)  # shape (6,)
        s = max_projs.max() / (np.sqrt(3) / 2)
        return s
    
    def obj(params):
        cx, cy = params
        return compute_s(cx, cy)
    
    # Optimize center
    res = minimize(obj, [cx0, cy0], method='Nelder-Mead', 
                   options={'maxiter': 5000, 'xatol': 1e-6, 'fatol': 1e-8})
    return compute_s(res.x[0], res.x[1]), (res.x[0], res.x[1])

def try_configuration(inner, outer_angle):
    """
    inner: list of (x, y, angle_deg) for N unit hexagons
    outer_angle: rotation of outer hexagon
    Returns (side, center) or (inf, None) if invalid.
    """
    inner_verts = [hex_vertices(x, y, 1.0, a) for (x, y, a) in inner]
    if not all_disjoint(inner_verts):
        return np.inf, None
    side, center = enclosing_hexagon_size(inner_verts, outer_angle)
    return side, center

def hexagonal_ring_config():
    """6 hexagons in a ring around center, plus 6 more."""
    # Central hexagon at origin
    # 6 hexagons around it at distance 2 (touching)
    # Then 6 more outside
    inner = []
    # Center
    inner.append((0.0, 0.0, 0.0))
    # Ring of 6 at distance 2, angles 0, 60, ..., 300
    for k in range(6):
        ang = np.radians(60 * k)
        x, y = 2.0 * np.cos(ang), 2.0 * np.sin(ang)
        inner.append((x, y, 0.0))
    # Outer ring of 6 at distance 4, angles 30, 90, ...
    for k in range(6):
        ang = np.radians(60 * k + 30)
        x, y = 4.0 * np.cos(ang), 4.0 * np.sin(ang)
        inner.append((x, y, 0.0))
    return inner

def two_row_config():
    """Two rows of 6 hexagons each."""
    inner = []
    for row in range(2):
        for col in range(6):
            x = 2.0 * col + (1.0 if row == 1 else 0.0)
            y = np.sqrt(3) * row
            inner.append((x, y, 0.0))
    return inner

def grid_3x4_config():
    """3 rows of 4 hexagons."""
    inner = []
    for row in range(3):
        for col in range(4):
            x = 2.0 * col + (1.0 if row % 2 == 1 else 0.0)
            y = np.sqrt(3) * row
            inner.append((x, y, 0.0))
    return inner

def optimize_positions(initial_inner, outer_angle, time_limit=20):
    """
    Optimize positions and angles of inner hexagons to minimize enclosing hexagon size.
    """
    start_time = time.time()
    
    # Parameters: for each hexagon, x, y, angle (3 params each) = 36 params
    def unpack(params):
        inner = []
        for i in range(N):
            x, y, a = params[3*i], params[3*i+1], params[3*i+2]
            inner.append((x, y, a))
        return inner
    
    def objective(params):
        if time.time() - start_time > time_limit:
            return 1e10
        inner = unpack(params)
        side, center = try_configuration(inner, outer_angle)
        return side
    
    # Initial parameters
    p0 = []
    for (x, y, a) in initial_inner:
        p0.extend([x, y, np.radians(a)])
    
    # Use differential evolution for global search
    from scipy.optimize import differential_evolution
    
    # Bounds: positions in [-10, 10], angles in [0, 2*pi]
    bounds = []
    for i in range(N):
        bounds.extend([(-10, 10), (-10, 10), (0, 2*np.pi)])
    
    def obj_wrapper(params):
        if time.time() - start_time > time_limit:
            return 1e10
        inner = unpack(params)
        side, center = try_configuration(inner, outer_angle)
        return side
    
    try:
        result = differential_evolution(obj_wrapper, bounds, 
                                        maxiter=100, popsize=15,
                                        tol=1e-6, seed=42,
                                        polish=False)
        if result.fun < 1e9:
            best_params = result.x
            inner_best = unpack(best_params)
            side, center = try_configuration(inner_best, outer_angle)
            if side < 1e9:
                return inner_best, side, center
    except Exception:
        pass
    
    # Also try local refinement from initial
    try:
        def obj_local(params):
            if time.time() - start_time > time_limit:
                return 1e10
            inner = unpack(params)
            side, center = try_configuration(inner, outer_angle)
            return side
        
        res = minimize(obj_local, p0, method='Nelder-Mead',
                      options={'maxiter': 5000, 'xatol': 1e-5, 'fatol': 1e-7})
        if res.fun < 1e9:
            inner_best = unpack(res.x)
            side, center = try_configuration(inner_best, outer_angle)
            if side < 1e9:
                return inner_best, side, center
    except Exception:
        pass
    
    return None, np.inf, None

def construct():
    start_time = time.time()
    
    best_side = np.inf
    best_result = None
    
    # Try several topologies and outer angles
    topologies = [
        hexagonal_ring_config,
        two_row_config,
        grid_3x4_config,
    ]
    
    outer_angles = [0.0, 30.0]
    
    for outer_ang in outer_angles:
        for top_func in topologies:
            if time.time() - start_time > DEADLINE - 10:
                break
            try:
                initial = top_func()
                # First, evaluate without optimization
                side, center = try_configuration(initial, outer_ang)
                if side < best_side:
                    best_side = side
                    inner_list = [(x, y, a) for (x, y, a) in initial]
                    best_result = (inner_list, center, side, outer_ang)
                
                # Now optimize if we have time
                if time.time() - start_time < DEADLINE - 30:
                    inner_opt, side_opt, center_opt = optimize_positions(
                        initial, outer_ang, time_limit=20)
                    if inner_opt is not None and side_opt < best_side:
                        best_side = side_opt
                        best_result = (inner_opt, center_opt, side_opt, outer_ang)
            except Exception:
                continue
    
    if best_result is not None:
        return best_result
    else:
        # Fallback: return the initial seed
        inner = [[4.0 * k, 0.0, 0.0] for k in range(-(N // 2), N - N // 2)]
        return inner, (0.0, 0.0), 40.0, 0.0

# EVOLVE-BLOCK-START
# (The construct function above is the main entry point)
# EVOLVE-BLOCK-END
