import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial import ConvexHull

N = 11

def _hex_vertices(cx, cy, angle_deg=0.0):
    """
    Get vertices of a unit regular hexagon centered at (cx, cy) rotated by angle_deg.
    Vertices are at distance 1 from center.
    """
    angle_rad = np.deg2rad(angle_deg)
    base_angles = np.array([0, 60, 120, 180, 240, 300]) * np.pi / 180.0
    angles = base_angles + angle_rad
    x = cx + np.cos(angles)
    y = cy + np.sin(angles)
    return np.column_stack((x, y))

def _outer_hex_vertices(cx, cy, S, angle_deg=0.0):
    """
    Get vertices of a regular hexagon centered at (cx, cy) with side length S
    rotated by angle_deg.
    """
    # Circumradius = S
    angle_rad = np.deg2rad(angle_deg)
    base_angles = np.array([0, 60, 120, 180, 240, 300]) * np.pi / 180.0
    angles = base_angles + angle_rad
    x = cx + S * np.cos(angles)
    y = cy + S * np.sin(angles)
    return np.column_stack((x, y))

def _min_outer_side(inner_centers, inner_angles, outer_angle_deg=0.0):
    """
    Calculate the minimum side length S of the outer hexagon (centered at 0,0, rotated by outer_angle_deg)
    that contains all inner hexagons.
    Uses the support function approach: for each of the 6 outward normals of the outer hexagon,
    find the maximum projection of all inner hexagon vertices, then S = max_proj / cos(30°) = max_proj * 2/sqrt(3).
    """
    theta_rad = np.deg2rad(outer_angle_deg)
    # The 6 outward normals of the outer hexagon (pointing to the midpoints of each side)
    normals = np.array([[np.cos(theta_rad + np.deg2rad(30 + k * 60)),
                         np.sin(theta_rad + np.deg2rad(30 + k * 60))] for k in range(6)])
    
    max_proj = 0.0
    for (cx, cy), ang in zip(inner_centers, inner_angles):
        verts = _hex_vertices(cx, cy, ang)
        # Project each vertex onto each normal
        projs = verts @ normals.T
        local_max = np.max(projs)
        if local_max > max_proj:
            max_proj = local_max
    
    # For a regular hexagon with side S, the apothem (distance from center to midpoint of side) is S*sqrt(3)/2
    # So max_proj = S * sqrt(3)/2  =>  S = max_proj * 2/sqrt(3)
    S = max_proj * 2.0 / np.sqrt(3.0)
    return S

def _check_disjoint(centers, angles, tol=1e-6):
    """
    Check that all pairs of inner hexagons are disjoint (no overlap).
    Returns True if all are disjoint.
    """
    n = len(centers)
    for i in range(n):
        vi = _hex_vertices(centers[i][0], centers[i][1], angles[i])
        for j in range(i+1, n):
            vj = _hex_vertices(centers[j][0], centers[j][1], angles[j])
            # Check if any vertex of vi is inside vj or vice versa
            # Use a simple point-in-convex-polygon test
            if _polygon_overlap(vi, vj, tol):
                return False
    return True

def _point_in_convex_polygon(pt, poly):
    """Check if point pt is inside convex polygon poly (vertices in CCW order)."""
    n = len(poly)
    for i in range(n):
        a = poly[i]
        b = poly[(i+1) % n]
        cross = (b[0]-a[0])*(pt[1]-a[1]) - (b[1]-a[1])*(pt[0]-a[0])
        if cross < -1e-9:
            return False
    return True

def _polygon_overlap(p1, p2, tol=1e-6):
    """Check if two convex polygons overlap (including touching)."""
    # Check if any vertex of p1 is inside p2
    for pt in p1:
        if _point_in_convex_polygon(pt, p2):
            return True
    # Check if any vertex of p2 is inside p1
    for pt in p2:
        if _point_in_convex_polygon(pt, p1):
            return True
    # Check edge intersections
    n1 = len(p1)
    n2 = len(p2)
    for i in range(n1):
        a1 = p1[i]
        b1 = p1[(i+1) % n1]
        for j in range(n2):
            a2 = p2[j]
            b2 = p2[(j+1) % n2]
            if _segments_intersect(a1, b1, a2, b2):
                return True
    return False

def _segments_intersect(p1, p2, p3, p4):
    """Check if segment p1-p2 intersects segment p3-p4."""
    def cross(o, a, b):
        return (a[0]-o[0])*(b[1]-o[1]) - (a[1]-o[1])*(b[0]-o[0])
    
    c1 = cross(p3, p4, p1)
    c2 = cross(p3, p4, p2)
    c3 = cross(p1, p2, p3)
    c4 = cross(p1, p2, p4)
    
    if ((c1 > 0 and c2 < 0) or (c1 < 0 and c2 > 0)) and \
       ((c3 > 0 and c4 < 0) or (c3 < 0 and c4 > 0)):
        return True
    return False

def _seed_solution():
    """
    Deterministic 'flower' packing: 1 center + 6 around + 4 more.
    This gives a known good starting point.
    """
    A = np.array([np.sqrt(3), 0.0])
    B = np.array([np.sqrt(3)/2, 1.5])
    
    # Flower pattern: center + 6 neighbors
    coords = [(0,0), (1,0), (0,1), (-1,1), (-1,0), (0,-1), (1,-1)]
    # Add 4 more in a ring
    coords += [(2,0), (1,1), (-1,1), (0,-2)]
    
    centers = []
    for i, j in coords:
        c = i * A + j * B
        centers.append((c[0], c[1]))
    
    angles = [0.0] * N
    S = _min_outer_side(centers, angles, 0.0)
    return centers, angles, S

def _objective(x, outer_angle_deg=0.0):
    """
    Objective function for optimization.
    x = [x1, y1, a1, x2, y2, a2, ..., xN, yN, aN] where angles are in radians.
    Returns the minimum outer hexagon side length.
    """
    n = len(x) // 3
    centers = [(x[3*i], x[3*i+1]) for i in range(n)]
    angles_deg = [np.rad2deg(x[3*i+2]) for i in range(n)]
    return _min_outer_side(centers, angles_deg, outer_angle_deg)

def _objective_with_outer(x):
    """
    Objective including outer angle optimization.
    x = [x1, y1, a1, ..., xN, yN, aN, outer_angle_rad]
    """
    n = (len(x) - 1) // 3
    centers = [(x[3*i], x[3*i+1]) for i in range(n)]
    angles_deg = [np.rad2deg(x[3*i+2]) for i in range(n)]
    outer_ang_deg = np.rad2deg(x[-1])
    return _min_outer_side(centers, angles_deg, outer_ang_deg)

def construct():
    start_time = time.time()
    deadline = 145.0  # Safety margin before 150s limit
    
    # Start with the deterministic flower packing
    centers, angles, S = _seed_solution()
    best_S = S
    best_centers = centers[:]
    best_angles = angles[:]
    best_outer_ang = 0.0
    
    # Verify the seed is valid
    if not _check_disjoint(centers, angles):
        # Try adjusting slightly
        pass
    
    # Generate multiple initial configurations
    A = np.array([np.sqrt(3), 0.0])
    B = np.array([np.sqrt(3)/2, 1.5])
    
    # Various topologies to try
    topologies = [
        # Flower + 4
        [(0,0), (1,0), (0,1), (-1,1), (-1,0), (0,-1), (1,-1), (2,0), (1,1), (-1,1), (0,-2)],
        # Flower + different 4
        [(0,0), (1,0), (0,1), (-1,1), (-1,0), (0,-1), (1,-1), (2,0), (1,1), (-1,-1), (0,-2)],
        [(0,0), (1,0), (0,1), (-1,1), (-1,0), (0,-1), (1,-1), (-2,0), (-1,1), (-1,-1), (0,2)],
        [(0,0), (1,0), (0,1), (-1,1), (-1,0), (0,-1), (1,-1), (0,2), (1,1), (1,-1), (0,-2)],
        # More spread out
        [(0,0), (1,0), (0,1), (-1,1), (-1,0), (0,-1), (1,-1), (2,0), (0,2), (-2,0), (0,-2)],
        [(0,0), (1,0), (0,1), (-1,1), (-1,0), (0,-1), (1,-1), (1,1), (1,-1), (-1,1), (-1,-1)],
    ]
    
    initial_configs = []
    
    # Add the seed
    initial_configs.append((best_centers, best_angles, 0.0))
    
    for topo in topologies:
        if time.time() > deadline - 80:
            break
        c_list = []
        for i, j in topo:
            c = i * A + j * B
            c_list.append((c[0], c[1]))
        
        for rot_deg in range(0, 60, 15):
            if time.time() > deadline - 75:
                break
            rot_rad = np.deg2rad(rot_deg)
            cos_r = np.cos(rot_rad)
            sin_r = np.sin(rot_rad)
            rotated = []
            for (x, y) in c_list:
                rx = x * cos_r - y * sin_r
                ry = x * sin_r + y * cos_r
                rotated.append((rx, ry))
            
            for inner_ang in [0, 30, 60]:
                S_val = _min_outer_side(rotated, [inner_ang]*N, 0.0)
                if S_val < best_S:
                    best_S = S_val
                    best_centers = rotated[:]
                    best_angles = [inner_ang]*N
                    best_outer_ang = 0.0
                initial_configs.append((rotated, [inner_ang]*N, 0.0))
    
    # Optimization phase
    # Phase 1: Optimize positions and inner angles (outer angle fixed at 0)
    if time.time() < deadline - 50:
        x0 = []
        for (cx, cy), ang in zip(best_centers, best_angles):
            x0.extend([cx, cy, np.deg2rad(ang)])
        
        try:
            result = minimize(_objective, x0, args=(0.0,), method='Nelder-Mead',
                             options={'maxiter': 1000, 'xatol': 1e-6, 'fatol': 1e-8, 'adaptive': True})
            if result.fun < best_S:
                opt_centers = [(result.x[3*i], result.x[3*i+1]) for i in range(N)]
                opt_angles = [np.rad2deg(result.x[3*i+2]) for i in range(N)]
                if _check_disjoint(opt_centers, opt_angles):
                    best_S = result.fun
                    best_centers = opt_centers
                    best_angles = opt_angles
        except Exception:
            pass
    
    # Phase 2: Optimize including outer angle
    if time.time() < deadline - 40:
        x0_full = []
        for (cx, cy), ang in zip(best_centers, best_angles):
            x0_full.extend([cx, cy, np.deg2rad(ang)])
        x0_full.append(np.deg2rad(best_outer_ang))
        
        try:
            result = minimize(_objective_with_outer, x0_full, method='Nelder-Mead',
                             options={'maxiter': 1000, 'xatol': 1e-6, 'fatol': 1e-8, 'adaptive': True})
            if result.fun < best_S:
                opt_centers = [(result.x[3*i], result.x[3*i+1]) for i in range(N)]
                opt_angles = [np.rad2deg(result.x[3*i+2]) for i in range(N)]
                opt_outer = np.rad2deg(result.x[-1])
                if _check_disjoint(opt_centers, opt_angles):
                    best_S = result.fun
                    best_centers = opt_centers
                    best_angles = opt_angles
                    best_outer_ang = opt_outer
        except Exception:
            pass
    
    # Phase 3: Try other initial configs
    if time.time() < deadline - 30 and len(initial_configs) > 1:
        for cfg in initial_configs[1:8]:
            if time.time() > deadline - 20:
                break
            c, a, oa = cfg
            x0_try = []
            for (cx, cy), ang in zip(c, a):
                x0_try.extend([cx, cy, np.deg2rad(ang)])
            x0_try.append(np.deg2rad(oa))
            
            try:
                result = minimize(_objective_with_outer, x0_try, method='Nelder-Mead',
                                 options={'maxiter': 500, 'xatol': 1e-5, 'fatol': 1e-7, 'adaptive': True})
                if result.fun < best_S:
                    opt_centers = [(result.x[3*i], result.x[3*i+1]) for i in range(N)]
                    opt_angles = [np.rad2deg(result.x[3*i+2]) for i in range(N)]
                    opt_outer = np.rad2deg(result.x[-1])
                    if _check_disjoint(opt_centers, opt_angles):
                        best_S = result.fun
                        best_centers = opt_centers
                        best_angles = opt_angles
                        best_outer_ang = opt_outer
            except Exception:
                pass
    
    # Phase 4: Final high-precision polish
    if time.time() < deadline - 5:
        x0_final = []
        for (cx, cy), ang in zip(best_centers, best_angles):
            x0_final.extend([cx, cy, np.deg2rad(ang)])
        x0_final.append(np.deg2rad(best_outer_ang))
        
        try:
            result = minimize(_objective_with_outer, x0_final, method='Nelder-Mead',
                             options={'maxiter': 2000, 'xatol': 1e-8, 'fatol': 1e-10, 'adaptive': True})
            if result.fun < best_S:
                opt_centers = [(result.x[3*i], result.x[3*i+1]) for i in range(N)]
                opt_angles = [np.rad2deg(result.x[3*i+2]) for i in range(N)]
                opt_outer = np.rad2deg(result.x[-1])
                if _check_disjoint(opt_centers, opt_angles):
                    best_S = result.fun
                    best_centers = opt_centers
                    best_angles = opt_angles
                    best_outer_ang = opt_outer
        except Exception:
            pass
    
    inner = [[c[0], c[1], a] for c, a in zip(best_centers, best_angles)]
    return inner, (0.0, 0.0), best_S, best_outer_ang

# EVOLVE-BLOCK-START
# The construct function is defined above
# EVOLVE-BLOCK-END
