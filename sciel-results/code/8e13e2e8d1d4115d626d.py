import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial import ConvexHull

N = 12
DEADLINE = 130.0  # safety margin before 150s kill

def hex_vertices(cx, cy, angle_deg, side=1.0):
    """Return 6 vertices of a regular hexagon with center (cx,cy), rotation angle_deg, side length."""
    angles = np.deg2rad(angle_deg) + np.deg2rad([0, 60, 120, 180, 240, 300])
    return np.column_stack([cx + side * np.cos(angles), cy + side * np.sin(angles)])

def hexagon_support(center, angle_deg, normals):
    """Support function of a unit hexagon: max dot product of vertex with each normal direction."""
    # normals: (6, 2) array of unit vectors
    angles = np.deg2rad(angle_deg) + np.deg2rad([0, 60, 120, 180, 240, 300])
    verts = np.column_stack([np.cos(angles), np.sin(angles)])
    # support = max over vertices of dot(vert, normal) for each normal
    # This gives the offset from center
    dots = verts @ normals.T  # (6, 6)
    return dots.max(axis=0) + center @ normals.T  # (6,)

def outer_hexagon_support(center, side, angle_deg, normals):
    """Support function of outer hexagon with given center, side length, rotation."""
    angles = np.deg2rad(angle_deg) + np.deg2rad([0, 60, 120, 180, 240, 300])
    verts = side * np.column_stack([np.cos(angles), np.sin(angles)])
    dots = verts @ normals.T
    return dots.max(axis=0) + center @ normals.T

def min_distance_sq(p1, p2):
    """Minimum squared distance between two points."""
    return (p1[0]-p2[0])**2 + (p1[1]-p2[1])**2

def point_in_hexagon_dist(pt, center, angle_deg, side=1.0):
    """Return 0 if point is inside or on hexagon, else distance to nearest edge (squared)."""
    # Transform point to hexagon's local frame
    dx, dy = pt[0] - center[0], pt[1] - center[1]
    theta = -np.deg2rad(angle_deg)
    cos_t, sin_t = np.cos(theta), np.sin(theta)
    lx = dx * cos_t - dy * sin_t
    ly = dx * sin_t + dy * cos_t
    
    # For a regular hexagon with side length s, the inradius is s*sqrt(3)/2
    # The hexagon is the intersection of 6 half-planes
    # Edge normals at angles 0, 60, 120, 180, 240, 300 (local frame)
    inradius = side * np.sqrt(3) / 2.0
    angles = np.deg2rad([0, 60, 120, 180, 240, 300])
    normals = np.column_stack([np.cos(angles), np.sin(angles)])
    
    # For each edge, check if point is outside
    dists = normals @ np.array([lx, ly]) - inradius
    # If all dists <= 0, point is inside
    if np.all(dists <= 1e-10):
        return 0.0
    # Distance to nearest edge (squared)
    # For the violated constraints, the distance is max(dists)
    return max(dists)**2

def check_disjoint(inner, eps=1e-7):
    """Check if inner hexagons are pairwise disjoint. inner: list of (cx, cy, angle_deg)."""
    n = len(inner)
    for i in range(n):
        cx1, cy1, a1 = inner[i]
        for j in range(i+1, n):
            cx2, cy2, a2 = inner[j]
            # Quick check: center distance
            dcx, dcy = cx1-cx2, cy1-cy2
            if dcx*dcx + dcy*dcy < (2.0 - eps)**2:
                # More detailed check needed
                # Check if any vertex of hex i is inside hex j or vice versa
                verts1 = hex_vertices(cx1, cy1, a1)
                verts2 = hex_vertices(cx2, cy2, a2)
                # Check vertices of hex1 against hex2
                for v in verts1:
                    d = point_in_hexagon_dist(v, cx2, cy2, a2)
                    if d < eps*eps:
                        return False
                for v in verts2:
                    d = point_in_hexagon_dist(v, cx1, cy1, a1)
                    if d < eps*eps:
                        return False
    return True

def compute_min_side(inner):
    """Compute minimum outer hexagon side length that can contain all inner hexagons.
    Returns (side, center, angle) of the outer hexagon."""
    # For a given outer hexagon orientation, the minimum side is determined by
    # the maximum support function value needed in each of the 6 directions.
    # We search over outer hexagon angle.
    
    best_side = float('inf')
    best_center = (0.0, 0.0)
    best_angle = 0.0
    
    # Try various outer hexagon orientations
    for outer_angle_deg in np.linspace(0, 60, 13):  # 0 to 60 degrees (hexagon symmetry)
        outer_angle = np.deg2rad(outer_angle_deg)
        # 6 normals for outer hexagon
        norm_angles = outer_angle + np.deg2rad([0, 60, 120, 180, 240, 300])
        normals = np.column_stack([np.cos(norm_angles), np.sin(norm_angles)])
        
        # For each inner hexagon, compute its support in these 6 directions
        # The outer hexagon must satisfy: for each normal direction n_k,
        # max over inner hexagons of support_inner(n_k) <= outer_support(n_k)
        # outer_support(n_k) = side * inradius_outer + center . n_k
        # where inradius_outer = side * sqrt(3)/2
        
        # So: side * sqrt(3)/2 + center . n_k >= max_inner(support_inner_k)
        # For the center, we want to minimize side, so we can center the outer hexagon
        # at the centroid of the inner hexagons' centers, or optimize.
        
        # Actually, for a fixed outer orientation, the optimal center and side can be found.
        # Let's parameterize: the outer hexagon has 6 supporting lines.
        # For each direction k, the outer hexagon's support in direction n_k is:
        #   S_k = side * (sqrt(3)/2) + c . n_k
        # And we need S_k >= h_k where h_k = max over inner hexagons of their support in n_k.
        # Also S_{k+3} = side * (sqrt(3)/2) + c . n_{k+3} = side*(sqrt(3)/2) - c . n_k
        # So S_k + S_{k+3} = 2 * side * sqrt(3)/2 = side * sqrt(3)
        # We need S_k >= h_k and S_{k+3} >= h_{k+3}
        # So side * sqrt(3) >= h_k + h_{k+3} for each k=0,1,2
        # Therefore side >= max over k=0,1,2 of (h_k + h_{k+3}) / sqrt(3)
        
        # Compute h_k for each direction
        h = np.zeros(6)
        for (cx, cy, a_deg) in inner:
            sup = hexagon_support(np.array([cx, cy]), a_deg, normals)
            h = np.maximum(h, sup)
        
        # Minimum side for this orientation
        side_candidate = max((h[0]+h[3], h[1]+h[4], h[2]+h[5])) / np.sqrt(3)
        
        # Center: we need c . n_k = (S_k + h_k - side*sqrt(3))/2 ... 
        # Actually, let's just find a center that works.
        # S_k = side*sqrt(3)/2 + c.n_k >= h_k
        # S_{k+3} = side*sqrt(3)/2 - c.n_k >= h_{k+3}
        # So: c.n_k >= h_k - side*sqrt(3)/2
        #     -c.n_k >= h_{k+3} - side*sqrt(3)/2  =>  c.n_k <= side*sqrt(3)/2 - h_{k+3}
        # So: h_k - side*sqrt(3)/2 <= c.n_k <= side*sqrt(3)/2 - h_{k+3}
        # This is feasible if h_k + h_{k+3} <= side*sqrt(3), which is exactly our condition.
        
        if side_candidate < best_side - 1e-10:
            best_side = side_candidate
            best_angle = outer_angle_deg
            # Find a feasible center
            # For simplicity, try center at origin first, then adjust
            # Actually, let's solve for center using the tightest constraints
            # c.n_0, c.n_1, c.n_2 are determined by 3 equations if we use midpoints
            # Or we can just use c = 0 and check, or optimize.
            # For now, use centroid of inner centers as initial guess
            cx_avg = np.mean([p[0] for p in inner])
            cy_avg = np.mean([p[1] for p in inner])
            best_center = (cx_avg, cy_avg)
    
    # Refine: try to optimize center for the best angle
    if best_side < float('inf'):
        outer_angle = np.deg2rad(best_angle)
        norm_angles = outer_angle + np.deg2rad([0, 60, 120, 180, 240, 300])
        normals = np.column_stack([np.cos(norm_angles), np.sin(norm_angles)])
        
        # For the best angle, optimize center to minimize side
        # Side = max over k=0,1,2 of (h_k(c) + h_{k+3}(c)) / sqrt(3)
        # where h_k(c) = max over inner hexagons of (support_inner_k + c.n_k)
        # Actually h_k doesn't depend on c; it's the max support of inner hexagons in direction n_k.
        # The constraint is: side*sqrt(3)/2 + c.n_k >= h_k for all k
        # To minimize side, we want to choose c such that the max of (h_k - c.n_k) is minimized
        # over k, and side = 2*max_k(h_k - c.n_k) / sqrt(3) ... wait
        # S_k = side*sqrt(3)/2 + c.n_k >= h_k => c.n_k >= h_k - side*sqrt(3)/2
        # S_{k+3} = side*sqrt(3)/2 - c.n_k >= h_{k+3} => c.n_k <= side*sqrt(3)/2 - h_{k+3}
        # Feasible if: h_k - side*sqrt(3)/2 <= side*sqrt(3)/2 - h_{k+3}
        # => h_k + h_{k+3} <= side*sqrt(3)
        # So side = max_k (h_k + h_{k+3})/sqrt(3) is the minimum side, independent of center!
        # The center just needs to satisfy the linear constraints.
        # So the minimum side is indeed independent of center for a fixed outer orientation.
        pass
    
    return best_side, best_center, best_angle

def objective(params, inner_angles):
    """Objective: minimize the outer hexagon side length given inner hexagon positions."""
    # params: flat array of 2*N values: [x1, y1, x2, y2, ..., xN, yN]
    # inner_angles: fixed angles for each inner hexagon
    centers = params.reshape(N, 2)
    inner = [(centers[i,0], centers[i,1], inner_angles[i]) for i in range(N)]
    
    side, _, _ = compute_min_side(inner)
    return side

def construct():
    t0 = time.time()
    
    # Initial configuration: arrange 12 hexagons in a honeycomb-like pattern
    # Hexagons with side 1 have inradius sqrt(3)/2, circumradius 1
    # Center-to-center distance for touching hexagons (same orientation) is 2*sqrt(3)/2 * 2 = 2*sqrt(3) ≈ 3.464
    # Actually, for two unit hexagons side by side with same orientation, center distance = 2 * inradius = 2*sqrt(3)/2 * 2? 
    # No. Inradius = sqrt(3)/2. Two hexagons touching along an edge: center distance = 2 * inradius = sqrt(3) ≈ 1.732? 
    # Let me recalculate: for a regular hexagon with side length s, the inradius is s*sqrt(3)/2.
    # Two hexagons sharing an edge: the distance between centers is 2 * inradius = 2 * s*sqrt(3)/2 = s*sqrt(3).
    # For s=1, that's sqrt(3) ≈ 1.732.
    
    # Let's use a pattern: 3 rows of 4, or 4 rows of 3, in a honeycomb
    # Row spacing (vertical) for honeycomb: sqrt(3)
    # Column spacing: 3/2 (horizontal offset for adjacent in same row)
    # Wait, for unit hexagons:
    # If flat-top: vertices at angles 0,60,120,180,240,300
    # Width = 2 (horizontal extent from -1 to 1)
    # Height = sqrt(3) (vertical extent)
    
    # Let me use pointy-top hexagons (rotation 30 degrees) for easier honeycomb
    # Or just use a grid and optimize.
    
    # Simple initial: 4x3 grid with hexagonal spacing
    # For hexagons with same orientation, minimum center distance to avoid overlap is 2*sqrt(3)/2 * 2?
    # Actually, let's just use a reasonable initial and let optimization work.
    
    # Use a honeycomb pattern with 12 hexagons: 3 in top row, 4 in middle, 3 in bottom? 
    # Or 4-4-4? Let's try a compact arrangement.
    
    # Initial: place in a 3x4 grid with hexagonal offsets
    dx = 1.5  # horizontal spacing for adjacent hexagons
    dy = np.sqrt(3)  # vertical spacing for rows
    
    initial_centers = []
    initial_angles = []
    
    # 3 rows of 4
    for row in range(3):
        for col in range(4):
            x = col * 2.0 + (row % 2) * 1.0
            y = row * dy
            initial_centers.append((x - 1.5, y - dy))  # center the pattern
            initial_angles.append(0.0)
    
    inner_init = initial_centers
    angles_init = initial_angles
    
    # Convert to flat params
    params0 = np.array([c[0] for c in initial_centers] + [c[1] for c in initial_centers])
    
    # First, check initial validity and get baseline
    inner0 = [(initial_centers[i][0], initial_centers[i][1], angles_init[i]) for i in range(N)]
    side0, center0, angle0 = compute_min_side(inner0)
    best_side = side0
    best_params = params0.copy()
    best_center = center0
    best_angle = angle0
    
    elapsed = time.time() - t0
    if elapsed > DEADLINE:
        return [(inner0[i][0], inner0[i][1], inner0[i][2]) for i in range(N)], best_center, best_side, best_angle
    
    # Optimization using Nelder-Mead
    # We optimize positions only, keeping angles fixed at 0 for simplicity
    # (rotations add 12 more variables which may be hard)
    
    def obj_func(params):
        centers = params.reshape(N, 2)
        inner = [(centers[i,0], centers[i,1], angles_init[i]) for i in range(N)]
        side, _, _ = compute_min_side(inner)
        return side
    
    # Try Nelder-Mead optimization
    try:
        res = minimize(obj_func, params0, method='Nelder-Mead',
                       options={'maxiter': 3000, 'xatol': 1e-12, 'fatol': 1e-12, 'adaptive': True})
        
        if res.fun < best_side:
            best_side = res.fun
            best_params = res.x
            centers = res.x.reshape(N, 2)
            inner = [(centers[i,0], centers[i,1], angles_init[i]) for i in range(N)]
            _, best_center, best_angle = compute_min_side(inner)
    except Exception:
        pass
    
    elapsed = time.time() - t0
    if elapsed > DEADLINE - 10:
        centers = best_params.reshape(N, 2)
        inner = [(centers[i,0], centers[i,1], angles_init[i]) for i in range(N)]
        return [(inner[i][0], inner[i][1], inner[i][2]) for i in range(N)], best_center, best_side, best_angle
    
    # Try with different initial angles for outer hexagon to refine
    # Also try allowing some inner hexagons to have different rotations
    # For now, stick with fixed angles but try multiple outer orientations in compute_min_side
    
    # Polish: try to improve with tighter tolerance
    try:
        res2 = minimize(obj_func, best_params, method='Nelder-Mead',
                       options={'maxiter': 5000, 'xatol': 1e-14, 'fatol': 1e-14, 'adaptive': True})
        if res2.fun < best_side:
            best_side = res2.fun
            best_params = res2.x
            centers = res2.x.reshape(N, 2)
            inner = [(centers[i,0], centers[i,1], angles_init[i]) for i in range(N)]
            _, best_center, best_angle = compute_min_side(inner)
    except Exception:
        pass
    
    centers = best_params.reshape(N, 2)
    inner = [(centers[i,0], centers[i,1], angles_init[i]) for i in range(N)]
    return [(inner[i][0], inner[i][1], inner[i][2]) for i in range(N)], best_center, best_side, best_angle
