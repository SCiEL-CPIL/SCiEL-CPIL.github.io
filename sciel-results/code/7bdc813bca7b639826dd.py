import math
import time
import random
import numpy as np
from scipy.optimize import minimize, basinhopping

N = 12
TIME_LIMIT = 140.0  # Safety margin below 150s
INRADIUS = math.sqrt(3) / 2.0

def _get_vertices_vectorized(cx, cy, angles):
    """
    Calculate vertices for N hexagons given centers and angles.
    Returns array of shape (N, 6, 2)
    """
    # angles in radians
    # Base vertices of a unit hexagon centered at origin, angle 0
    # Vertices at 0, 60, 120, 180, 240, 300 degrees
    thetas = np.array([0, 60, 120, 180, 240, 300]) * math.pi / 180.0
    # Base vertices (x, y)
    base_x = np.cos(thetas)
    base_y = np.sin(thetas)
    
    # Rotation matrices for each hexagon
    # cos(a), -sin(a); sin(a), cos(a)
    ca = np.cos(angles)
    sa = np.sin(angles)
    
    # We need to rotate each base vertex for each hexagon
    # Base vertices: (6, 2)
    # Angles: (N,)
    # Result: (N, 6, 2)
    
    # x' = x*cos(a) - y*sin(a)
    # y' = x*sin(a) + y*cos(a)
    
    # base_x is (6,), base_y is (6,)
    # ca is (N,), sa is (N,)
    
    # Expand dims for broadcasting
    bx = base_x[None, :] # (1, 6)
    by = base_y[None, :] # (1, 6)
    ca_exp = ca[:, None] # (N, 1)
    sa_exp = sa[:, None] # (N, 1)
    cx_exp = cx[:, None] # (N, 1)
    cy_exp = cy[:, None] # (N, 1)
    
    vx = cx_exp + bx * ca_exp - by * sa_exp
    vy = cy_exp + bx * sa_exp + by * ca_exp
    
    verts = np.stack([vx, vy], axis=-1) # (N, 6, 2)
    return verts

def _outer_hex_side(verts, center, outer_angle_deg):
    """
    Calculate the side length of the smallest regular hexagon enclosing the vertices.
    The outer hexagon is centered at 'center' and has angle 'outer_angle_deg'.
    """
    a = math.radians(outer_angle_deg)
    # Normals of the outer hexagon sides
    # The sides are perpendicular to the normals pointing to the midpoints of the sides.
    # For a hexagon with flat top (angle 0), normals are at 0, 60, 120, 180, 240, 300?
    # Let's define the outer hexagon by its support function.
    # A regular hexagon with side S and center C can be defined by 6 half-planes.
    # The distance from center to each side (apothem) is S * sqrt(3) / 2.
    # The normals to the sides are at angles: outer_angle + 90, outer_angle + 150, ...
    # Wait, standard hexagon vertices are at 0, 60, ...
    # Sides are between vertices. The normal to the side between 0 and 60 is at 30.
    # So normals are at outer_angle + 30 + 60*k.
    
    normals = []
    for k in range(6):
        na = a + math.radians(30 + 60 * k)
        normals.append((math.cos(na), math.sin(na)))
    
    cx, cy = center
    # Shift vertices relative to center
    verts_rel = verts - np.array([cx, cy])
    
    max_dist = 0.0
    for nx, ny in normals:
        # Project all vertices onto the normal
        # verts_rel is (N, 6, 2)
        # Projection: vx*nx + vy*ny
        proj = verts_rel[..., 0] * nx + verts_rel[..., 1] * ny
        current_max = np.max(proj)
        if current_max > max_dist:
            max_dist = current_max
    
    # max_dist is the apothem of the required outer hexagon
    # Side S = 2 * apothem / sqrt(3)
    return 2.0 * max_dist / math.sqrt(3)

def _collision_penalty(verts, threshold=0.0):
    """
    Calculate penalty for overlapping hexagons.
    Uses a simple distance check between centers first, then a more precise check if needed.
    For regular hexagons, if distance between centers < 2 * inradius, they might overlap.
    Inradius = sqrt(3)/2.
    2 * inradius = sqrt(3) ~ 1.732.
    If distance < 1.732, they are definitely close.
    
    To be robust and smooth, we can use a "soft" collision metric.
    However, for basin hopping, a hard penalty or a very steep penalty works well if the initialization is good.
    Let's use the minimum translation distance (MTD) approximation or just SAT.
    SAT is expensive. 
    Alternative: For regular hexagons, the union of 3 rhombi or just checking the 6 axes of one hexagon against the other.
    
    Let's stick to a fast SAT check for pairs that are close.
    """
    penalty = 0.0
    min_dist_sq = (2 * INRADIUS - 1e-6) ** 2 # If closer than this, check SAT
    
    centers = verts[:, 3, :] # Wait, verts are just vertices. We need centers.
    # We should pass centers separately to avoid extracting them from vertices which is messy.
    # Let's refactor to pass centers and angles.
    
    return penalty # Placeholder

def _objective(params):
    """
    Objective function for optimization.
    params: [x1, y1, a1, ..., xN, yN, aN]
    """
    cx = params[:N]
    cy = params[N:2*N]
    angles_deg = params[2*N:]
    
    # Wrap angles to [0, 60)
    angles_deg = angles_deg % 60.0
    angles_rad = np.radians(angles_deg)
    
    # Get vertices
    verts = _get_vertices_vectorized(cx, cy, angles_rad)
    
    # Calculate center of mass for outer hexagon center
    # Using mean of all vertices is a good approximation for the center of the enclosing hexagon
    all_verts = verts.reshape(-1, 2)
    center = np.mean(all_verts, axis=0)
    
    # Optimize outer hexagon angle
    # The optimal outer angle is likely close to one of the inner hexagon angles or 0
    # Let's try a few angles: 0, 15, 30, 45, 60 (which is 0)
    # Or just 0, since we can rotate the whole system?
    # No, the relative angles matter. But the outer hexagon can rotate freely.
    # The cost function is the side length.
    # For a fixed set of inner hexagons, the minimal enclosing regular hexagon side is minimized
    # when the outer hexagon is aligned with the "widest" projection.
    # We can search for the best outer angle in a small range.
    
    best_side = float('inf')
    # Try angles relative to the first hexagon? Or just 0..60?
    # 60 degrees is the period.
    # We can sample a few angles.
    for oa in [0, 10, 20, 30, 40, 50]:
        side = _outer_hex_side(verts, center, oa)
        if side < best_side:
            best_side = side
    
    # Collision Penalty
    # We need to check if any two hexagons overlap.
    # Fast check: distance between centers
    cx_arr = cx
    cy_arr = cy
    penalty = 0.0
    
    # Pre-compute vertex lists for SAT if needed
    # To save time, we only do SAT if centers are close.
    min_dist_sq = (2 * INRADIUS - 0.1) ** 2 # Buffer
    
    for i in range(N):
        for j in range(i + 1, N):
            dx = cx_arr[i] - cx_arr[j]
            dy = cy_arr[i] - cy_arr[j]
            dist_sq = dx*dx + dy*dy
            if dist_sq < min_dist_sq:
                # Potential overlap, check with SAT
                # Get vertices for i and j
                verts_i = verts[i]
                verts_j = verts[j]
                if _sat_overlap(verts_i, verts_j):
                    # Calculate overlap depth for penalty
                    depth = _sat_overlap_depth(verts_i, verts_j)
                    penalty += depth * depth
                    
    return best_side + 100.0 * penalty

def _sat_overlap(vertsA, vertsB):
    """Check if two convex polygons overlap using SAT. Return True if they overlap."""
    axes = _get_axes(vertsA) + _get_axes(vertsB)
    for axis in axes:
        minA, maxA = _project(vertsA, axis)
        minB, maxB = _project(vertsB, axis)
        if maxA <= minB + 1e-9 or maxB <= minA + 1e-9:
            return False
    return True

def _sat_overlap_depth(vertsA, vertsB):
    """Compute maximum overlap depth between two polygons via SAT."""
    axes = _get_axes(vertsA) + _get_axes(vertsB)
    max_overlap = 0.0
    for axis in axes:
        minA, maxA = _project(vertsA, axis)
        minB, maxB = _project(vertsB, axis)
        overlap = min(maxA, maxB) - max(minA, minB)
        if overlap > max_overlap:
            max_overlap = overlap
    return max_overlap

def _get_axes(verts):
    axes = []
    n = len(verts)
    for i in range(n):
        x1, y1 = verts[i]
        x2, y2 = verts[(i + 1) % n]
        dx, dy = x2 - x1, y2 - y1
        length = math.hypot(dx, dy)
        if length < 1e-12:
            continue
        nx, ny = -dy / length, dx / length
        axes.append((nx, ny))
    return axes

def _project(verts, axis):
    ax, ay = axis
    vals = [vx * ax + vy * ay for vx, vy in verts]
    return min(vals), max(vals)

def _make_honeycomb_init():
    """
    Create a tight honeycomb cluster.
    Center (0,0).
    Ring 1: 6 hexagons at distance 2*sqrt(3) * cos(30) ?
    Distance between centers of touching hexagons is 2 * inradius = sqrt(3).
    Wait, side length is 1. Inradius is sqrt(3)/2.
    Distance between centers of two touching hexagons is 2 * inradius = sqrt(3).
    """
    pts = []
    # Center
    pts.append([0.0, 0.0, 0.0])
    
    # Ring 1: 6 hexagons
    r1 = math.sqrt(3)
    for i in range(6):
        angle = math.radians(60 * i + 30) # Offset by 30 to align with honeycomb
        x = r1 * math.cos(angle)
        y = r1 * math.sin(angle)
        pts.append([x, y, 0.0])
        
    # Ring 2: We need 5 more.
    # In a perfect honeycomb, the next ring has 12 hexagons.
    # We take 5 from the "corners" of the first ring?
    # Or just place them further out.
    r2 = r1 * math.sqrt(3) # Distance to next ring centers in honeycomb
    # The hexagons in the second ring touch the ones in the first ring.
    # Let's place them at angles 0, 60, 120, 180, 240 (5 of them)
    # But we need to ensure they don't overlap with ring 1.
    # In a honeycomb, the distance between a center and a second-ring hexagon is 2 * r1? No.
    # Let's just use a simple lattice.
    
    # Alternative: 3-4-3-2 lattice
    # Row 1: 3
    # Row 2: 4
    # Row 3: 3
    # Row 4: 2
    # This is a common compact packing.
    
    # Let's stick to the honeycomb-like radial placement for the first 7,
    # and place the remaining 5 in the gaps of the second ring.
    
    # Actually, the 3-4-3-2 lattice is very good.
    # Let's generate that.
    pts = []
    dy = math.sqrt(3) / 2 * 2 # Vertical distance between rows = sqrt(3)
    dx = 1.0 # Horizontal distance between adjacent hexagons in same row = 2? No.
    # In a honeycomb lattice:
    # Vector 1: (2, 0) ? No, side 1.
    # Distance between centers horizontally is 2 * inradius? No.
    # If flat-top:
    # Horizontal distance between centers is 2 * inradius? No, it's 2 * (side * cos(30)) = 2 * sqrt(3)/2 = sqrt(3)?
    # Let's check:
    # Hexagon side 1.
    # Width (vertex to vertex) = 2.
    # Height (flat to flat) = sqrt(3).
    # If we stack them flat-top:
    # Row 0: y=0. x = 0, 2, 4... (distance 2)
    # Row 1: y=sqrt(3)/2. x = 1, 3, 5... (distance 2, offset 1)
    # Distance between (0,0) and (1, sqrt(3)/2) is sqrt(1 + 3/4) = sqrt(7/4) != sqrt(3).
    # They don't touch.
    
    # To make them touch:
    # Distance must be sqrt(3).
    # If dx = 1, dy = sqrt(3)/2. dist = sqrt(1 + 3/4) = 1.32. Too close.
    # If dx = 1, dy = 0.866 (sqrt(3)/2).
    # We need dx^2 + dy^2 = 3.
    # If dy = sqrt(3)/2, dx^2 = 3 - 3/4 = 9/4 => dx = 1.5.
    # So, horizontal spacing is 1.5, vertical is sqrt(3)/2?
    # Let's verify.
    # Hexagon at (0,0).
    # Hexagon at (1.5, sqrt(3)/2).
    # Distance = sqrt(2.25 + 0.75) = sqrt(3). Correct.
    
    # So, lattice vectors:
    # v1 = (1.5, sqrt(3)/2)
    # v2 = (0, sqrt(3)) ? No, that's not a lattice.
    # Standard honeycomb lattice for hexagons:
    # Basis:
    # a1 = (3/2, sqrt(3)/2)
    # a2 = (0, sqrt(3))
    # Or simply:
    # Row 0: y=0, x = 0, 3, 6... (step 3? No)
    # Let's use the 3-4-3-2 grid with correct spacing.
    
    # Spacing:
    # Horizontal step within row: 3.0? No.
    # If I place hexagons at (0,0) and (3,0), do they touch?
    # Distance 3. Inradius sum = sqrt(3) ~ 1.732. Gap = 3 - 1.732.
    # To touch, distance must be sqrt(3).
    # So horizontal step is sqrt(3) * 2? No.
    # If they are in the same row (flat top), they touch if distance is 2 * inradius?
    # No, flat-top hexagons touch side-to-side.
    # Side length 1.
    # Distance between centers of two touching flat-top hexagons is 2 * inradius = sqrt(3).
    
    # So, x-step = sqrt(3).
    # y-step = 1.5?
    # Let's check (0,0) and (sqrt(3)/2, 1.5).
    # dist^2 = 3/4 + 2.25 = 3.0. dist = sqrt(3).
    # So, dx = sqrt(3)/2, dy = 1.5.
    
    # Let's build 3-4-3-2.
    # Row 0 (3 hex): y = -1.5
    # Row 1 (4 hex): y = 0
    # Row 2 (3 hex): y = 1.5
    # Row 3 (2 hex): y = 3.0
    
    # X positions:
    # Row 0 (3): centered. x = -sqrt(3)/2, 0, sqrt(3)/2
    # Row 1 (4): x = -sqrt(3), -sqrt(3)/2, sqrt(3)/2, sqrt(3)
    # Row 2 (3): x = -sqrt(3)/2, 0, sqrt(3)/2
    # Row 3 (2): x = -sqrt(3)/2, sqrt(3)/2 ? Or -sqrt(3), 0?
    # Let's center them.
    
    pts = []
    # Row 0
    for i in range(3):
        x = (i - 1) * math.sqrt(3)
        y = -1.5
        pts.append([x, y, 0.0])
        
    # Row 1
    for i in range(4):
        x = (i - 1.5) * math.sqrt(3)
        y = 0.0
        pts.append([x, y, 0.0])
        
    # Row 2
    for i in range(3):
        x = (i - 1) * math.sqrt(3)
        y = 1.5
        pts.append([x, y, 0.0])
        
    # Row 3
    for i in range(2):
        x = (i - 0.5) * math.sqrt(3)
        y = 3.0
        pts.append([x, y, 0.0])
        
    return pts

def construct():
    t0 = time.time()
    random.seed(42)
    np.random.seed(42)
    
    init_pts = _make_honeycomb_init()
    
    # Flatten to params
    params0 = np.array([p[0] for p in init_pts] + [p[1] for p in init_pts] + [p[2] for p in init_pts])
    
    # Initial evaluation
    current_best = _objective(params0)
    best_params = params0.copy()
    
    # Basin Hopping
    # We need a step function
    def step(x):
        # Random perturbation
        step_size = 0.5
        return x + step_size * np.random.randn(len(x))
    
    # We will run basin hopping for a limited time
    # SciPy's basinhopping can be slow. We will implement a simple loop.
    
    current_params = best_params.copy()
    current_score = current_best
    
    while time.time() - t0 < TIME_LIMIT:
        # Local minimization using Nelder-Mead
        try:
            res = minimize(_objective, current_params, method='Nelder-Mead', 
                           options={'maxiter': 200, 'xatol': 1e-4, 'fatol': 1e-4})
            if res.fun < current_score:
                current_score = res.fun
                current_params = res.x
                best_params = res.x.copy()
                best_score = res.fun
            else:
                # If it didn't improve, maybe it got stuck
                pass
        except:
            break
            
        # Check time
        if time.time() - t0 > TIME_LIMIT - 10:
            break
            
        # Global move (Basin Hopping step)
        # Perturb the best found so far
        step_size = 0.2
        perturbed = best_params + step_size * np.random.randn(len(best_params))
        
        # Optional: Wrap angles in perturbed? The objective handles it.
        
        # Try to minimize from this new point
        # We can just set current_params to perturbed and let the next Nelder-Mead run handle it
        current_params = perturbed
        
        # To avoid getting stuck in the same basin, we can increase step size if no improvement
        # But for now, fixed step is okay.
        
    # Final Polish
    # Run a tighter Nelder-Mead on the best parameters found
    try:
        res = minimize(_objective, best_params, method='Nelder-Mead', 
                       options={'maxiter': 1000, 'xatol': 1e-6, 'fatol': 1e-6})
        if res.fun < best_score:
            best_params = res.x
            best_score = res.fun
    except:
        pass
        
    # Extract results
    cx = best_params[:N]
    cy = best_params[N:2*N]
    angles_deg = best_params[2*N:]
    angles_deg = angles_deg % 60.0
    
    # Calculate final outer hexagon
    verts = _get_vertices_vectorized(cx, cy, np.radians(angles_deg))
    all_verts = verts.reshape(-1, 2)
    center = np.mean(all_verts, axis=0)
    
    # Find best outer angle
    best_side = float('inf')
    best_outer_angle = 0
    for oa in np.arange(0, 60, 1.0):
        side = _outer_hex_side(verts, center, oa)
        if side < best_side:
            best_side = side
            best_outer_angle = oa
            
    # Refine outer angle
    for oa in np.arange(best_outer_angle - 1, best_outer_angle + 1, 0.1):
        side = _outer_hex_side(verts, center, oa)
        if side < best_side:
            best_side = side
            best_outer_angle = oa

    inner = [[cx[i], cy[i], angles_deg[i]] for i in range(N)]
    return inner, (center[0], center[1]), best_side, best_outer_angle

