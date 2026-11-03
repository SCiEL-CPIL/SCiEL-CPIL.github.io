"""
Packing 12 unit regular hexagons inside a regular hexagon.
Minimizes the outer side length.
"""
import time
import numpy as np
from scipy.spatial import ConvexHull
from scipy.optimize import minimize
import random

N = 12

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin
    
    # --- Geometry Helpers ---
    
    def get_hex_vertices(cx, cy, side, angle_deg):
        angle_rad = np.deg2rad(angle_deg)
        # Vertices of a regular hexagon with side length 'side'
        # Centered at (cx, cy), rotated by angle_rad
        # Standard orientation: flat top/bottom? Pointy top/bottom?
        # Let's use standard math convention: vertex at angle 0 is (side, 0) relative to center
        # But usually "regular hexagon" in packing problems might imply specific alignment.
        # The problem says "regular hexagons". Side length 1.
        # Vertices are at angles 0, 60, 120, 180, 240, 300 degrees from center?
        # Distance from center to vertex is side length (circumradius = side).
        angles = np.deg2rad([0, 60, 120, 180, 240, 300]) + angle_rad
        x = cx + side * np.cos(angles)
        y = cy + side * np.sin(angles)
        return np.column_stack((x, y))

    def get_hex_inradius(side):
        return side * np.sqrt(3) / 2.0

    def get_hex_apothem(side):
        return side * np.sqrt(3) / 2.0

    def get_hex_circumradius(side):
        return side

    def point_in_hexagon(pt, center, side, angle_deg):
        # Check if point (x,y) is inside hexagon centered at center with given side and angle
        # Transform point to hexagon local frame
        x, y = pt
        cx, cy = center
        # Rotate by -angle
        angle_rad = -np.deg2rad(angle_deg)
        cos_a = np.cos(angle_rad)
        sin_a = np.sin(angle_rad)
        dx = x - cx
        dy = y - cy
        local_x = dx * cos_a - dy * sin_a
        local_y = dx * sin_a + dy * cos_a
        
        # In local frame, hexagon vertices are at (side*cos(k*60), side*sin(k*60))
        # The hexagon is the intersection of 6 half-planes.
        # For a hexagon with circumradius R=side, the inradius (apothem) is r = side * sqrt(3)/2.
        # The boundaries are at distance r from the center, perpendicular to the axes at 30, 90, ... degrees?
        # Let's define the 6 normal vectors.
        # Vertices at 0, 60, ...
        # Edges are between vertices.
        # Normal to edge between v0 and v1 (at 30 deg) points at 30 deg.
        # Distance from center to edge is r = side * cos(30) = side * sqrt(3)/2.
        # So constraints: dot(p, n_i) <= r for i=0..5
        # Normals are at 30, 90, 150, 210, 270, 330 degrees.
        
        normals = np.deg2rad([30, 90, 150, 210, 270, 330])
        nx = np.cos(normals)
        ny = np.sin(normals)
        
        r = side * np.sqrt(3) / 2.0
        
        dots = local_x * nx + local_y * ny
        return np.all(dots <= r + 1e-9)

    def hexes_overlap_sat(h1, h2):
        # h1, h2: (cx, cy, side, angle_deg)
        # Use Separating Axis Theorem for two convex polygons.
        # We can get vertices for both.
        v1 = get_hex_vertices(h1[0], h1[1], h1[2], h1[3])
        v2 = get_hex_vertices(h2[0], h2[1], h2[2], h2[3])
        
        # Collect all edges from both polygons
        edges = []
        for v in [v1, v2]:
            for i in range(len(v)):
                p1 = v[i]
                p2 = v[(i+1)%len(v)]
                edge = p2 - p1
                # Normal
                length = np.linalg.norm(edge)
                if length < 1e-9: continue
                normal = np.array([-edge[1], edge[0]]) / length
                edges.append(normal)
        
        # Project both polygons onto each normal
        for normal in edges:
            proj1 = v1 @ normal
            proj2 = v2 @ normal
            min1, max1 = np.min(proj1), np.max(proj1)
            min2, max2 = np.min(proj2), np.max(proj2)
            
            # Check for gap
            if max1 < min2 - 1e-9 or max2 < min1 - 1e-9:
                return False # Separated
        
        return True # Overlapping

    def compute_outer_side(inner, outer_center, outer_angle):
        # Given positions of inner hexes, find minimum outer hex side length
        # such that all inner hexes fit inside.
        # We can treat this as: Find S such that for all inner hexes,
        # their vertices are inside the outer hex.
        # Or more efficiently, compute the "extent" of the union of inner hexes
        # along the 6 normal directions of the outer hex.
        
        # Outer hex is centered at outer_center, rotated by outer_angle.
        # Its sides are perpendicular to normals at angles: outer_angle + 30, 90, ...
        # Let's define the 6 normals for the outer hex.
        # If outer hex has angle theta, its vertices are at theta, theta+60...
        # Its edges are perpendicular to directions theta+30, theta+90...
        
        normals = np.deg2rad([30, 90, 150, 210, 270, 330]) + np.deg2rad(outer_angle)
        nx = np.cos(normals)
        ny = np.sin(normals)
        
        # For each inner hex, we need the maximum projection of its vertices
        # onto each normal direction, relative to the outer center.
        # The outer hex constraint for a point p is:
        # (p - c_outer) dot n_i <= S * sqrt(3)/2  (where S is side length)
        # Wait, for a hex with side S, the apothem is S*sqrt(3)/2.
        # So the boundary is at distance S*sqrt(3)/2 from center along the normals.
        
        # We need: max_{p in inner_i} (p - c_outer) dot n_i <= S * sqrt(3)/2
        # So S >= max_{i, k} ( (p_{i,k} - c_outer) dot n_i ) / (sqrt(3)/2)
        
        max_extent = 0.0
        for hx, hy, ha in inner:
            verts = get_hex_vertices(hx, hy, 1.0, ha)
            for i in range(6):
                # Project vertices onto normal i
                # p dot n_i - c_outer dot n_i
                # Max of (p dot n_i) - c_outer dot n_i
                proj = verts @ np.array([nx[i], ny[i]])
                max_proj = np.max(proj)
                outer_proj = outer_center[0]*nx[i] + outer_center[1]*ny[i]
                extent = max_proj - outer_proj
                if extent > max_extent:
                    max_extent = extent
        
        # S * sqrt(3)/2 >= max_extent
        # S >= max_extent / (sqrt(3)/2)
        return max_extent / (np.sqrt(3)/2.0)

    def total_cost(inner, outer_center, outer_angle):
        # We want to minimize the outer side length.
        # We can also optimize the outer center and angle to minimize this.
        # But let's first fix outer center at (0,0) and angle at 0 for simplicity?
        # No, we should allow outer hex to rotate and translate to best fit the cluster.
        # However, the problem asks to return the outer center and side.
        # We can optimize the inner positions AND the outer hex parameters.
        
        # For now, let's assume outer hex is centered at (0,0) and aligned with axes (angle 0).
        # We can refine this later.
        return compute_outer_side(inner, np.array(outer_center), outer_angle)

    def objective(params):
        # params: [x1, y1, a1, x2, y2, a2, ...]
        # angles in radians
        inner = []
        for i in range(N):
            x = params[i*3]
            y = params[i*3+1]
            a = params[i*3+2]
            inner.append([x, y, np.rad2deg(a)])
        
        # Outer hex center (0,0), angle 0
        outer_center = (0.0, 0.0)
        outer_angle = 0.0
        
        side = compute_outer_side(inner, outer_center, outer_angle)
        
        # Add penalty for overlaps
        penalty = 0.0
        for i in range(N):
            for j in range(i+1, N):
                h1 = (inner[i][0], inner[i][1], 1.0, inner[i][2])
                h2 = (inner[j][0], inner[j][1], 1.0, inner[j][2])
                if hexes_overlap_sat(h1, h2):
                    penalty += 10.0
        
        return side + penalty

    # --- Initial Guess ---
    # Arrange in a honeycomb pattern
    # Side length 1.
    # Horizontal distance between centers in same row: 2 * inradius = sqrt(3)
    # Vertical distance between rows: 1.5 * side = 1.5
    # Offset for alternating rows: sqrt(3)/2
    
    dx = np.sqrt(3)
    dy = 1.5
    offset = dx / 2.0
    
    init_params = np.zeros(N*3)
    
    # 4 rows of 3 hexes? Or 3 rows of 4?
    # Let's try a compact cluster.
    # Row 0: 3 hexes
    # Row 1: 4 hexes
    # Row 2: 3 hexes
    # Row 3: 2 hexes
    # Total 12.
    
    positions = []
    # Row 0 (y=0)
    for k in range(3):
        x = (k - 1) * dx
        y = 0.0
        positions.append([x, y, 0.0])
    # Row 1 (y=dy)
    for k in range(4):
        x = (k - 1.5) * dx
        y = dy
        positions.append([x, y, 0.0])
    # Row 2 (y=2*dy)
    for k in range(3):
        x = (k - 1) * dx
        y = 2*dy
        positions.append([x, y, 0.0])
    # Row 3 (y=3*dy)
    for k in range(2):
        x = (k - 0.5) * dx
        y = 3*dy
        positions.append([x, y, 0.0])
        
    # Center the cluster
    cx = np.mean([p[0] for p in positions])
    cy = np.mean([p[1] for p in positions])
    for p in positions:
        p[0] -= cx
        p[1] -= cy
        
    for i in range(N):
        init_params[i*3] = positions[i][0]
        init_params[i*3+1] = positions[i][1]
        init_params[i*3+2] = positions[i][2] # 0 radians

    # --- Optimization ---
    # Use L-BFGS-B
    # This might be slow if we call SAT for every evaluation.
    # Let's try a few restarts.
    
    best_result = None
    best_score = float('inf')
    
    num_restarts = 5
    
    for r in range(num_restarts):
        if time.time() > deadline - 20:
            break
            
        # Perturb initial guess
        if r == 0:
            x0 = init_params.copy()
        else:
            x0 = init_params + np.random.normal(0, 0.5, size=N*3)
            # Randomize angles slightly
            x0[2::3] += np.random.normal(0, np.pi/6, size=N)
            
        try:
            res = minimize(objective, x0, method='L-BFGS-B', options={'maxiter': 500, 'disp': False})
            
            if res.success or res.fun < best_score:
                # Extract inner
                inner = []
                for i in range(N):
                    x = res.x[i*3]
                    y = res.x[i*3+1]
                    a = res.x[i*3+2]
                    inner.append([x, y, np.rad2deg(a)])
                
                # Check validity (no overlaps)
                valid = True
                for i in range(N):
                    for j in range(i+1, N):
                        h1 = (inner[i][0], inner[i][1], 1.0, inner[i][2])
                        h2 = (inner[j][0], inner[j][1], 1.0, inner[j][2])
                        if hexes_overlap_sat(h1, h2):
                            valid = False
                            break
                    if not valid: break
                
                if valid:
                    side = compute_outer_side(inner, (0.0, 0.0), 0.0)
                    if side < best_score:
                        best_score = side
                        best_result = (inner, (0.0, 0.0), side, 0.0)
                        
        except Exception as e:
            pass

    if best_result is not None:
        return best_result
    else:
        # Fallback to initial guess
        inner = []
        for i in range(N):
            inner.append([positions[i][0], positions[i][1], 0.0])
        side = compute_outer_side(inner, (0.0, 0.0), 0.0)
        return inner, (0.0, 0.0), side, 0.0

# EVOLVE-BLOCK-START
# The function above is the main logic.
# We need to wrap it in the expected structure.
# The prompt says: "Keep the function names and signatures of the seed program."
# Seed: def construct(): ...
# So I will define construct() to call the logic above.
# But the logic above is defined inside construct? No, I defined helpers inside construct?
# No, I defined helpers inside the module scope in my thought process, but in the code block I will define them inside construct or outside?
# Defining inside construct is slower due to redefinition. Let's define them outside.

def _get_hex_vertices(cx, cy, side, angle_deg):
    angle_rad = np.deg2rad(angle_deg)
    angles = np.deg2rad([0, 60, 120, 180, 240, 300]) + angle_rad
    x = cx + side * np.cos(angles)
    y = cy + side * np.sin(angles)
    return np.column_stack((x, y))

def _hexes_overlap_sat(h1, h2):
    v1 = _get_hex_vertices(h1[0], h1[1], h1[2], h1[3])
    v2 = _get_hex_vertices(h2[0], h2[1], h2[2], h2[3])
    
    edges = []
    for v in [v1, v2]:
        for i in range(len(v)):
            p1 = v[i]
            p2 = v[(i+1)%len(v)]
            edge = p2 - p1
            length = np.linalg.norm(edge)
            if length < 1e-9: continue
            normal = np.array([-edge[1], edge[0]]) / length
            edges.append(normal)
    
    for normal in edges:
        proj1 = v1 @ normal
        proj2 = v2 @ normal
        min1, max1 = np.min(proj1), np.max(proj1)
        min2, max2 = np.min(proj2), np.max(proj2)
        
        if max1 < min2 - 1e-9 or max2 < min1 - 1e-9:
            return False
    
    return True

def _compute_outer_side(inner, outer_center, outer_angle):
    normals = np.deg2rad([30, 90, 150, 210, 270, 330]) + np.deg2rad(outer_angle)
    nx = np.cos(normals)
    ny = np.sin(normals)
    
    max_extent = 0.0
    for hx, hy, ha in inner:
        verts = _get_hex_vertices(hx, hy, 1.0, ha)
        for i in range(6):
            proj = verts @ np.array([nx[i], ny[i]])
            max_proj = np.max(proj)
            outer_proj = outer_center[0]*nx[i] + outer_center[1]*ny[i]
            extent = max_proj - outer_proj
            if extent > max_extent:
                max_extent = extent
    
    return max_extent / (np.sqrt(3)/2.0)

def _objective(params, N):
    inner = []
    for i in range(N):
        x = params[i*3]
        y = params[i*3+1]
        a = params[i*3+2]
        inner.append([x, y, np.rad2deg(a)])
    
    outer_center = (0.0, 0.0)
    outer_angle = 0.0
    
    side = _compute_outer_side(inner, outer_center, outer_angle)
    
    penalty = 0.0
    for i in range(N):
        for j in range(i+1, N):
            h1 = (inner[i][0], inner[i][1], 1.0, inner[i][2])
            h2 = (inner[j][0], inner[j][1], 1.0, inner[j][2])
            if _hexes_overlap_sat(h1, h2):
                penalty += 10.0
    
    return side + penalty

def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    
    N = 12
    dx = np.sqrt(3)
    dy = 1.5
    
    # Initial guess: Honeycomb
    positions = []
    # Row 0: 3
    for k in range(3):
        positions.append([(k - 1) * dx, 0.0, 0.0])
    # Row 1: 4
    for k in range(4):
        positions.append([(k - 1.5) * dx, dy, 0.0])
    # Row 2: 3
    for k in range(3):
        positions.append([(k - 1) * dx, 2*dy, 0.0])
    # Row 3: 2
    for k in range(2):
        positions.append([(k - 0.5) * dx, 3*dy, 0.0])
        
    cx = np.mean([p[0] for p in positions])
    cy = np.mean([p[1] for p in positions])
    for p in positions:
        p[0] -= cx
        p[1] -= cy
        
    init_params = np.zeros(N*3)
    for i in range(N):
        init_params[i*3] = positions[i][0]
        init_params[i*3+1] = positions[i][1]
        init_params[i*3+2] = 0.0

    best_result = None
    best_score = float('inf')
    
    # Run optimization
    for r in range(5):
        if time.time() > deadline - 20:
            break
            
        if r == 0:
            x0 = init_params.copy()
        else:
            x0 = init_params + np.random.normal(0, 0.5, size=N*3)
            x0[2::3] += np.random.normal(0, np.pi/6, size=N)
            
        try:
            res = minimize(lambda p: _objective(p, N), x0, method='L-BFGS-B', options={'maxiter': 1000, 'disp': False})
            
            inner = []
            for i in range(N):
                x = res.x[i*3]
                y = res.x[i*3+1]
                a = res.x[i*3+2]
                inner.append([x, y, np.rad2deg(a)])
                
            valid = True
            for i in range(N):
                for j in range(i+1, N):
                    h1 = (inner[i][0], inner[i][1], 1.0, inner[i][2])
                    h2 = (inner[j][0], inner[j][1], 1.0, inner[j][2])
                    if _hexes_overlap_sat(h1, h2):
                        valid = False
                        break
                if not valid: break
            
            if valid:
                side = _compute_outer_side(inner, (0.0, 0.0), 0.0)
                if side < best_score:
                    best_score = side
                    best_result = (inner, (0.0, 0.0), side, 0.0)
        except Exception:
            pass

    if best_result is not None:
        return best_result
    else:
        inner = []
        for i in range(N):
            inner.append([positions[i][0], positions[i][1], 0.0])
        side = _compute_outer_side(inner, (0.0, 0.0), 0.0)
        return inner, (0.0, 0.0), side, 0.0
# EVOLVE-BLOCK-END
