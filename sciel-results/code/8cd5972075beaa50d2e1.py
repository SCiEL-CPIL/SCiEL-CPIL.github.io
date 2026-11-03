"""Improved packing of 11 unit hexagons in a regular hexagon."""
import numpy as np
import time
from scipy.optimize import minimize

N = 11
DEADLINE = time.time() + 125.0  # safety margin before 150s kill

def hexagon_vertices(cx, cy, angle_deg, side=1.0):
    """Return vertices of a regular hexagon centered at (cx,cy) with given rotation."""
    a = np.radians(angle_deg)
    # Standard hexagon: flat-top orientation has vertices at angles 0,60,120,180,240,300
    # Pointy-top: 30,90,150,210,270,330
    # We use flat-top: vertex i at angle = a + i*60 degrees
    angles = a + np.radians(np.arange(6) * 60)
    return np.column_stack([cx + side * np.cos(angles), cy + side * np.sin(angles)])

def convex_hull(points):
    """Simple convex hull via scipy if available, else Andrew's monotone chain."""
    try:
        from scipy.spatial import ConvexHull
        if len(points) < 3:
            return points
        hull = ConvexHull(points)
        return points[hull.vertices]
    except Exception:
        # Fallback: Andrew's monotone chain
        pts = sorted(set(map(tuple, points)))
        if len(pts) <= 1:
            return np.array(pts)
        def cross(o, a, b):
            return (a[0]-o[0])*(b[1]-o[1]) - (a[1]-o[1])*(b[0]-o[0])
        lower = []
        for p in pts:
            while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
                lower.pop()
            lower.append(p)
        upper = []
        for p in reversed(pts):
            while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
                upper.pop()
            upper.append(p)
        hull = lower[:-1] + upper[:-1]
        return np.array(hull)

def outer_hexagon_containing(hull_pts, angle_deg=0.0):
    """Find minimal side length of a regular hexagon (with given rotation) containing hull_pts.
    Returns (side_length, center_x, center_y)."""
    a = np.radians(angle_deg)
    # A regular hexagon with side s, centered at (cx,cy), rotated by a,
    # has support function: for direction u, max over vertices of dot(v - c, u) <= s * cos(30 deg) = s * sqrt(3)/2
    # Actually, the support function of a regular hexagon (flat-top, side s) in direction theta:
    # h(theta) = s * cos(30 deg) * |cos(3*theta_mod_60)| ... let me think more carefully.
    
    # A regular hexagon with side length s has inradius r = s * sqrt(3)/2.
    # The support function in direction u is: h(u) = r * max_{i=0..5} cos(angle(u) - angle_i)
    # where angle_i are the angles to the 6 edge normals.
    
    # For a flat-top hexagon (vertex at angle 0), the edge normals are at 30, 90, 150, 210, 270, 330 degrees.
    # With rotation a, edge normals are at a + 30 + i*60 degrees.
    
    # The constraint is: for all points p in hull, and for each edge normal direction n_i:
    #   dot(p - c, n_i) <= r
    #   dot(c - p, n_i) <= r  (i.e., dot(p-c, n_i) >= -r)
    
    # This is equivalent to: c must satisfy:
    #   dot(c, n_i) >= max_p dot(p, n_i) - r  for all i
    #   dot(c, n_i) <= min_p dot(p, n_i) + r  for all i
    
    # For a given r (hence s = 2r/sqrt(3)), we need to find if there exists c satisfying all 12 inequalities.
    # The minimum r is found by binary search or by solving the LP.
    
    # Let's use a simpler approach: the minimal enclosing regular hexagon.
    # We can parameterize by center (cx, cy) and find the minimal r such that all points are inside.
    # For fixed center, r_min = max over all points p and all edge normals n_i of dot(p-c, n_i).
    # Then we minimize r_min over (cx, cy).
    
    # Edge normal directions (unit vectors):
    base_angles = np.radians(30 + np.arange(6) * 60)
    normals = np.column_stack([np.cos(a + base_angles), np.sin(a + base_angles)])  # shape (6,2)
    
    # For center c, the required r is max_i max_p dot(p - c, n_i) = max_i (max_p dot(p, n_i) - dot(c, n_i))
    # We want to minimize this over c.
    
    # This is a convex problem: minimize max_i (h_i - dot(c, n_i)) where h_i = max_p dot(p, n_i)
    # Using scipy minimize with L-BFGS-B or SLSQP.
    
    h_vals = np.array([np.max(hull_pts @ n) for n in normals])  # shape (6,)
    
    def objective(c):
        cx, cy = c
        vals = h_vals - (cx * normals[:, 0] + cy * normals[:, 1])
        return np.max(vals)
    
    # Initial guess: centroid of hull
    c0 = np.array([np.mean(hull_pts[:, 0]), np.mean(hull_pts[:, 1])])
    
    result = minimize(objective, c0, method='Nelder-Mead', 
                      options={'maxiter': 500, 'xatol': 1e-8, 'fatol': 1e-8})
    c_opt = result.x
    r_opt = objective(c_opt)
    
    # Side length s = 2r / sqrt(3)
    s = 2 * r_opt / np.sqrt(3)
    
    return s, c_opt[0], c_opt[1]

def check_validity(inner, outer_center, side, angle_deg, tol=1e-6):
    """Check if all inner hexagons are disjoint and inside the outer hexagon."""
    ox, oy = outer_center
    outer_verts = hexagon_vertices(ox, oy, angle_deg, side)
    
    # Check containment: all vertices of inner hexagons must be inside outer hexagon
    for (ix, iy, iangle) in inner:
        iv = hexagon_vertices(ix, iy, iangle, 1.0)
        # Point in convex polygon test using cross products
        for j in range(6):
            p1 = outer_verts[j]
            p2 = outer_verts[(j + 1) % 6]
            for v in iv:
                cross_val = (p2[0]-p1[0])*(v[1]-p1[1]) - (p2[1]-p1[1])*(v[0]-p1[0])
                if cross_val < -tol:
                    return False
    return True

def compute_side_length(inner, angle_deg=0.0):
    """Compute minimal outer hexagon side length for given inner positions."""
    all_pts = []
    for (ix, iy, iangle) in inner:
        v = hexagon_vertices(ix, iy, iangle, 1.0)
        all_pts.append(v)
    all_pts = np.vstack(all_pts)
    hull = convex_hull(all_pts)
    s, cx, cy = outer_hexagon_containing(hull, angle_deg)
    return s, cx, cy

def construct():
    start_time = time.time()
    
    # Strategy: Start with a hexagonal lattice arrangement of 11 hexagons,
    # then optimize positions to minimize the outer hexagon side length.
    
    # Hexagonal lattice: central hexagon + 6 around + 4 more in second ring
    # Actually, 11 hexagons: 1 center, 6 in first ring, 4 in second ring (partial)
    # Or: rows of 3, 4, 3, 1? Let's use a compact arrangement.
    
    # For unit hexagons in a hexagonal lattice (flat-top), the spacing is:
    # Horizontal distance between adjacent columns: 3/2 * side = 1.5
    # Vertical distance: sqrt(3) * side = sqrt(3)
    # Adjacent in same row: 3 * side = 3 (no, that's wrong)
    
    # For flat-top hexagons with side 1:
    # Width = 2, Height = sqrt(3)
    # Horizontal spacing between centers in same row: 3 (since width is 2, gap is 1)
    # Actually for touching: horizontal distance = 3/2 * 2 = 3? No.
    # Flat-top hex: vertices at (1,0), (0.5, sqrt(3)/2), (-0.5, sqrt(3)/2), (-1,0), (-0.5,-sqrt(3)/2), (0.5,-sqrt(3)/2)
    # Width = 2, Height = sqrt(3)
    # To tile: horizontal spacing = 3/2 * width = 3? No, = 1.5 * 2 = 3. 
    # Wait, in a hex grid, adjacent centers are at distance sqrt(3) apart for pointy-top,
    # or 3/2 apart horizontally for flat-top with vertical offset.
    
    # Let me just use pointy-top hexagons for easier lattice.
    # Pointy-top hex with side 1: vertices at angles 30,90,150,210,270,330
    # Width = sqrt(3), Height = 2
    # In a hex grid: 
    #   Same row: horizontal spacing = sqrt(3)
    #   Adjacent rows: horizontal offset = sqrt(3)/2, vertical spacing = 1.5
    
    # Let's use flat-top for consistency with the outer hexagon.
    # Flat-top hex vertices (side=1): 
    #   (1, 0), (0.5, sqrt(3)/2), (-0.5, sqrt(3)/2), (-1, 0), (-0.5, -sqrt(3)/2), (0.5, -sqrt(3)/2)
    
    # For a hexagonal packing of flat-top unit hexagons:
    # Centers in a hex grid:
    #   Row 0: (0, 0)
    #   Row 1: (3/2, sqrt(3)/2), (3/2, -sqrt(3)/2)  -- wait, this isn't right either.
    
    # Let me just place them in a simple pattern and optimize.
    # Start with a roughly circular cluster.
    
    np.random.seed(42)
    
    # Initial positions: hexagonal lattice
    # For flat-top hexagons, the lattice vectors are:
    # v1 = (3, 0)  [horizontal neighbor]
    # v2 = (1.5, 3*sqrt(3)/2)  [diagonal neighbor]
    # Wait, let me verify. Two flat-top hexagons touching side by side:
    # Left hex center at (0,0), right hex center at (3, 0)? 
    # Left hex rightmost point: (1, 0). Right hex leftmost point: (cx-1, 0).
    # For them to touch: cx - 1 = 1 => cx = 2. No wait, they share an edge.
    # The right edge of left hex goes from (0.5, sqrt(3)/2) to (0.5, -sqrt(3)/2).
    # The left edge of right hex (centered at (dx, 0)) goes from (dx-0.5, sqrt(3)/2) to (dx-0.5, -sqrt(3)/2).
    # For shared edge: dx - 0.5 = 0.5 => dx = 1. 
    # Hmm, that doesn't seem right. Let me reconsider.
    
    # Actually for flat-top hex with side 1, the vertices are:
    # v0 = (1, 0)
    # v1 = (0.5, sqrt(3)/2)  
    # v2 = (-0.5, sqrt(3)/2)
    # v3 = (-1, 0)
    # v4 = (-0.5, -sqrt(3)/2)
    # v5 = (0.5, -sqrt(3)/2)
    
    # The right side is the edge from v0 to v5: from (1,0) to (0.5, -sqrt(3)/2).
    # A hexagon to the right sharing this edge would have its left side on this edge.
    # Its left side goes from its v3 to v4. If centered at (cx, cy):
    # v3 = (cx-1, cy), v4 = (cx-0.5, cy-sqrt(3)/2)
    # For these to match: cx-1 = 1 and cy = 0 => cx = 2? And cx-0.5 = 0.5 => cx = 1. Contradiction.
    
    # I think I'm overcomplicating this. Let me just use a known hex grid.
    # For a hexagonal tiling with flat-top hexagons of side 1:
    # The centers form a triangular lattice with spacing 3/2 in one direction... 
    # Actually, the distance between centers of adjacent hexagons in a hex tiling is sqrt(3).
    
    # Let me just use pointy-top hexagons (rotated 30 deg) for the inner ones,
    # which makes the lattice easier.
    
    # Pointy-top hex with side 1, center at origin:
    # Vertices at angles 30, 90, 150, 210, 270, 330 degrees
    # (sqrt(3)/2, 1/2), (0, 1), (-sqrt(3)/2, 1/2), (-sqrt(3)/2, -1/2), (0, -1), (sqrt(3)/2, -1/2)
    # Width = sqrt(3), Height = 2
    
    # Hex grid for pointy-top:
    # Row spacing (vertical): 1.5
    # Column spacing (horizontal): sqrt(3)
    # Odd rows offset by sqrt(3)/2
    
    sqrt3 = np.sqrt(3)
    
    # Place 11 hexagons in a compact hexagonal arrangement:
    # Center: (0, 0)
    # First ring (6): at distance sqrt(3) in 6 directions
    # Second ring (4): partial
    
    # Directions for pointy-top hex neighbors:
    # The 6 neighbors are at angles 0, 60, 120, 180, 240, 300 from center, at distance sqrt(3)
    # Wait, for pointy-top, the neighbors are at:
    # (sqrt(3), 0), (sqrt(3)/2, 1.5), (-sqrt(3)/2, 1.5), (-sqrt(3), 0), (-sqrt(3)/2, -1.5), (sqrt(3)/2, -1.5)
    
    # Let me verify: two pointy-top hexagons side by side (east-west).
    # East hex center at (dx, 0). Its leftmost x = dx - sqrt(3)/2.
    # West hex center at (0,0). Its rightmost x = sqrt(3)/2.
    # For touching: dx - sqrt(3)/2 = sqrt(3)/2 => dx = sqrt(3). Yes!
    
    # NE neighbor: center at (sqrt(3)/2, 1.5). 
    # Check: bottom of NE hex = 1.5 - 1 = 0.5. Top of center hex = 1.
    # They should share an edge. The top-right edge of center hex goes from (sqrt(3)/2, 0.5) to (0, 1).
    # The bottom-left edge of NE hex (centered at (sqrt(3)/2, 1.5)) goes from (0, 1) to (sqrt(3)/2, 0.5).
    # Yes! They share that edge.
    
    inner = []
    # Center
    inner.append([0.0, 0.0, 30.0])  # 30 deg rotation = pointy-top
    
    # First ring: 6 hexagons
    for i in range(6):
        angle = np.radians(i * 60)
        x = sqrt3 * np.cos(angle)
        y = sqrt3 * np.sin(angle)
        # Wait, the neighbors are at distance sqrt(3) but in specific directions.
        # For pointy-top, the 6 neighbor directions are at angles 0, 60, 120, 180, 240, 300
        # but the actual positions are:
        # East: (sqrt(3), 0)
        # NE: (sqrt(3)/2, 1.5)
        # NW: (-sqrt(3)/2, 1.5)
        # West: (-sqrt(3), 0)
        # SW: (-sqrt(3)/2, -1.5)
        # SE: (sqrt(3)/2, -1.5)
        # These are NOT at distance sqrt(3) from center in all cases.
        # East: dist = sqrt(3) ✓
        # NE: dist = sqrt(3/4 + 9/4) = sqrt(3) ✓
        # So they ARE at distance sqrt(3).
        pass
    
    neighbor_positions = [
        (sqrt3, 0),
        (sqrt3/2, 1.5),
        (-sqrt3/2, 1.5),
        (-sqrt3, 0),
        (-sqrt3/2, -1.5),
        (sqrt3/2, -1.5)
    ]
    for (nx, ny) in neighbor_positions:
        inner.append([nx, ny, 30.0])
    
    # Second ring: add 4 more hexagons
    # The second ring in a hex grid has 12 positions, but we only need 4.
    # Place them at the "corners" to make a compact shape.
    # Second ring positions (distance 2*sqrt(3) or so):
    second_ring = [
        (2*sqrt3, 0),
        (sqrt3, 3.0),
        (-sqrt3, 3.0),
        (0, 3*sqrt3/2 * 2)  # top
    ]
    # Actually let me just pick 4 reasonable positions
    second_ring = [
        (2*sqrt3, 0),
        (sqrt3, 3.0),
        (-sqrt3, 3.0),
        (-2*sqrt3, 0)
    ]
    for (sx, sy) in second_ring:
        inner.append([sx, sy, 30.0])
    
    # Now optimize: minimize the outer hexagon side length
    # Variables: positions of all 11 hexagons (22 variables)
    
    def objective(x):
        if time.time() > DEADLINE - 10:
            return 1e10
        positions = x.reshape(11, 2)
        test_inner = []
        for i in range(11):
            test_inner.append([positions[i, 0], positions[i, 1], 30.0])
        
        all_pts = []
        for (ix, iy, iangle) in test_inner:
            v = hexagon_vertices(ix, iy, iangle, 1.0)
            all_pts.append(v)
        all_pts = np.vstack(all_pts)
        
        hull = convex_hull(all_pts)
        s, cx, cy = outer_hexagon_containing(hull, 0.0)
        
        # Penalty for overlapping hexagons
        penalty = 0.0
        for i in range(11):
            for j in range(i+1, 11):
                # Check if hexagons i and j overlap
                # Approximate: check distance between centers
                dx = positions[i, 0] - positions[j, 0]
                dy = positions[i, 1] - positions[j, 1]
                dist = np.sqrt(dx*dx + dy*dy)
                if dist < sqrt3 - 1e-4:  # Minimum distance for non-overlapping pointy-top hexagons
                    penalty += (sqrt3 - dist) ** 2
        return s + 100 * penalty
    
    x0 = np.array(inner)[:, :2].flatten()
    
    best_x = x0.copy()
    best_s = 1e10
    
    # Try multiple random restarts
    n_restarts = 5
    for r in range(n_restarts):
        if time.time() > DEADLINE - 20:
            break
        
        if r == 0:
            x_start = x0.copy()
        else:
            x_start = x0 + np.random.normal(0, 0.5, size=x0.shape)
        
        result = minimize(objective, x_start, method='Nelder-Mead',
                         options={'maxiter': 2000, 'xatol': 1e-6, 'fatol': 1e-6})
        
        if result.fun < best_s and result.fun < 1e9:
            best_s = result.fun
            best_x = result.x.copy()
    
    # Extract final positions
    final_positions = best_x.reshape(11, 2)
    final_inner = []
    for i in range(11):
        final_inner.append([float(final_positions[i, 0]), float(final_positions[i, 1]), 30.0])
    
    # Compute the actual outer hexagon
    all_pts = []
    for (ix, iy, iangle) in final_inner:
        v = hexagon_vertices(ix, iy, iangle, 1.0)
        all_pts.append(v)
    all_pts = np.vstack(all_pts)
    hull = convex_hull(all_pts)
    s, cx, cy = outer_hexagon_containing(hull, 0.0)
    
    # Verify validity
    if not check_validity(final_inner, (cx, cy), s, 0.0):
        # Fallback to the original seed solution
        inner = [[4.0 * k, 0.0, 0.0] for k in range(-(N // 2), N - N // 2)]
        return inner, (0.0, 0.0), 40.0, 0.0
    
    return final_inner, (float(cx), float(cy)), float(s), 0.0

if __name__ == '__main__':
    inner, center, side, angle = construct()
    print(f"Side length: {side:.6f}")
    print(f"Center: {center}")
