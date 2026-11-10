# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time

def _hex_vertices(cx, cy, d, angle_deg):
    a = np.radians(angle_deg)
    angles = a + np.arange(6) * np.pi / 3
    return np.column_stack([cx + d * np.cos(angles), cy + d * np.sin(angles)])

def _sat_separation(v1, v2, tol=1e-6):
    """SAT: return max gap between convex polygons v1, v2.
    Returns negative if overlapping. Positive = separated by that amount."""
    n1, n2 = len(v1), len(v2)
    best_gap = -np.inf
    # Edges of v1
    for i in range(n1):
        p1 = v1[i]; p2 = v1[(i+1) % n1]
        edge = p2 - p1
        normal = np.array([-edge[1], edge[0]])
        nl = np.linalg.norm(normal)
        if nl < 1e-12: continue
        normal /= nl
        proj1 = v1 @ normal
        proj2 = v2 @ normal
        gap = min(proj2) - max(proj1)
        best_gap = max(best_gap, gap)
    # Edges of v2
    for i in range(n2):
        p1 = v2[i]; p2 = v2[(i+1) % n2]
        edge = p2 - p1
        normal = np.array([-edge[1], edge[0]])
        nl = np.linalg.norm(normal)
        if nl < 1e-12: continue
        normal /= nl
        proj1 = v1 @ normal
        proj2 = v2 @ normal
        gap = min(proj1) - max(proj2)
        best_gap = max(best_gap, gap)
    return best_gap

def _vertex_inside_outer(v, outer_verts):
    """Check if vertex v is inside or on boundary of outer polygon."""
    n = len(outer_verts)
    for i in range(n):
        p1 = outer_verts[i]; p2 = outer_verts[(i+1) % n]
        cross = (p2[0]-p1[0])*(v[1]-p1[1]) - (p2[1]-p1[1])*(v[0]-p1[0])
        if cross < -1e-6:
            return False
    return True

def _compute_min_side(inner_data, outer_center, outer_angle):
    """Given inner hex positions/angles and outer center/angle, find minimum outer side length s
    such that all inner hexagons fit inside outer hexagon and are mutually separated."""
    # Check separation between all pairs
    verts = []
    for i in range(12):
        x, y, ang = inner_data[i]
        verts.append(_hex_vertices(x, y, 1.0, ang))
    
    # Check pairwise separation
    for i in range(12):
        for j in range(i+1, 12):
            gap = _sat_separation(verts[i], verts[j])
            if gap < -1e-6:
                return np.inf
    
    # Find min s: for each inner vertex, compute required s
    # Outer hex vertices for side s: center + s*cos(a+60k), center + s*sin(a+60k)
    # For a vertex v to be inside: for each edge, cross >= 0
    # Edge from V_k to V_{k+1}: direction d = V_{k+1} - V_k
    # cross(d, v - V_k) >= 0
    # V_k = oc + s * u_k where u_k = (cos(a+60k), sin(a+60k))
    # d = s * (u_{k+1} - u_k)
    # cross(s*(u_{k+1}-u_k), v - oc - s*u_k) >= 0
    # s * cross(u_{k+1}-u_k, v-oc) - s^2 * cross(u_{k+1}-u_k, u_k) >= 0
    # Let w = u_{k+1}-u_k, c = v-oc
    # s * cross(w, c) - s^2 * cross(w, u_k) >= 0
    # cross(w, u_k) = cross(u_{k+1}-u_k, u_k) = cross(u_{k+1}, u_k) - 0 = cross(u_{k+1}, u_k)
    # For regular hexagon, cross(u_{k+1}, u_k) = sin(60deg) = sqrt(3)/2 (for CCW)
    # Actually let me just compute numerically
    
    a_rad = np.radians(outer_angle)
    u = np.array([[np.cos(a_rad + k*np.pi/3), np.sin(a_rad + k*np.pi/3)] for k in range(6)])
    
    min_s = 0.0
    for i in range(12):
        for v in verts[i]:
            c = v - outer_center
            for k in range(6):
                w = u[(k+1)%6] - u[k]
                cross_wc = w[0]*c[1] - w[1]*c[0]
                cross_wu = w[0]*u[k][1] - w[1]*u[k][0]
                # s * cross_wc - s^2 * cross_wu >= 0
                # If cross_wu > 0: s^2 * cross_wu <= s * cross_wc => s <= cross_wc / cross_wu (if cross_wc > 0)
                # If cross_wu < 0: s^2 * |cross_wu| >= s * cross_wc => need s >= cross_wc / cross_wu (careful with signs)
                # Let's solve: -cross_wu * s^2 + cross_wc * s >= 0
                # s * (cross_wc - cross_wu * s) >= 0
                # For s > 0: cross_wc - cross_wu * s >= 0 => s <= cross_wc / cross_wu if cross_wu > 0
                # If cross_wu > 0: s <= cross_wc / cross_wu (requires cross_wc > 0)
                # If cross_wu < 0: s >= cross_wc / cross_wu (requires cross_wc < 0 for positive s bound)
                # If cross_wu == 0: need cross_wc >= 0
                if abs(cross_wu) < 1e-12:
                    if cross_wc < -1e-10:
                        return np.inf
                elif cross_wu > 0:
                    # s <= cross_wc / cross_wu
                    if cross_wc < 0:
                        return np.inf
                    s_req = cross_wc / cross_wu
                    min_s = max(min_s, s_req)
                else:
                    # cross_wu < 0: s >= cross_wc / cross_wu
                    # cross_wc / cross_wu is negative if cross_wc > 0, positive if cross_wc < 0
                    # s >= cross_wc / cross_wu
                    s_req = cross_wc / cross_wu
                    min_s = max(min_s, s_req)
    
    return min_s

def _objective(params, outer_angle):
    """Objective: minimize outer side length. params = [x1,y1,ang1, ..., x12,y12,ang12, ox, oy]"""
    inner = params[:36].reshape(12, 3)
    ox, oy = params[36], params[37]
    outer_center = np.array([ox, oy])
    
    s = _compute_min_side(inner, outer_center, outer_angle)
    if np.isinf(s):
        return 1e6
    return s

def hexagon_packing_12():
    start_time = time.time()
    deadline = start_time + 580  # safety margin
    
    best_s = np.inf
    best_inner = None
    best_outer_center = None
    best_outer_angle = 0.0
    
    # Initial configurations
    def make_grid_config():
        # Hexagonal grid: 1 center + 6 ring + 5 more
        configs = []
        # Configuration 1: 3-4-3-3 arrangement (common for 12 hexes)
        # Row-based: rows of 3,4,3,2 or similar
        # Let's use a known good starting point
        # 12 hexagons in a hexagonal arrangement
        # Positions for unit hexagons (flat-top, angle=0 means vertex at right)
        # For flat-top hex with side 1: width=2, height=sqrt(3)
        # Spacing: horizontal=1.5, vertical=sqrt(3)/2 for adjacent rows
        
        # Try a 4-3-4-1 or similar
        # Actually, let me use the honeycomb lattice
        # Hex lattice: axial coords (q, r) -> pixel: x = 1.5*q, y = sqrt(3)*(r + q/2)
        # 12 hexes: center + 6 neighbors + 5 more from second ring
        # Second ring has 12 hexes, pick 5
        
        sqrt3 = np.sqrt(3)
        # Center
        positions = [(0, 0)]
        # First ring (6 hexes)
        for k in range(6):
            angle = k * np.pi / 3
            x = sqrt3 * np.cos(angle)
            y = sqrt3 * np.sin(angle)
            positions.append((x, y))
        # Second ring: 12 positions at distance 2*sqrt(3) in hex directions
        # Actually second ring in hex lattice: positions at axial distance 2
        # Axial (2,0), (1,1), (0,2), (-1,2), (-2,1), (-2,0), (-1,-1), (0,-2), (1,-2), (2,-1), (2,0), (1,1)
        # Pick 5 from second ring
        # Let's just add 5 more at reasonable positions
        # For a tighter pack, use positions that fill gaps
        # Second ring axial coords: (2,0),(1,1),(0,2),(-1,2),(-2,1),(-2,0),(-1,-1),(0,-2),(1,-2),(2,-1)
        # Pick: (2,0), (0,2), (-2,1), (-2,0), (0,-2) - 5 hexes
        second_ring = [(2,0), (0,2), (-2,1), (-2,0), (0,-2)]
        for (q, r) in second_ring:
            x = 1.5 * q
            y = sqrt3 * (r + q/2.0)
            positions.append((x, y))
        
        inner = np.array([[x, y, 0.0] for x, y in positions])
        return inner
    
    def make_symmetric_config():
        """Create a symmetric 12-hexagon configuration"""
        sqrt3 = np.sqrt(3)
        # Use a more compact arrangement
        # 12 hexes: try 2 rows of 6, or 3+4+3+2, etc.
        # For flat-top hexagons:
        # Width of hex = 2, height = sqrt(3)
        # Horizontal spacing for same row: 2 (edge to edge) or 3 (vertex to vertex with gap)
        # Actually for touching: horizontal = 1.5 * 2 = 3? No.
        # Flat-top hex: vertices at (1,0), (0.5, sqrt(3)/2), (-0.5, sqrt(3)/2), (-1,0), (-0.5,-sqrt(3)/2), (0.5,-sqrt(3)/2)
        # Two adjacent flat-top hexes touching on right/left: centers separated by (1.5, 0)? 
        # Right vertex of hex1 at (1,0) relative to center. Left vertex of hex2 at (-1,0) relative.
        # For them to touch: center2 - center1 = (1.5, 0)? No, that gives gap.
        # Actually the rightmost point is at x=1, leftmost at x=-1. For edge contact:
        # The right edge of hex1 connects (1,0) and (0.5, sqrt(3)/2). 
        # Hmm, let me just use the honeycomb lattice properly.
        
        # Honeycomb: each hex touches 6 neighbors
        # For flat-top: neighbor offsets are (1.5, sqrt(3)/2), (1.5, -sqrt(3)/2), (0, sqrt(3)), 
        # and their negatives, plus (3, 0) is NOT a neighbor
        # Actually: the 6 neighbors of a flat-top hex at origin are at:
        # (1.5, sqrt(3)/2), (1.5, -sqrt(3)/2), (0, sqrt(3)), (-1.5, sqrt(3)/2), (-1.5, -sqrt(3)/2), (0, -sqrt(3))
        
        neighbors = [
            (1.5, sqrt3/2), (1.5, -sqrt3/2), (0, sqrt3),
            (-1.5, sqrt3/2), (-1.5, -sqrt3/2), (0, -sqrt3)
        ]
        positions = [(0, 0)] + neighbors
        
        # Add 5 more from second ring
        # Second ring neighbors (axial distance 2):
        # (3, 0), (3, sqrt3), (1.5, 2*sqrt3), (-1.5, 2*sqrt3), (-3, sqrt3), (-3, 0),
        # (-1.5, -sqrt3), (1.5, -sqrt3), etc.
        # Let me compute: from center, go 2 steps in each of 6 directions
        # Direction 0: (1.5, sqrt3/2) * 2 = (3, sqrt3)
        # Direction 1: (1.5, -sqrt3/2) * 2 = (3, -sqrt3)
        # Direction 2: (0, sqrt3) * 2 = (0, 2*sqrt3)
        # Direction 3: (-1.5, sqrt3/2) * 2 = (-3, sqrt3)
        # Direction 4: (-1.5, -sqrt3/2) * 2 = (-3, -sqrt3)
        # Direction 5: (0, -sqrt3) * 2 = (0, -2*sqrt3)
        # Also diagonal second ring:
        # (3, 0): from (1.5, sqrt3/2) then (1.5, -sqrt3/2) -> (3, 0)
        # (0, sqrt3*sqrt3)? No.
        # Let me just pick 5 reasonable second-ring positions
        second_ring = [
            (3, 0), (3, sqrt3), (0, 2*sqrt3), (-3, sqrt3), (0, -2*sqrt3)
        ]
        positions += second_ring
        
        inner = np.array([[x, y, 0.0] for x, y in positions])
        return inner
    
    def make_compact_config():
        """Try a more compact 12-hex arrangement"""
        sqrt3 = np.sqrt(3)
        # 4 rows: 3, 4, 3, 2 hexes
        # Or 2, 3, 4, 3, 2? That's 14.
        # 3, 4, 3, 2 = 12
        # Row 0: 3 hexes
        # Row 1: 4 hexes
        # Row 2: 3 hexes
        # Row 3: 2 hexes
        positions = []
        # Row 0 (3 hexes): x = -1.5, 0, 1.5
        for i in range(3):
            positions.append((i * 1.5 - 1.5, 0))
        # Row 1 (4 hexes): offset by 0.75 in x, sqrt3/2 in y
        for i in range(4):
            positions.append((i * 1.5 - 1.5 + 0.75, sqrt3/2))
        # Row 2 (3 hexes):
        for i in range(3):
            positions.append((i * 1.5 - 1.5, sqrt3))
        # Row 3 (2 hexes):
        for i in range(2):
            positions.append((i * 1.5 - 0.75, 1.5*sqrt3/2))  # Hmm this doesn't look right
        
        # Let me redo with proper hex grid
        # Flat-top hex grid:
        # Row spacing: sqrt(3)
        # Same row: x spacing = 3 (for touching edges)
        # Adjacent row: x offset = 1.5, y offset = sqrt(3)/2
        # Wait, for flat-top hexes:
        # Width = 2 (vertex to vertex horizontally)
        # Height = sqrt(3) (flat to flat vertically)
        # Two hexes side by side (sharing vertical edge): centers separated by (1.5, 0)? 
        # No: the rightmost vertex is at (1, 0), the leftmost at (-1, 0).
        # For edge sharing, the right edge of hex A and left edge of hex B must coincide.
        # Right edge of A: from (1, 0) to (0.5, sqrt(3)/2)
        # Left edge of B: from (-1, 0) to (-0.5, sqrt(3)/2)
        # For these to be the same edge: B_center - A_center = (1.5, 0)
        # Check: A's right vertex (1,0) + (1.5, 0) = (2.5, 0). B's left vertex = (-1, 0) + (1.5, 0) = (0.5, 0). Not matching.
        # I think I'm confusing myself. Let me use the standard hex grid formulas.
        
        # Standard pointy-top hex grid:
        # Size s (distance from center to vertex)
        # Width = sqrt(3) * s, Height = 2 * s
        # Horizontal spacing = sqrt(3) * s
        # Vertical spacing = 1.5 * s
        # Odd rows offset by sqrt(3)/2 * s
        
        # For our case, "side" d = 1 means the edge length is 1.
        # For a regular hexagon with edge length d:
        # Circumradius (center to vertex) = d
        # Inradius (center to edge midpoint) = d * sqrt(3)/2
        # Width (pointy-top) = sqrt(3) * d
        # Height (pointy-top) = 2 * d
        
        # So for d=1, pointy-top:
        # Width = sqrt(3), Height = 2
        # Horizontal spacing = sqrt(3)
        # Vertical spacing = 1.5
        # Odd row offset = sqrt(3)/2
        
        # Let me use pointy-top orientation (angle = 30 degrees for our parameterization)
        # Or just use the coordinates directly.
        
        # For simplicity, let me place hexes on a hex lattice with the correct spacing.
        # Using axial coordinates (q, r):
        # x = sqrt(3) * (q + r/2)
        # y = 1.5 * r
        # (for pointy-top, size=1)
        
        # 12 hexes: center + 6 first ring + 5 second ring
        positions = []
        # Center (0,0)
        positions.append((0, 0))
        # First ring: (1,0), (1,-1), (0,-1), (-1,0), (-1,1), (0,1)
        first_ring = [(1,0), (1,-1), (0,-1), (-1,0), (-1,1), (0,1)]
        for (q, r) in first_ring:
            x = sqrt3 * (q + r/2.0)
            y = 1.5 * r
            positions.append((x, y))
        # Second ring: pick 5
        # (2,0), (2,-1), (1,-2), (0,-2), (-1,-1), (-2,0), (-2,1), (-1,2), (0,2), (1,1), (2,0), (2,-1)
        # Pick: (2,0), (0,-2), (-2,1), (-1,2), (1,1)
        second_ring = [(2,0), (0,-2), (-2,1), (-1,2), (1,1)]
        for (q, r) in second_ring:
            x = sqrt3 * (q + r/2.0)
            y = 1.5 * r
            positions.append((x, y))
        
        inner = np.array([[x, y, 30.0] for x, y in positions])  # 30 deg for pointy-top
        return inner
    
    # Generate multiple initial configurations
    initial_configs = [
        make_grid_config(),
        make_symmetric_config(),
        make_compact_config()
    ]
    
    # Also try with different outer angles
    outer_angles = [0, 30, 60, 90, 120, 150]
    
    for outer_angle in outer_angles:
        if time.time() > deadline - 60:
            break
        
        for init_inner in initial_configs:
            if time.time() > deadline - 60:
                break
            
            # Initial outer center: centroid of inner hexes
            ox, oy = np.mean(init_inner[:, 0]), np.mean(init_inner[:, 1])
            
            params0 = np.concatenate([init_inner.flatten(), [ox, oy]])
            
            # Run Nelder-Mead
            try:
                result = minimize(
                    _objective,
                    params0,
                    args=(outer_angle,),
                    method='Nelder-Mead',
                    options={
                        'maxiter': 2000,
                        'xatol': 1e-6,
                        'fatol': 1e-6,
                        'adaptive': True
                    }
                )
                
                if result.fun < best_s and np.isfinite(result.fun):
                    best_s = result.fun
                    best_inner = result.x[:36].reshape(12, 3)
                    best_outer_center = np.array([result.x[36], result.x[37]])
                    best_outer_angle = outer_angle
            except Exception:
                pass
            
            # Also try a second round with tighter tolerances if we have time
            if time.time() < deadline - 120 and result.fun < 5:
                try:
                    result2 = minimize(
                        _objective,
                        result.x,
                        args=(outer_angle,),
                        method='Nelder-Mead',
                        options={
                            'maxiter': 3000,
                            'xatol': 1e-8,
                            'fatol': 1e-8,
                            'adaptive': True
                        }
                    )
                    if result2.fun < best_s and np.isfinite(result2.fun):
                        best_s = result2.fun
                        best_inner = result2.x[:36].reshape(12, 3)
                        best_outer_center = np.array([result2.x[36], result2.x[37]])
                        best_outer_angle = outer_angle
                except Exception:
                    pass
    
    # If optimization failed, use a fallback
    if not np.isfinite(best_s) or best_inner is None:
        # Use the simple grid from the original
        inner_hex_data = np.array(
            [
                [0, 0, 0],
                [-2.5, 0, 0],
                [2.5, 0, 0],
                [-1.25, 2.17, 0],
                [1.25, 2.17, 0],
                [-1.25, -2.17, 0],
                [1.25, -2.17, 0],
                [-3.75, 2.17, 0],
                [3.75, 2.17, 0],
                [-3.75, -2.17, 0],
                [3.75, -2.17, 0],
                [0, -4, 0],
            ]
        )
        outer_hex_data = np.array([0, 0, 0])
        s = 8.0
        return inner_hex_data, outer_hex_data, s
    
    # Final result
    inner_hex_data = best_inner
    outer_hex_data = np.array([best_outer_center[0], best_outer_center[1], best_outer_angle])
    s = best_s
    
    # Ensure s is at least sqrt(12)
    if s < np.sqrt(12):
        s = np.sqrt(12)
    
    return inner_hex_data, outer_hex_data, float(s)

# EVOLVE-BLOCK-END
