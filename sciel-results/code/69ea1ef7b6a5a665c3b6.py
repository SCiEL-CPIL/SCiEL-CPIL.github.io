"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
from scipy.optimize import minimize
import time

N = 12

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin
    
    # Unit regular hexagon vertices (side length 1), centered at origin
    # Flat-top orientation: vertices at angles 0, 60, 120, 180, 240, 300
    hex_verts = np.array([
        [1.0, 0.0],
        [0.5, np.sqrt(3)/2],
        [-0.5, np.sqrt(3)/2],
        [-1.0, 0.0],
        [-0.5, -np.sqrt(3)/2],
        [0.5, -np.sqrt(3)/2]
    ])
    
    # Outer hexagon vertices for given side length s, center (cx, cy), rotation angle
    def outer_hex_verts(cx, cy, s, angle_deg):
        angle_rad = np.radians(angle_deg)
        # Regular hexagon with side length s, flat-top
        verts = np.array([
            [s, 0.0],
            [s/2, s*np.sqrt(3)/2],
            [-s/2, s*np.sqrt(3)/2],
            [-s, 0.0],
            [-s/2, -s*np.sqrt(3)/2],
            [s/2, -s*np.sqrt(3)/2]
        ])
        # Rotate
        cos_a = np.cos(angle_rad)
        sin_a = np.sin(angle_rad)
        R = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
        verts = verts @ R.T
        # Translate
        verts[:, 0] += cx
        verts[:, 1] += cy
        return verts
    
    def inner_hex_verts(cx, cy, angle_deg):
        angle_rad = np.radians(angle_deg)
        cos_a = np.cos(angle_rad)
        sin_a = np.sin(angle_rad)
        R = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
        verts = hex_verts @ R.T
        verts[:, 0] += cx
        verts[:, 1] += cy
        return verts
    
    def point_in_hexagon(px, py, hx, hy, s, angle_deg):
        """Check if point (px,py) is inside hexagon centered at (hx,hy) with side s and rotation angle_deg."""
        # Transform point to hexagon's local frame
        dx = px - hx
        dy = py - hy
        angle_rad = np.radians(angle_deg)
        cos_a = np.cos(-angle_rad)
        sin_a = np.sin(-angle_rad)
        R = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
        local_x = dx * cos_a - dy * sin_a
        local_y = dx * sin_a + dy * cos_a
        
        # For a regular hexagon with side length s (flat-top), the distance from center to each edge
        # The hexagon can be defined by 6 half-planes
        # For flat-top hexagon with "radius" (center to vertex) = s:
        # Actually for side length s, the distance from center to vertex is s, and to edge is s*sqrt(3)/2
        
        # Let's use the support function approach. A point is inside if it satisfies all 6 inequalities.
        # For a regular hexagon centered at origin with flat-top and side length s:
        # The vertices are at (s,0), (s/2, s*sqrt(3)/2), etc.
        # The edges are:
        # Edge 1: from (s,0) to (s/2, s*sqrt(3)/2). Normal pointing inward.
        # The edge line: x*cos(30) + y*sin(30) = s*cos(30) = s*sqrt(3)/2
        # Actually, let's just compute the 6 edge normals.
        
        # For a regular hexagon, the 6 edge outward normals point at angles 30, 90, 150, 210, 270, 330 degrees
        # The distance from center to each edge (apothem) = s * sqrt(3) / 2
        
        apothem = s * np.sqrt(3) / 2
        normal_angles = np.radians([30, 90, 150, 210, 270, 330])
        
        for na in normal_angles:
            nx = np.cos(na)
            ny = np.sin(na)
            # The edge is at distance apothem from center in direction (nx, ny)
            # Point is inside if local_x * nx + local_y * ny <= apothem
            if local_x * nx + local_y * ny > apothem + 1e-10:
                return False
        return True
    
    def check_feasibility(inner_positions, outer_cx, outer_cy, outer_s, outer_angle):
        """Check if all inner hexagons fit inside the outer hexagon and don't overlap each other."""
        # Check containment: all vertices of each inner hexagon must be inside outer hexagon
        for i in range(N):
            ix, iy, iangle = inner_positions[i]
            verts = inner_hex_verts(ix, iy, iangle)
            for vx, vy in verts:
                if not point_in_hexagon(vx, vy, outer_cx, outer_cy, outer_s, outer_angle):
                    return False
        
        # Check non-overlap between pairs of inner hexagons
        # Two convex polygons don't overlap if there exists a separating axis
        # For regular hexagons, we can check all 12 edge normals (6 from each)
        for i in range(N):
            for j in range(i+1, N):
                ix1, iy1, ia1 = inner_positions[i]
                ix2, iy2, ia2 = inner_positions[j]
                verts1 = inner_hex_verts(ix1, iy1, ia1)
                verts2 = inner_hex_verts(ix2, iy2, ia2)
                
                if polygons_overlap(verts1, verts2):
                    return False
        return True
    
    def polygons_overlap(verts1, verts2):
        """Check if two convex polygons overlap using separating axis theorem."""
        all_edges = []
        for verts in [verts1, verts2]:
            for k in range(6):
                p1 = verts[k]
                p2 = verts[(k+1) % 6]
                edge = p2 - p1
                # Normal to edge
                normal = np.array([-edge[1], edge[0]])
                norm = np.linalg.norm(normal)
                if norm > 1e-10:
                    normal /= norm
                    all_edges.append(normal)
        
        for normal in all_edges:
            # Project both polygons onto the normal axis
            proj1 = verts1 @ normal
            proj2 = verts2 @ normal
            min1, max1 = np.min(proj1), np.max(proj1)
            min2, max2 = np.min(proj2), np.max(proj2)
            
            # Check for gap
            if max1 < min2 - 1e-10 or max2 < min1 - 1e-10:
                return False  # Separating axis found, no overlap
        
        return True  # No separating axis found, polygons overlap
    
    def objective(params):
        """Minimize outer side length. Params: outer_cx, outer_cy, outer_s, outer_angle, then 3*N inner params."""
        if time.time() > deadline:
            return 1e10
        
        outer_cx = params[0]
        outer_cy = params[1]
        outer_s = params[2]
        outer_angle = params[3]
        
        inner_positions = []
        for i in range(N):
            ix = params[4 + 3*i]
            iy = params[4 + 3*i + 1]
            iangle = params[4 + 3*i + 2]
            inner_positions.append([ix, iy, iangle])
        
        # Penalty for infeasibility
        penalty = 0.0
        
        # Check containment
        for i in range(N):
            ix, iy, iangle = inner_positions[i]
            verts = inner_hex_verts(ix, iy, iangle)
            for vx, vy in verts:
                if not point_in_hexagon(vx, vy, outer_cx, outer_cy, outer_s, outer_angle):
                    # Penalty proportional to how far outside
                    # Compute distance to nearest edge
                    dx = vx - outer_cx
                    dy = vy - outer_cy
                    angle_rad = np.radians(outer_angle)
                    cos_a = np.cos(-angle_rad)
                    sin_a = np.sin(-angle_rad)
                    local_x = dx * cos_a - dy * sin_a
                    local_y = dx * sin_a + dy * cos_a
                    
                    apothem = outer_s * np.sqrt(3) / 2
                    normal_angles = np.radians([30, 90, 150, 210, 270, 330])
                    
                    for na in normal_angles:
                        nx = np.cos(na)
                        ny = np.sin(na)
                        dist = local_x * nx + local_y * ny - apothem
                        if dist > 0:
                            penalty += dist * dist
        
        # Check non-overlap
        for i in range(N):
            for j in range(i+1, N):
                ix1, iy1, ia1 = inner_positions[i]
                ix2, iy2, ia2 = inner_positions[j]
                verts1 = inner_hex_verts(ix1, iy1, ia1)
                verts2 = inner_hex_verts(ix2, iy2, ia2)
                
                # Compute minimum separation
                all_edges = []
                for verts in [verts1, verts2]:
                    for k in range(6):
                        p1 = verts[k]
                        p2 = verts[(k+1) % 6]
                        edge = p2 - p1
                        normal = np.array([-edge[1], edge[0]])
                        norm = np.linalg.norm(normal)
                        if norm > 1e-10:
                            normal /= norm
                            all_edges.append(normal)
                
                for normal in all_edges:
                    proj1 = verts1 @ normal
                    proj2 = verts2 @ normal
                    min1, max1 = np.min(proj1), np.max(proj1)
                    min2, max2 = np.min(proj2), np.max(proj2)
                    
                    overlap = min(max1, max2) - max(min1, min2)
                    if overlap > 0:
                        penalty += overlap * overlap
        
        return outer_s + penalty
    
    # Initial guess: place hexagons in a hexagonal grid pattern
    # For N=12, we can use a pattern of 12 hexagons
    # A natural arrangement: 4 rows of 3, or 3 rows of 4, or other patterns
    
    # Let's try a hexagonal close-packed arrangement
    # Center the cluster at origin
    # Hexagonal lattice: positions at (i*a, j*b) with appropriate offsets
    
    # For 12 hexagons, a good initial arrangement:
    # Row 0: 3 hexagons
    # Row 1: 4 hexagons  
    # Row 2: 3 hexagons
    # Row 3: 2 hexagons
    # Total: 12
    
    # Actually, let's use a more symmetric pattern
    # 12 hexagons in a hexagonal shape: center + 6 around + 5 more?
    # Or just use a rectangular grid with hexagonal offsets
    
    # Let's use a simple approach: place them in a hexagonal pattern
    # The distance between centers of adjacent non-overlapping hexagons is at least 2*apothem = sqrt(3)
    
    # For a flat-top hexagon with side 1, the distance from center to vertex is 1
    # Two such hexagons touching have center distance = 2 (vertex to vertex) or sqrt(3) (edge to edge)
    
    # Let's initialize with a hexagonal grid
    # Spacing = 2.0 (vertex to vertex for flat-top)
    
    spacing = 2.0
    positions = []
    
    # Hexagonal grid positions for 12 hexagons
    # Use axial coordinates
    # For a hexagonal grid with flat-top hexagons:
    # x = spacing * (q + r/2)
    # y = spacing * sqrt(3)/2 * r
    
    # Let's generate 12 positions in a compact hexagonal pattern
    # Center at (0,0)
    # Ring 0: 1 hex (center)
    # Ring 1: 6 hexes
    # Ring 2: we need 5 more from the 12 in ring 2
    
    # Actually, let's just use a simple grid
    # 4 columns x 3 rows with hexagonal offset
    
    col_x = [0, 1, 2, 3]
    row_y = [0, 1, 2]
    
    idx = 0
    for r in range(3):
        for c in range(4):
            if idx >= 12:
                break
            x = c * spacing + (r % 2) * spacing / 2
            y = r * spacing * np.sqrt(3) / 2
            # Center the pattern
            x -= 1.5 * spacing
            y -= spacing * np.sqrt(3)
            positions.append([x, y, 0.0])
            idx += 1
    
    # Center the pattern
    cx = np.mean([p[0] for p in positions])
    cy = np.mean([p[1] for p in positions])
    for p in positions:
        p[0] -= cx
        p[1] -= cy
    
    # Initial outer hexagon
    outer_cx = 0.0
    outer_cy = 0.0
    outer_s = 6.0  # Initial guess
    outer_angle = 0.0
    
    # Build initial params
    params = [outer_cx, outer_cy, outer_s, outer_angle]
    for p in positions:
        params.extend(p)
    params = np.array(params)
    
    # Optimize
    best_params = params.copy()
    best_obj = objective(params)
    
    # Try multiple random restarts
    n_restarts = 5
    for restart in range(n_restarts):
        if time.time() > deadline - 10:
            break
        
        if restart == 0:
            current_params = params.copy()
        else:
            # Perturb initial positions
            current_params = params.copy()
            for i in range(N):
                current_params[4 + 3*i] += np.random.uniform(-0.5, 0.5)
                current_params[4 + 3*i + 1] += np.random.uniform(-0.5, 0.5)
                current_params[4 + 3*i + 2] += np.random.uniform(-30, 30)
        
        result = minimize(objective, current_params, method='Nelder-Mead',
                          options={'maxiter': 2000, 'xatol': 1e-6, 'fatol': 1e-6, 'adaptive': True})
        
        if result.fun < best_obj:
            best_obj = result.fun
            best_params = result.x
        
        if time.time() > deadline - 20:
            break
    
    # Extract best solution
    outer_cx = best_params[0]
    outer_cy = best_params[1]
    outer_s = best_params[2]
    outer_angle = best_params[3]
    
    inner = []
    for i in range(N):
        ix = best_params[4 + 3*i]
        iy = best_params[4 + 3*i + 1]
        iangle = best_params[4 + 3*i + 2]
        inner.append([ix, iy, iangle])
    
    # Verify feasibility
    if not check_feasibility(inner, outer_cx, outer_cy, outer_s, outer_angle):
        # If infeasible, use a conservative fallback
        # Scale up the outer hexagon until feasible
        for scale in np.arange(1.0, 10.0, 0.1):
            test_s = outer_s * scale
            if check_feasibility(inner, outer_cx, outer_cy, test_s, outer_angle):
                outer_s = test_s
                break
        else:
            # Last resort: very large outer hexagon
            outer_s = 20.0
    
    return inner, (outer_cx, outer_cy), outer_s, outer_angle
# EVOLVE-BLOCK-END
