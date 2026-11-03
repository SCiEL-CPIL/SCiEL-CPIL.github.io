"""Packing 12 unit hexagons into a minimal outer hexagon."""
import numpy as np
import time
from scipy.optimize import minimize

N = 12
DEADLINE = time.time() + 110.0  # safety margin before 120s kill

def hex_vertices(cx, cy, angle_deg, side=1.0):
    """Return 6 vertices of a regular hexagon centered at (cx,cy) with rotation angle_deg."""
    a = np.radians(angle_deg)
    angles = a + np.arange(6) * np.pi / 3.0
    return np.column_stack([cx + side * np.cos(angles), cy + side * np.sin(angles)])

def hex_vertices_outer(cx, cy, angle_deg, side):
    """Outer hexagon vertices."""
    a = np.radians(angle_deg)
    angles = a + np.arange(6) * np.pi / 3.0
    return np.column_stack([cx + side * np.cos(angles), cy + side * np.sin(angles)])

def poly_collision(v1, v2):
    """Check if two convex polygons (arrays of 6 vertices) overlap using SAT-lite.
    For each edge of v1, check if any vertex of v2 is on the wrong side.
    Returns True if they overlap (penetration)."""
    # Check edges of v1 against vertices of v2
    for i in range(6):
        e1 = v1[(i+1) % 6] - v1[i]
        # Normal pointing inward (to the left of edge direction for CCW polygon)
        nx = -e1[1]
        ny = e1[0]
        # For a CCW polygon, interior is to the left of each edge
        # A vertex of v2 is inside v1's half-plane if dot(normal, vertex - edge_start) >= 0
        # But we need to check if ALL vertices of v2 are on the interior side
        # If ANY vertex of v2 is on the exterior side, no penetration from this edge
        # Actually for collision: if for every edge of v1, all vertices of v2 are on interior side -> v2 is inside v1
        # For overlap: if there exists an edge of v1 such that some vertex of v2 is outside, AND vice versa
        # Simpler: use the fact that two convex polygons overlap iff they share a separating axis
        # Let's just check: for each edge of v1, is there a vertex of v2 strictly outside?
        # And for each edge of v2, is there a vertex of v1 strictly outside?
        # If for all edges of v1, all vertices of v2 are inside -> v2 contained in v1 (overlap)
        # If for all edges of v2, all vertices of v1 are inside -> v1 contained in v2 (overlap)
        # Otherwise no overlap
        pass
    
    # Better approach: check if any vertex of v2 is inside v1 OR any vertex of v1 is inside v2
    # OR if any edges intersect
    # For hexagons, a simpler robust check:
    # Two convex polygons are disjoint iff there exists a separating line.
    # We'll check all edge normals of both polygons.
    
    eps = 1e-9
    # Check all edges of both polygons as separating axes
    for poly_a, poly_b in [(v1, v2), (v2, v1)]:
        for i in range(6):
            e = poly_a[(i+1) % 6] - poly_a[i]
            nx = -e[1]
            ny = e[0]
            # Project all vertices of poly_b onto this normal
            dots_b = (poly_b[:, 0] - poly_a[i, 0]) * nx + (poly_b[:, 1] - poly_a[i, 1]) * ny
            # Project all vertices of poly_a onto this normal
            dots_a = (poly_a[:, 0] - poly_a[i, 0]) * nx + (poly_a[:, 1] - poly_a[i, 1]) * ny
            # If max of poly_b < min of poly_a (with eps), they're separated
            if np.max(dots_b) < np.min(dots_a) + eps:
                return False  # Separated, no collision
    return True  # No separating axis found -> they overlap

def point_in_hex(px, py, cx, cy, angle_deg, side):
    """Check if point (px,py) is inside hexagon centered at (cx,cy) with given angle and side."""
    # Transform to hexagon's local frame
    a = np.radians(angle_deg)
    # Rotate point into hexagon frame
    dx = px - cx
    dy = py - cy
    # Rotate by -angle
    cos_a = np.cos(-a)
    sin_a = np.sin(-a)
    lx = dx * cos_a - dy * sin_a
    ly = dx * sin_a + dy * cos_a
    # In local frame, hexagon has vertices at angles 0, 60, 120, 180, 240, 300
    # The hexagon is the intersection of 6 half-planes
    # For each edge, check if point is on the interior side
    for k in range(6):
        ang = k * np.pi / 3.0
        # Edge midpoint direction
        ex = np.cos(ang)
        ey = np.sin(ang)
        # The inradius is side * sqrt(3)/2
        inradius = side * np.sqrt(3.0) / 2.0
        # The edge is at distance inradius from center, perpendicular to (ex, ey)
        # A point is inside if dot(point, edge_normal) <= inradius
        # Edge normal points outward along (ex, ey)
        if lx * ex + ly * ey > inradius + 1e-9:
            return False
    return True

def compute_outer_side(inner, outer_cx, outer_cy, outer_angle):
    """Compute the minimum outer side length needed to contain all inner hexagons.
    inner: list of [x, y, angle_deg]
    Returns the required side length."""
    # For each inner hexagon, find the maximum extent in each of the 6 outer hexagon edge directions
    # The outer hexagon of side S centered at (cx,cy) with rotation theta has 6 edges
    # Edge k (k=0..5) has outward normal at angle theta + k*pi/3
    # The constraint: for each inner hex vertex, its projection onto each outward normal
    # must be <= S * sqrt(3)/2 (the inradius of the outer hex)
    
    outer_inradius_needed = 0.0
    theta = np.radians(outer_angle)
    
    for ix, iy, iang in inner:
        verts = hex_vertices(ix, iy, iang, 1.0)
        for k in range(6):
            normal_angle = theta + k * np.pi / 3.0
            nx = np.cos(normal_angle)
            ny = np.sin(normal_angle)
            # Project all vertices onto this normal
            projections = (verts[:, 0] - outer_cx) * nx + (verts[:, 1] - outer_cy) * ny
            max_proj = np.max(projections)
            if max_proj > outer_inradius_needed:
                outer_inradius_needed = max_proj
    
    # inradius = side * sqrt(3)/2, so side = inradius / (sqrt(3)/2) = 2*inradius/sqrt(3)
    return 2.0 * outer_inradius_needed / np.sqrt(3.0)

def check_validity(inner, outer_cx, outer_cy, outer_angle, outer_side):
    """Check if configuration is valid: no overlaps and all contained."""
    # Check containment
    inradius = outer_side * np.sqrt(3.0) / 2.0
    theta = np.radians(outer_angle)
    for ix, iy, iang in inner:
        verts = hex_vertices(ix, iy, iang, 1.0)
        for k in range(6):
            normal_angle = theta + k * np.pi / 3.0
            nx = np.cos(normal_angle)
            ny = np.sin(normal_angle)
            projections = (verts[:, 0] - outer_cx) * nx + (verts[:, 1] - outer_cy) * ny
            if np.max(projections) > inradius + 1e-6:
                return False
    
    # Check no overlaps
    for i in range(len(inner)):
        v1 = hex_vertices(inner[i][0], inner[i][1], inner[i][2], 1.0)
        for j in range(i+1, len(inner)):
            v2 = hex_vertices(inner[j][0], inner[j][1], inner[j][2], 1.0)
            if poly_collision(v1, v2):
                return False
    return True

def objective(params):
    """Objective: outer side length. Returns large value if invalid."""
    # params: 3*N + 3 = 39 floats
    # First 3*N: inner hexagons (x, y, angle_deg)
    # Last 3: outer (cx, cy, angle_deg)
    inner = []
    for i in range(N):
        x = params[3*i]
        y = params[3*i + 1]
        a = params[3*i + 2]
        inner.append([x, y, a])
    outer_cx = params[3*N]
    outer_cy = params[3*N + 1]
    outer_angle = params[3*N + 2]
    
    # Compute required outer side
    side = compute_outer_side(inner, outer_cx, outer_cy, outer_angle)
    
    # Check validity (overlaps)
    for i in range(N):
        v1 = hex_vertices(inner[i][0], inner[i][1], inner[i][2], 1.0)
        for j in range(i+1, N):
            v2 = hex_vertices(inner[j][0], inner[j][1], inner[j][2], 1.0)
            if poly_collision(v1, v2):
                return 1000.0  # Penalty
    
    return side

def initial_config():
    """Create initial hexagonal lattice arrangement."""
    inner = []
    # Arrange in rows: 2, 3, 4, 3 or similar
    # Hexagonal lattice spacing: horizontal = 2*sqrt(3)/2 * 2 = 2*sqrt(3) for non-overlapping
    # Actually for unit hexagons in flat-top orientation:
    # Horizontal spacing = 2*sqrt(3)/2 * 2 = 2*sqrt(3) ≈ 3.464
    # Vertical spacing = 3 (for pointy-top) or 2*sqrt(3)/2 * ... 
    # Let's use a simple grid with some offset
    
    # Use pointy-top hexagons (angle=0 means vertex at top? Let's define: angle=0 -> vertex at 0 degrees = right)
    # For flat-top hexagons, the width is 2*side=2, height is sqrt(3)*side=sqrt(3)
    # For pointy-top, width is sqrt(3), height is 2
    
    # Let's arrange in a compact hexagonal pattern
    # Row 0: 2 hexagons
    # Row 1: 3 hexagons  
    # Row 2: 3 hexagons
    # Row 3: 2 hexagons
    # Total: 10... need 12
    
    # Better: 3 rows of 4? Or a hexagonal cluster
    # Let's do: row0=2, row1=4, row2=4, row3=2 = 12
    
    dx = 2.0 * np.sqrt(3) / 2 * 2  # horizontal spacing for flat-top = sqrt(3)*2? 
    # For flat-top hexagons (vertex at left/right): width = 2*side = 2, so centers need to be > 2 apart horizontally
    # But they can be closer if offset vertically
    # Standard hex packing: horizontal spacing = sqrt(3)*side for pointy-top = sqrt(3)
    # For flat-top: horizontal spacing = 1.5*side*2? 
    
    # Let me just use a reasonable spacing and let optimization fix it
    # Start with a loose grid
    spacing_x = 3.5
    spacing_y = 3.0
    
    # 3-4-3-2 arrangement? No, let's do a symmetric hexagonal cluster
    # Center + 6 around + 5 more = 12
    # Or just a 4x3 grid
    
    # Simple: 4 columns x 3 rows
    positions = []
    for row in range(3):
        for col in range(4):
            x = col * spacing_x
            y = row * spacing_y
            if row % 2 == 1:
                x += spacing_x / 2
            positions.append((x, y))
    
    for i, (x, y) in enumerate(positions[:N]):
        inner.append([x, y, 0.0])
    
    # Center the configuration
    xs = [p[0] for p in inner]
    ys = [p[1] for p in inner]
    cx = (min(xs) + max(xs)) / 2
    cy = (min(ys) + max(ys)) / 2
    for p in inner:
        p[0] -= cx
        p[1] -= cy
    
    return inner, 0.0, 0.0, 0.0

def construct():
    global DEADLINE
    
    best_side = 1000.0
    best_config = None
    
    # Initial configuration
    inner, ocx, ocy, oang = initial_config()
    
    # Compute initial side
    side = compute_outer_side(inner, ocx, ocy, oang)
    
    # Check validity
    valid = True
    for i in range(N):
        v1 = hex_vertices(inner[i][0], inner[i][1], inner[i][2], 1.0)
        for j in range(i+1, N):
            v2 = hex_vertices(inner[j][0], inner[j][1], inner[j][2], 1.0)
            if poly_collision(v1, v2):
                valid = False
                break
        if not valid:
            break
    
    if valid:
        best_side = side
        best_config = (inner.copy(), ocx, ocy, oang)
    
    # Simulated Annealing
    current_inner = [list(p) for p in inner]
    current_ocx, current_ocy, current_oang = ocx, ocy, oang
    
    def get_params(inner_list, ocx, ocy, oang):
        params = []
        for p in inner_list:
            params.extend([p[0], p[1], p[2]])
        params.extend([ocx, ocy, oang])
        return np.array(params)
    
    def set_params(params, inner_list):
        for i in range(N):
            inner_list[i][0] = params[3*i]
            inner_list[i][1] = params[3*i + 1]
            inner_list[i][2] = params[3*i + 2]
        ocx = params[3*N]
        ocy = params[3*N + 1]
        oang = params[3*N + 2]
        return ocx, ocy, oang
    
    T = 1.0
    T_min = 0.01
    cooling = 0.999
    max_iter = 50000
    
    current_params = get_params(current_inner, current_ocx, current_ocy, current_oang)
    current_side = compute_outer_side(current_inner, current_ocx, current_ocy, current_oang)
    
    for it in range(max_iter):
        if time.time() > DEADLINE:
            break
        
        # Perturb
        new_params = current_params.copy()
        # Randomly perturb a few inner hexagons
        n_perturb = np.random.randint(1, 4)
        for _ in range(n_perturb):
            idx = np.random.randint(0, N)
            new_params[3*idx + 0] += np.random.randn() * 0.3
            new_params[3*idx + 1] += np.random.randn() * 0.3
            new_params[3*idx + 2] += np.random.randn() * 10.0
        
        # Occasionally perturb outer
        if np.random.rand() < 0.1:
            new_params[3*N + 0] += np.random.randn() * 0.1
            new_params[3*N + 1] += np.random.randn() * 0.1
            new_params[3*N + 2] += np.random.randn() * 5.0
        
        new_inner = []
        for i in range(N):
            new_inner.append([new_params[3*i], new_params[3*i+1], new_params[3*i+2]])
        new_ocx = new_params[3*N]
        new_ocy = new_params[3*N + 1]
        new_oang = new_params[3*N + 2]
        
        new_side = compute_outer_side(new_inner, new_ocx, new_ocy, new_oang)
        
        # Check overlaps
        overlap = False
        for i in range(N):
            v1 = hex_vertices(new_inner[i][0], new_inner[i][1], new_inner[i][2], 1.0)
            for j in range(i+1, N):
                v2 = hex_vertices(new_inner[j][0], new_inner[j][1], new_inner[j][2], 1.0)
                if poly_collision(v1, v2):
                    overlap = True
                    break
            if overlap:
                break
        
        if overlap:
            new_side = 1000.0
        
        delta = new_side - current_side
        if delta < 0 or np.random.rand() < np.exp(-delta / T):
            current_params = new_params
            current_inner = new_inner
            current_ocx, current_ocy, current_oang = new_ocx, new_ocy, new_oang
            current_side = new_side
            if current_side < best_side:
                best_side = current_side
                best_config = ([list(p) for p in current_inner], current_ocx, current_ocy, current_oang)
        
        T *= cooling
        if T < T_min:
            T = T_min
    
    # Nelder-Mead refinement on best config
    if best_config is not None:
        best_inner, b_ocx, b_ocy, b_oang = best_config
        x0 = get_params(best_inner, b_ocx, b_ocy, b_oang)
        
        def obj(x):
            inner_list = []
            for i in range(N):
                inner_list.append([x[3*i], x[3*i+1], x[3*i+2]])
            ocx = x[3*N]
            ocy = x[3*N+1]
            oang = x[3*N+2]
            
            side = compute_outer_side(inner_list, ocx, ocy, oang)
            
            # Check overlaps
            for i in range(N):
                v1 = hex_vertices(inner_list[i][0], inner_list[i][1], inner_list[i][2], 1.0)
                for j in range(i+1, N):
                    v2 = hex_vertices(inner_list[j][0], inner_list[j][1], inner_list[j][2], 1.0)
                    if poly_collision(v1, v2):
                        return 1000.0
            return side
        
        try:
            result = minimize(obj, x0, method='Nelder-Mead', 
                             options={'maxiter': 500, 'xatol': 1e-6, 'fatol': 1e-8, 'adaptive': True})
            if result.fun < best_side:
                best_side = result.fun
                best_inner = []
                for i in range(N):
                    best_inner.append([result.x[3*i], result.x[3*i+1], result.x[3*i+2]])
                b_ocx = result.x[3*N]
                b_ocy = result.x[3*N+1]
                b_oang = result.x[3*N+2]
                best_config = (best_inner, b_ocx, b_ocy, b_oang)
        except:
            pass
    
    # Final validation
    if best_config is None:
        # Fallback: return initial
        return inner, (ocx, ocy), side, oang
    
    final_inner, f_ocx, f_ocy, f_oang = best_config
    
    # Re-verify validity
    if not check_validity(final_inner, f_ocx, f_ocy, f_oang, best_side):
        # Try with slightly larger side
        for margin in [0.01, 0.05, 0.1, 0.5, 1.0]:
            if check_validity(final_inner, f_ocx, f_ocy, f_oang, best_side + margin):
                best_side += margin
                break
        else:
            # Return a safe fallback
            return inner, (ocx, ocy), side, oang
    
    return final_inner, (f_ocx, f_ocy), best_side, f_oang

# EVOLVE-BLOCK-START
def construct():
    return _construct_impl()

def _construct_impl():
    global DEADLINE
    DEADLINE = time.time() + 110.0
    
    best_side = 1000.0
    best_config = None
    
    # Initial configuration
    inner, ocx, ocy, oang = initial_config()
    
    side = compute_outer_side(inner, ocx, ocy, oang)
    
    valid = True
    for i in range(N):
        v1 = hex_vertices(inner[i][0], inner[i][1], inner[i][2], 1.0)
        for j in range(i+1, N):
            v2 = hex_vertices(inner[j][0], inner[j][1], inner[j][2], 1.0)
            if poly_collision(v1, v2):
                valid = False
                break
        if not valid:
            break
    
    if valid:
        best_side = side
        best_config = (inner.copy(), ocx, ocy, oang)
    
    # SA
    current_inner = [list(p) for p in inner]
    current_ocx, current_ocy, current_oang = ocx, ocy, oang
    
    def get_params(inner_list, ocx, ocy, oang):
        params = []
        for p in inner_list:
            params.extend([p[0], p[1], p[2]])
        params.extend([ocx, ocy, oang])
        return np.array(params)
    
    T = 1.0
    T_min = 0.005
    cooling = 0.9995
    max_iter = 100000
    
    current_params = get_params(current_inner, current_ocx, current_ocy, current_oang)
    current_side = compute_outer_side(current_inner, current_ocx, current_ocy, current_oang)
    
    for it in range(max_iter):
        if time.time() > DEADLINE:
            break
        
        new_params = current_params.copy()
        n_perturb = np.random.randint(1, 4)
        for _ in range(n_perturb):
            idx = np.random.randint(0, N)
            new_params[3*idx + 0] += np.random.randn() * 0.2
            new_params[3*idx + 1] += np.random.randn() * 0.2
            new_params[3*idx + 2] += np.random.randn() * 8.0
        
        if np.random.rand() < 0.05:
            new_params[3*N + 0] += np.random.randn() * 0.05
            new_params[3*N + 1] += np.random.randn() * 0.05
            new_params[3*N + 2] += np.random.randn() * 3.0
        
        new_inner = []
        for i in range(N):
            new_inner.append([new_params[3*i], new_params[3*i+1], new_params[3*i+2]])
        new_ocx = new_params[3*N]
        new_ocy = new_params[3*N + 1]
        new_oang = new_params[3*N + 2]
        
        new_side = compute_outer_side(new_inner, new_ocx, new_ocy, new_oang)
        
        overlap = False
        for i in range(N):
            v1 = hex_vertices(new_inner[i][0], new_inner[i][1], new_inner[i][2], 1.0)
            for j in range(i+1, N):
                v2 = hex_vertices(new_inner[j][0], new_inner[j][1], new_inner[j][2], 1.0)
                if poly_collision(v1, v2):
                    overlap = True
                    break
            if overlap:
                break
        
        if overlap:
            new_side = 1000.0
        
        delta = new_side - current_side
        if delta < 0 or np.random.rand() < np.exp(-delta / T):
            current_params = new_params
            current_inner = new_inner
            current_ocx, current_ocy, current_oang = new_ocx, new_ocy, new_oang
            current_side = new_side
            if current_side < best_side:
                best_side = current_side
                best_config = ([list(p) for p in current_inner], current_ocx, current_ocy, current_oang)
        
        T *= cooling
        if T < T_min:
            T = T_min
    
    # NM refinement
    if best_config is not None:
        best_inner, b_ocx, b_ocy, b_oang = best_config
        x0 = get_params(best_inner, b_ocx, b_ocy, b_oang)
        
        def obj(x):
            inner_list = []
            for i in range(N):
                inner_list.append([x[3*i], x[3*i+1], x[3*i+2]])
            ocx = x[3*N]
            ocy = x[3*N+1]
            oang = x[3*N+2]
            
            s = compute_outer_side(inner_list, ocx, ocy, oang)
            
            for i in range(N):
                v1 = hex_vertices(inner_list[i][0], inner_list[i][1], inner_list[i][2], 1.0)
                for j in range(i+1, N):
                    v2 = hex_vertices(inner_list[j][0], inner_list[j][1], inner_list[j][2], 1.0)
                    if poly_collision(v1, v2):
                        return 1000.0
            return s
        
        try:
            result = minimize(obj, x0, method='Nelder-Mead',
                             options={'maxiter': 300, 'xatol': 1e-7, 'fatol': 1e-9, 'adaptive': True})
            if result.fun < best_side:
                best_side = result.fun
                best_inner = []
                for i in range(N):
                    best_inner.append([result.x[3*i], result.x[3*i+1], result.x[3*i+2]])
                b_ocx = result.x[3*N]
                b_ocy = result.x[3*N+1]
                b_oang = result.x[3*N+2]
                best_config = (best_inner, b_ocx, b_ocy, b_oang)
        except:
            pass
    
    if best_config is None:
        return inner, (ocx, ocy), side, oang
    
    final_inner, f_ocx, f_ocy, f_oang = best_config
    
    if not check_validity(final_inner, f_ocx, f_ocy, f_oang, best_side):
        for margin in [0.01, 0.05, 0.1, 0.5, 1.0, 2.0]:
            if check_validity(final_inner, f_ocx, f_ocy, f_oang, best_side + margin):
                best_side += margin
                break
        else:
            return inner, (ocx, ocy), side, oang
    
    return final_inner, (f_ocx, f_ocy), best_side, f_oang
# EVOLVE-BLOCK-END
