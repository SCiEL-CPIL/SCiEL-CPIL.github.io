import numpy as np
from scipy.optimize import minimize, differential_evolution
import time

N = 12

def _hex_vertices(cx, cy, side, angle_deg):
    """Return 6 vertices of a regular hexagon centered at (cx,cy), side length, rotated by angle_deg."""
    angle = np.radians(angle_deg)
    vertices = []
    for i in range(6):
        theta = angle + i * np.pi / 3
        x = cx + side * np.cos(theta)
        y = cy + side * np.sin(theta)
        vertices.append((x, y))
    return np.array(vertices)

def _hex_face_normals(angle_deg):
    """Return 6 unit outward normals for a regular hexagon with given rotation."""
    angle = np.radians(angle_deg)
    normals = []
    for i in range(6):
        theta = angle + i * np.pi / 3 + np.pi / 6  # normals are offset by 30 degrees from vertices
        normals.append((np.cos(theta), np.sin(theta)))
    return np.array(normals)

def _point_in_hexagon(px, py, cx, cy, side, angle_deg):
    """Check if point (px,py) is inside hexagon. Returns True if inside or on boundary."""
    r = side * np.sqrt(3) / 2  # inradius
    normals = _hex_face_normals(angle_deg)
    dx = px - cx
    dy = py - cy
    for nx, ny in normals:
        if dx * nx + dy * ny > r + 1e-9:
            return False
    return True

def _min_enclosing_hexagon_side(points, center, angle_deg):
    """Given points and a hexagon center/angle, find minimum side length to contain all points."""
    normals = _hex_face_normals(angle_deg)
    max_proj = 0
    for px, py in points:
        dx = px - center[0]
        dy = py - center[1]
        for nx, ny in normals:
            proj = dx * nx + dy * ny
            if proj > max_proj:
                max_proj = proj
    # side = inradius / (sqrt(3)/2) = max_proj * 2 / sqrt(3)
    return max_proj * 2.0 / np.sqrt(3)

def _all_hexagon_extremes(inner_hexes):
    """For each inner hexagon, compute the max projection in 6 directions of the outer hexagon.
    Actually, we need the max over all points of each inner hexagon in each outer normal direction.
    For a convex polygon, max projection in a direction is achieved at a vertex."""
    pass

def _objective(params):
    """Objective: minimize outer hexagon side length given inner hex positions/angles and outer center/angle."""
    n = len(params)
    # params: [x1,y1,a1, x2,y2,a2, ..., xN,yN,aN, cx, cy, ca]
    inner = []
    for i in range(N):
        x = params[i*3]
        y = params[i*3+1]
        a = params[i*3+2]
        inner.append((x, y, a))
    cx = params[N*3]
    cy = params[N*3+1]
    ca = params[N*3+2]
    
    # Get all vertices of all inner hexagons
    all_vertices = []
    for (x, y, a) in inner:
        verts = _hex_vertices(x, y, 1.0, a)
        all_vertices.extend(verts)
    
    # Compute max projection in each of the 6 outer normal directions
    normals = _hex_face_normals(ca)
    max_proj = 0
    for (px, py) in all_vertices:
        dx = px - cx
        dy = py - cy
        for (nx, ny) in normals:
            proj = dx * nx + dy * ny
            if proj > max_proj:
                max_proj = proj
    
    side = max_proj * 2.0 / np.sqrt(3)
    
    # Check disjointness: for each pair of inner hexagons, verify no overlap
    # Use vertex-edge or SAT test
    for i in range(N):
        for j in range(i+1, N):
            xi, yi, ai = inner[i]
            xj, yj, aj = inner[j]
            vi = _hex_vertices(xi, yi, 1.0, ai)
            vj = _hex_vertices(xj, yj, 1.0, aj)
            # Simple check: if any vertex of one is inside the other, they overlap
            # But this isn't sufficient for edge-edge intersection. Use a more robust check.
            # For now, use center distance as a heuristic lower bound
            dist = np.sqrt((xi-xj)**2 + (yi-yj)**2)
            if dist < 1.7:  # too close, likely overlapping
                return side + 1000 * (1.7 - dist)
    
    return side

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    # Initial arrangement: hexagonal lattice
    # Place 12 hexagons in a compact hexagonal pattern
    # Hex lattice spacing: sqrt(3) for edge-sharing
    spacing = np.sqrt(3)
    
    # Arrangement: rows of hexagons
    # Row 0: 3 hexagons
    # Row 1: 4 hexagons  
    # Row 2: 3 hexagons
    # Row 3: 2 hexagons
    # Total: 12
    # Or: 2, 3, 4, 3 = 12
    # Let's try a more standard hexagonal cluster
    
    # Hexagonal lattice positions (axial coordinates)
    # For a hex with 12 cells, use a shape like:
    #   . X X .
    #   X X X X
    #   . X X .
    # That's 1+2+3+2+1 = 9, not 12
    # Try:
    #   X X X
    #   X X X X
    #   X X X
    #   . X .
    # That's 3+4+3+1 = 11
    # Try 4x3 grid:
    #   X X X X
    #   X X X X
    #   X X X X
    # That's 12, but rectangular
    
    # Better: use a hexagonal arrangement
    # Axial coordinates for hexagonal lattice
    # Let's use offset coordinates
    # Row 0: 4 hexagons
    # Row 1: 4 hexagons (shifted)
    # Row 2: 4 hexagons
    # With hex lattice, row spacing = 1.5 (for unit hexagons with side 1, the vertical distance between rows is 1.5*side for pointy-top, or sqrt(3)/2*side for flat-top)
    
    # For flat-top hexagons (vertices at 0, 60, 120, 180, 240, 300 degrees):
    # Width = 2*side = 2, Height = sqrt(3)*side = sqrt(3)
    # Horizontal spacing between adjacent hexagons in same row: 1.5*side = 1.5
    # Vertical spacing between rows: sqrt(3)/2 * side = sqrt(3)/2 ≈ 0.866
    # But for disjoint hexagons, we need more space
    
    # Actually, for two flat-top unit hexagons to share an edge:
    # Centers are sqrt(3) apart horizontally? No...
    # Let me just use a simple grid and optimize
    
    # Simple initial: 4x3 grid with spacing 2 (diameter)
    positions = []
    for row in range(3):
        for col in range(4):
            x = col * 2.0
            y = row * np.sqrt(3)
            positions.append((x, y, 0.0))
    
    # Center of mass
    cx0 = np.mean([p[0] for p in positions])
    cy0 = np.mean([p[1] for p in positions])
    
    # Initial outer hexagon
    ca0 = 0.0
    
    # Build initial params
    params0 = []
    for (x, y, a) in positions:
        params0.extend([x, y, a])
    params0.extend([cx0, cy0, ca0])
    
    best_params = params0
    best_side = _objective(params0)
    
    # Try multiple initial configurations
    configs = []
    
    # Config 1: 4x3 grid
    configs.append(params0.copy())
    
    # Config 2: hexagonal cluster
    hex_positions = []
    # Center hex + 5 around + 6 more
    # Axial coords: (0,0), (1,0), (0,1), (-1,1), (-1,0), (0,-1), (1,-1), (2,0), (1,1), (0,2), (-1,2), (-2,1)
    axial = [(0,0), (1,0), (0,1), (-1,1), (-1,0), (0,-1), (1,-1), (2,0), (1,1), (0,2), (-1,2), (-2,1)]
    for (q, r) in axial:
        # Convert axial to cartesian for flat-top hexagons
        x = 1.5 * q
        y = np.sqrt(3) * (r + q/2.0)
        hex_positions.append((x, y, 0.0))
    
    cx0 = np.mean([p[0] for p in hex_positions])
    cy0 = np.mean([p[1] for p in hex_positions])
    params1 = []
    for (x, y, a) in hex_positions:
        params1.extend([x, y, a])
    params1.extend([cx0, cy0, 0.0])
    configs.append(params1)
    
    # Config 3: 2x6 grid
    grid26 = []
    for row in range(2):
        for col in range(6):
            x = col * 2.0
            y = row * np.sqrt(3)
            grid26.append((x, y, 0.0))
    cx0 = np.mean([p[0] for p in grid26])
    cy0 = np.mean([p[1] for p in grid26])
    params2 = []
    for (x, y, a) in grid26:
        params2.extend([x, y, a])
    params2.extend([cx0, cy0, 0.0])
    configs.append(params2)
    
    # Config 4: 3x4 grid
    grid34 = []
    for row in range(4):
        for col in range(3):
            x = col * 2.0
            y = row * np.sqrt(3)
            grid34.append((x, y, 0.0))
    cx0 = np.mean([p[0] for p in grid34])
    cy0 = np.mean([p[1] for p in grid34])
    params3 = []
    for (x, y, a) in grid34:
        params3.extend([x, y, a])
    params3.extend([cx0, cy0, 0.0])
    configs.append(params3)
    
    # Config 5: rotated hexagonal cluster
    hex_pos_rot = []
    for (q, r) in axial:
        x = 1.5 * q
        y = np.sqrt(3) * (r + q/2.0)
        hex_pos_rot.append((x, y, 30.0))  # 30 degree rotation
    cx0 = np.mean([p[0] for p in hex_pos_rot])
    cy0 = np.mean([p[1] for p in hex_pos_rot])
    params4 = []
    for (x, y, a) in hex_pos_rot:
        params4.extend([x, y, a])
    params4.extend([cx0, cy0, 0.0])
    configs.append(params4)
    
    # Optimize each config
    for idx, init_params in enumerate(configs):
        if time.time() > deadline - 20:
            break
        
        # Use L-BFGS-B for local optimization
        try:
            result = minimize(
                _objective,
                init_params,
                method='L-BFGS-B',
                options={'maxiter': 200, 'ftol': 1e-10, 'gtol': 1e-8}
            )
            if result.fun < best_side:
                best_side = result.fun
                best_params = result.x
        except Exception:
            pass
        
        # Also try a few random perturbations
        for trial in range(3):
            if time.time() > deadline - 15:
                break
            perturbed = init_params.copy()
            for i in range(len(perturbed)):
                perturbed[i] += np.random.randn() * 0.5
            try:
                result = minimize(
                    _objective,
                    perturbed,
                    method='L-BFGS-B',
                    options={'maxiter': 100, 'ftol': 1e-10, 'gtol': 1e-8}
                )
                if result.fun < best_side:
                    best_side = result.fun
                    best_params = result.x
            except Exception:
                pass
    
    # Extract final result
    inner = []
    for i in range(N):
        x = best_params[i*3]
        y = best_params[i*3+1]
        a = best_params[i*3+2]
        inner.append([x, y, a])
    cx = best_params[N*3]
    cy = best_params[N*3+1]
    ca = best_params[N*3+2]
    
    # Compute the actual side length
    all_vertices = []
    for (x, y, a) in inner:
        verts = _hex_vertices(x, y, 1.0, a)
        all_vertices.extend(verts)
    
    normals = _hex_face_normals(ca)
    max_proj = 0
    for (px, py) in all_vertices:
        dx = px - cx
        dy = py - cy
        for (nx, ny) in normals:
            proj = dx * nx + dy * ny
            if proj > max_proj:
                max_proj = proj
    
    side = max_proj * 2.0 / np.sqrt(3)
    
    # Verify disjointness more carefully
    # If there's overlap, add a penalty
    for i in range(N):
        for j in range(i+1, N):
            xi, yi, ai = inner[i]
            xj, yj, aj = inner[j]
            dist = np.sqrt((xi-xj)**2 + (yi-yj)**2)
            if dist < 1.7:
                # Overlap likely, inflate side
                side += 100 * (1.7 - dist)
    
    return inner, (cx, cy), side, ca

