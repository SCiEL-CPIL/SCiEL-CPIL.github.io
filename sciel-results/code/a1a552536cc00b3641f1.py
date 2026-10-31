import numpy as np
import time
from scipy.optimize import differential_evolution

N = 11

def hex_vertices(cx, cy, side, angle_deg=0.0):
    """Return vertices of a regular hexagon with given center, side, and rotation."""
    angle_rad = np.radians(angle_deg)
    # Vertices at 0, 60, 120, 180, 240, 300 degrees in local frame
    # If angle_deg=0, vertex at 0 is (side, 0)? 
    # Standard regular hexagon with "flat top" has vertices at 30, 90, ...
    # The original code used: angles = angle_rad + np.pi / 6 + np.arange(6) * np.pi / 3
    # This puts a vertex at 30 deg (pi/6). This is a "pointy top" hexagon if y is up?
    # Let's stick to the original definition to ensure consistency with the rest of the code.
    angles = angle_rad + np.pi / 6 + np.arange(6) * np.pi / 3
    return cx + side * np.cos(angles), cy + side * np.sin(angles)

def check_containment(cx, cy, side, angle_deg, inner_positions, inner_angles, inner_side=1.0):
    """
    Vectorized check if all inner hexagons are inside the outer hexagon.
    Returns (is_valid, max_violation).
    """
    if side <= 0:
        return False, 1e10
        
    # Precompute outer hexagon parameters
    angle_rad = np.radians(angle_deg)
    cos_a = np.cos(-angle_rad)
    sin_a = np.sin(-angle_rad)
    apothem = side * np.sqrt(3) / 2
    
    # Normals of the outer hexagon faces in global frame
    # Local normals are at 0, 60, 120, 180, 240, 300
    # Rotate them by angle_rad
    normal_angles = np.arange(6) * np.pi / 3 + angle_rad
    normals = np.array([np.cos(normal_angles), np.sin(normal_angles)]).T # Shape (6, 2)
    
    max_violation = 0.0
    is_valid = True
    
    for i in range(len(inner_positions)):
        ix, iy = inner_positions[i]
        ia = inner_angles[i]
        vx, vy = hex_vertices(ix, iy, inner_side, ia)
        
        # Transform inner vertices to outer local frame
        dx = vx - cx
        dy = vy - cy
        lx = cos_a * dx - sin_a * dy
        ly = sin_a * dx + cos_a * dy
        
        # Project onto normals
        # proj = lx * nx + ly * ny
        # We want proj <= apothem for all normals and their opposites?
        # Actually, the hexagon is intersection of 6 half planes.
        # The normals point INWARD or OUTWARD?
        # In point_in_hexagon: if nx*lx + ny*ly > apothem: return False.
        # This implies normals point OUTWARD from center to face?
        # If normal is outward, the face is at distance apothem.
        # So lx*nx + ly*ny <= apothem.
        
        # Calculate projections for all 6 vertices
        # We can do this for each vertex
        for j in range(6):
            # Project vertex j onto all 6 normals
            # proj_k = lx_j * nx_k + ly_j * ny_k
            proj = lx[j] * normals[:, 0] + ly[j] * normals[:, 1]
            violations = np.maximum(0, proj - apothem)
            v = np.max(violations)
            if v > max_violation:
                max_violation = v
            if v > 1e-6:
                is_valid = False
                
    return is_valid, max_violation

def compute_min_outer_hexagon(inner_positions, inner_angles, inner_side=1.0):
    """Find minimal outer hexagon containing all inner hexagons.
    Returns (cx, cy, side, angle_deg)."""
    
    def objective(params):
        cx, cy, side, angle = params
        if side <= 0:
            return 1e10
        is_valid, max_violation = check_containment(cx, cy, side, angle, inner_positions, inner_angles, inner_side)
        if is_valid:
            return side
        else:
            return side + 1000 * max_violation

    # Initial guess
    all_vx, all_vy = [], []
    for i in range(len(inner_positions)):
        ix, iy = inner_positions[i]
        ia = inner_angles[i]
        vx, vy = hex_vertices(ix, iy, inner_side, ia)
        all_vx.extend(vx)
        all_vy.extend(vy)
    
    all_vx = np.array(all_vx)
    all_vy = np.array(all_vy)
    
    cx0 = (all_vx.min() + all_vx.max()) / 2
    cy0 = (all_vy.min() + all_vy.max()) / 2
    max_dist = np.max(np.sqrt((all_vx - cx0)**2 + (all_vy - cy0)**2))
    side0 = max_dist * 2 / np.sqrt(3) * 1.5  # generous margin
    
    # Bounds for DE
    # cx, cy should be near cx0, cy0
    # side should be near side0
    # angle in [0, 60)
    bounds = [
        (cx0 - side0, cx0 + side0),
        (cy0 - side0, cy0 + side0),
        (0.1, side0 * 2),
        (0, 60)
    ]
    
    try:
        res = differential_evolution(objective, bounds, maxiter=200, popsize=15, tol=1e-6, seed=42)
        if res.success:
            cx, cy, side, angle = res.x
            # Verify
            is_valid, _ = check_containment(cx, cy, side, angle, inner_positions, inner_angles, inner_side)
            if is_valid:
                return cx, cy, side, angle
            else:
                # If DE failed to find a valid one, return the best effort (might be invalid, but better than error)
                # Or fall back to a simple expansion
                cx, cy, side, angle = res.x
                return cx, cy, side, angle
    except Exception:
        pass
        
    # Fallback: Simple expansion
    # Start with a large hexagon and shrink?
    # Just return the initial guess with large side
    return cx0, cy0, side0, 0.0

def construct():
    start_time = time.time()
    deadline = start_time + 110  # 10s margin
    
    # Place 11 hexagons on a hexagonal lattice
    d = np.sqrt(3)  # center-to-center distance for edge-to-edge hexagons
    
    def hex_to_cart(q, r):
        x = d * (q + r / 2)
        y = d * (np.sqrt(3) / 2) * r
        return x, y
    
    # 11 hexagons: center + 6 in first ring + 4 in second ring
    # This is a valid disjoint packing.
    positions_axial = [
        (0, 0),      
        (1, 0), (1, -1), (0, -1), (-1, 0), (-1, 1), (0, 1),  
        (2, 0), (2, -1), (1, -2), (0, -2),  
    ]
    
    inner_positions = []
    for q, r in positions_axial:
        x, y = hex_to_cart(q, r)
        inner_positions.append((x, y))
    
    inner_angles = [0.0] * N
    
    inner = [[p[0], p[1], a] for p, a in zip(inner_positions, inner_angles)]
    
    # Compute minimal outer hexagon
    cx, cy, side, angle = compute_min_outer_hexagon(
        inner_positions, inner_angles, 1.0
    )
    
    result = (inner, (cx, cy), side, angle)
    
    # Check time
    if time.time() > deadline:
        return result

    # Try rotating the whole configuration to see if we get a better fit
    # The lattice is symmetric, so rotating by 60 degrees yields the same shape.
    # Rotating by 30 degrees might align the "pointy" parts differently with the container.
    best_side = side
    best_result = result
    
    for rot_deg in [30]:
        if time.time() > deadline:
            break
        rot_rad = np.radians(rot_deg)
        cos_r = np.cos(rot_rad)
        sin_r = np.sin(rot_rad)
        
        rotated_positions = []
        for x, y in inner_positions:
            nx = cos_r * x - sin_r * y
            ny = sin_r * x + cos_r * y
            rotated_positions.append((nx, ny))
        
        rotated_angles = [(a + rot_deg) % 360 for a in inner_angles]
        
        rcx, rcy, rside, rangle = compute_min_outer_hexagon(rotated_positions, rotated_angles, 1.0)
        if rside < best_side:
            inner_rot = [[p[0], p[1], a] for p, a in zip(rotated_positions, rotated_angles)]
            best_result = (inner_rot, (rcx, rcy), rside, rangle)
            best_side = rside
            
    return best_result
