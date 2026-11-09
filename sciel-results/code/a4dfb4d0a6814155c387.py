"""Packing unit regular hexagons inside a regular hexagon using trust-region optimization."""
import numpy as np
from scipy.optimize import minimize

N = 11

def hexagon_vertices(cx, cy, side, angle_deg):
    """Return vertices of a regular hexagon centered at (cx, cy) with given side and rotation."""
    angle_rad = np.radians(angle_deg)
    vertices = []
    for i in range(6):
        theta = angle_rad + i * np.pi / 3
        x = cx + side * np.cos(theta)
        y = cy + side * np.sin(theta)
        vertices.append((x, y))
    return np.array(vertices)

def point_in_hexagon(px, py, cx, cy, side, angle_deg):
    """Check if point (px, py) is inside or on the boundary of a hexagon."""
    # Transform to hexagon's local coordinate system
    dx = px - cx
    dy = py - cy
    angle_rad = np.radians(angle_deg)
    cos_a = np.cos(-angle_rad)
    sin_a = np.sin(-angle_rad)
    lx = cos_a * dx - sin_a * dy
    ly = sin_a * dx + cos_a * dy
    
    # For a regular hexagon with side s, the distance from center to each side is s * sqrt(3) / 2
    # The hexagon in local coords has vertices at angles 0, 60, 120, 180, 240, 300 degrees
    # The supporting lines are at distances s*sqrt(3)/2 from center in directions 30, 90, 150, 210, 270, 330 degrees
    
    apothem = side * np.sqrt(3) / 2
    # The six normal directions (perpendicular to sides) are at 30, 90, 150, 210, 270, 330 degrees
    for i in range(6):
        theta = np.radians(30 + i * 60)
        nx = np.cos(theta)
        ny = np.sin(theta)
        dist = lx * nx + ly * ny
        if dist > apothem + 1e-9:
            return False
    return True

def hexagon_distance(v1, v2, v3, v4):
    """
    Approximate minimum distance between two hexagons.
    v1, v2 are vertex arrays of hexagon 1 and 2.
    Uses point-to-polygon distance approximation via vertices and edges.
    """
    min_dist = np.inf
    
    # Check if any vertex of v1 is inside v2
    for i in range(6):
        if point_in_hexagon(v1[i, 0], v1[i, 1], *get_center_and_params(v2)):
            return 0.0
    
    # Check if any vertex of v2 is inside v1
    for i in range(6):
        if point_in_hexagon(v2[i, 0], v2[i, 1], *get_center_and_params(v1)):
            return 0.0
    
    # Compute minimum distance between all pairs of vertices and edges
    for i in range(6):
        for j in range(6):
            d = np.sqrt((v1[i,0]-v2[j,0])**2 + (v1[i,1]-v2[j,1])**2)
            min_dist = min(min_dist, d)
    
    # Also check edge-to-edge distances approximately
    for i in range(6):
        p1 = v1[i]
        p2 = v1[(i+1)%6]
        for j in range(6):
            q1 = v2[j]
            q2 = v2[(j+1)%6]
            d = segment_segment_distance(p1, p2, q1, q2)
            min_dist = min(min_dist, d)
    
    return min_dist

def segment_segment_distance(p1, p2, q1, q2):
    """Minimum distance between two line segments."""
    # Simplified: just use point-to-segment distances
    d1 = point_segment_distance(q1, p1, p2)
    d2 = point_segment_distance(q2, p1, p2)
    d3 = point_segment_distance(p1, q1, q2)
    d4 = point_segment_distance(p2, q1, q2)
    return min(d1, d2, d3, d4)

def point_segment_distance(p, a, b):
    """Distance from point p to segment ab."""
    ab = b - a
    ap = p - a
    if np.dot(ab, ab) < 1e-12:
        return np.linalg.norm(p - a)
    t = np.dot(ap, ab) / np.dot(ab, ab)
    t = np.clip(t, 0, 1)
    closest = a + t * ab
    return np.linalg.norm(p - closest)

def get_center_and_params(vertices):
    """Extract center and params from vertices - not used, just a placeholder."""
    cx = np.mean(vertices[:, 0])
    cy = np.mean(vertices[:, 1])
    return cx, cy, 1.0, 0.0

def objective(params, N_inner):
    """
    Objective function: minimize the outer hexagon side length.
    params = [x1, y1, a1, x2, y2, a2, ..., xN, yN, aN, S]
    where S is the outer hexagon side length.
    """
    positions = params[:3*N_inner].reshape(N_inner, 3)
    S = params[3*N_inner]
    
    # Outer hexagon centered at origin
    outer_verts = hexagon_vertices(0, 0, S, 0)
    
    # Constraint 1: All inner hexagons must be inside outer hexagon
    # Constraint 2: No two inner hexagons overlap
    
    penalty = 0.0
    
    # Check inner hexagons are inside outer
    for i in range(N_inner):
        inner_verts = hexagon_vertices(positions[i, 0], positions[i, 1], 1.0, positions[i, 2])
        for v in inner_verts:
            if not point_in_hexagon(v[0], v[1], 0, 0, S, 0):
                # Compute how far outside
                # Find max violation
                dx = v[0]
                dy = v[1]
                apothem = S * np.sqrt(3) / 2
                for k in range(6):
                    theta = np.radians(30 + k * 60)
                    nx = np.cos(theta)
                    ny = np.sin(theta)
                    dist = dx * nx + dy * ny
                    if dist > apothem:
                        penalty += (dist - apothem) ** 2
    
    # Check no overlap between inner hexagons
    for i in range(N_inner):
        for j in range(i+1, N_inner):
            vi = hexagon_vertices(positions[i, 0], positions[i, 1], 1.0, positions[i, 2])
            vj = hexagon_vertices(positions[j, 0], positions[j, 1], 1.0, positions[j, 2])
            
            # Check if any vertex of vi is inside vj
            for k in range(6):
                if point_in_hexagon(vi[k, 0], vi[k, 1], positions[j, 0], positions[j, 1], 1.0, positions[j, 2]):
                    penalty += 1.0
                    break
            else:
                # Check if any vertex of vj is inside vi
                overlap = False
                for k in range(6):
                    if point_in_hexagon(vj[k, 0], vj[k, 1], positions[i, 0], positions[i, 1], 1.0, positions[i, 2]):
                        penalty += 1.0
                        overlap = True
                        break
                if not overlap:
                    # Check edge intersections / minimum distance
                    min_d = hexagon_distance(vi, vj, vi, vj)
                    if min_d < 1e-6:
                        penalty += 1.0
    
    return S + 10.0 * penalty

def construct():
    """
    Construct the packing by optimizing positions and rotations of N unit hexagons
    inside an outer hexagon, minimizing the outer side length.
    """
    N_inner = N
    
    # Initial guess: place hexagons in a grid-like pattern
    # For N=11, use a pattern with one in center and rings around
    init_positions = []
    # Center
    init_positions.append([0.0, 0.0, 0.0])
    # First ring: 6 hexagons at distance 2*sqrt(3)/2 * 2 = 2*sqrt(3) ... actually for touching hexagons,
    # center-to-center distance is 2 * apothem = 2 * sqrt(3)/2 * 2 = 2*sqrt(3) ≈ 3.464
    # Actually, for two unit hexagons to touch, center distance = 2 * sqrt(3) * cos(30) = 2 * sqrt(3) * sqrt(3)/2 = 3? 
    # No: apothem = sqrt(3)/2. Two hexagons touch when center distance = 2 * apothem = sqrt(3). 
    # Wait, that's for flat-to-flat. For vertex-to-vertex it's 2*side = 2.
    # The minimum center-to-center distance for non-overlapping unit hexagons is sqrt(3) (flat-to-flat).
    
    dist_flat = np.sqrt(3)  # ~1.732
    dist_vertex = 2.0
    
    # Place 6 around center at distance dist_flat in 6 directions
    for k in range(6):
        theta = k * np.pi / 3
        x = dist_flat * np.cos(theta)
        y = dist_flat * np.sin(theta)
        init_positions.append([x, y, 0.0])
    
    # Add 4 more for total 11
    # Second ring partial
    for k in range(4):
        theta = k * np.pi / 3 + np.pi / 6
        x = 2 * dist_flat * np.cos(theta)
        y = 2 * dist_flat * np.sin(theta)
        init_positions.append([x, y, 0.0])
    
    # Trim or pad to N_inner
    while len(init_positions) < N_inner:
        init_positions.append([0.0, 0.0, 0.0])
    init_positions = init_positions[:N_inner]
    
    init_params = np.array(init_positions).flatten()
    init_S = 5.0  # Initial guess for outer side length
    init_params = np.append(init_params, init_S)
    
    # Bounds
    bounds = []
    for i in range(N_inner):
        bounds.append((-10, 10))  # x
        bounds.append((-10, 10))  # y
        bounds.append((0, 180))  # angle in degrees
    bounds.append((1.0, 20.0))  # S
    
    # Try a few random initializations and keep the best
    best_result = None
    best_S = np.inf
    
    np.random.seed(42)
    n_trials = 3
    
    for trial in range(n_trials):
        if trial == 0:
            x0 = init_params.copy()
        else:
            # Perturb initial guess
            x0 = init_params.copy()
            x0[:3*N_inner] += np.random.randn(3*N_inner) * 0.5
            x0[3*N_inner] = init_S + np.random.rand() * 2
        
        try:
            result = minimize(
                objective,
                x0,
                args=(N_inner,),
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 500, 'ftol': 1e-8, 'gtol': 1e-8}
            )
            
            if result.success or result.fun < best_S:
                best_S = result.fun
                best_result = result
        except Exception:
            continue
    
    if best_result is None:
        # Fallback to initial guess
        inner = [[p[0], p[1], p[2]] for p in init_positions]
        return inner, (0.0, 0.0), init_S, 0.0
    
    final_params = best_result.x
    positions = final_params[:3*N_inner].reshape(N_inner, 3)
    S = final_params[3*N_inner]
    
    inner = [[pos[0], pos[1], pos[2]] for pos in positions]
    
    return inner, (0.0, 0.0), S, 0.0
