"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from scipy.optimize import minimize

N = 11

# EVOLVE-BLOCK-START

def hexagon_vertices(cx, cy, s, angle_deg):
    """Return vertices of a regular hexagon with center (cx,cy), side s, rotation angle_deg."""
    angle = np.radians(angle_deg)
    vertices = []
    for i in range(6):
        theta = angle + i * np.pi / 3
        x = cx + s * np.cos(theta)
        y = cy + s * np.sin(theta)
        vertices.append([x, y])
    return np.array(vertices)

def polygon_area(vertices):
    """Shoelace formula for area of a polygon."""
    n = len(vertices)
    if n < 3:
        return 0.0
    area = 0.0
    for i in range(n):
        j = (i + 1) % n
        area += vertices[i][0] * vertices[j][1] - vertices[j][0] * vertices[i][1]
    return abs(area) / 2.0

def polygons_overlap(v1, v2, tol=1e-9):
    """Check if two convex polygons overlap using SAT (Separating Axis Theorem)."""
    n1 = len(v1)
    n2 = len(v2)
    
    # Get all edge normals
    axes = []
    for i in range(n1):
        j = (i + 1) % n1
        edge = v1[j] - v1[i]
        normal = np.array([-edge[1], edge[0]])
        norm = np.linalg.norm(normal)
        if norm > 1e-12:
            normal /= norm
            axes.append(normal)
    for i in range(n2):
        j = (i + 1) % n2
        edge = v2[j] - v2[i]
        normal = np.array([-edge[1], edge[0]])
        norm = np.linalg.norm(normal)
        if norm > 1e-12:
            normal /= norm
            axes.append(normal)
    
    for axis in axes:
        # Project both polygons onto the axis
        proj1 = np.dot(v1, axis)
        proj2 = np.dot(v2, axis)
        
        min1, max1 = np.min(proj1), np.max(proj1)
        min2, max2 = np.min(proj2), np.max(proj2)
        
        # Check for separation
        if max1 < min2 + tol or max2 < min1 + tol:
            return False
    
    return True

def contains_point_in_hexagon(pt, center, s, angle_deg):
    """Check if point pt is inside hexagon with given center, side, and angle."""
    # Transform point to hexagon's local coordinate system
    angle = np.radians(angle_deg)
    cos_a = np.cos(angle)
    sin_a = np.sin(angle)
    
    dx = pt[0] - center[0]
    dy = pt[1] - center[1]
    
    # Rotate point back
    local_x = cos_a * dx + sin_a * dy
    local_y = -sin_a * dx + cos_a * dy
    
    # Regular hexagon with side s has inradius r = s * sqrt(3)/2
    # In local coords, the hexagon is defined by |local_y| <= r and appropriate x constraints
    r = s * np.sqrt(3) / 2
    
    if abs(local_y) > r + 1e-9:
        return False
    
    # For the remaining, check the x bounds which depend on y
    # The hexagon in local coords has vertices at angles 0, 60, 120, 180, 240, 300
    # with radius s (circumradius = side length for regular hexagon)
    # Actually for a regular hexagon, circumradius = side length
    # The width in x direction at height y is:
    # For |y| <= r: x_max = s - |y|/sqrt(3) ... let me think more carefully
    
    # A regular hexagon with circumradius R (= side s) has:
    # Vertices at (R, 0), (R/2, R*sqrt(3)/2), (-R/2, R*sqrt(3)/2), (-R, 0), (-R/2, -R*sqrt(3)/2), (R/2, -R*sqrt(3)/2)
    # Inradius = R * sqrt(3)/2
    
    # For a given |local_y| = y_abs <= r:
    # The x boundary is: x_max = R - y_abs/sqrt(3) when in the "flat" sides region
    # Actually, let's just use the vertex-based check
    
    # Simpler: check against all edges using half-plane tests
    verts = hexagon_vertices(0, 0, s, 0)  # local frame
    for i in range(6):
        j = (i + 1) % 6
        edge = verts[j] - verts[i]
        normal = np.array([-edge[1], edge[0]])
        # Ensure normal points inward
        mid = (verts[i] + verts[j]) / 2
        if np.dot(normal, -mid) < 0:
            normal = -normal
        if np.dot(normal, [local_x, local_y]) < -1e-9:
            return False
    return True

def compute_max_containing_side(inner_centers, inner_angles, outer_center, outer_angle, s_lower, s_upper, max_iter=200, tol=1e-8):
    """
    Given fixed inner hexagon positions/angles and outer center/angle,
    find the minimum outer side length such that all inner hexagons fit inside.
    Uses bisection on the outer side length.
    """
    # For a given outer side length S, check if all inner hexagons are contained
    def fits(S):
        outer_verts = hexagon_vertices(outer_center[0], outer_center[1], S, outer_angle)
        
        for i in range(len(inner_centers)):
            c = inner_centers[i]
            a = inner_angles[i]
            inner_verts = hexagon_vertices(c[0], c[1], 1.0, a)
            
            # Check if all vertices of inner hexagon are inside outer hexagon
            for v in inner_verts:
                if not contains_point_in_hexagon(v, outer_center, S, outer_angle):
                    return False
        return True
    
    # Bisection
    lo = s_lower
    hi = s_upper
    
    if not fits(hi):
        # Need to expand further
        hi = s_lower * 2
        while not fits(hi) and hi < 100:
            hi *= 2
        if not fits(hi):
            return hi
    
    for _ in range(max_iter):
        if time.time() > 145:
            break
        mid = (lo + hi) / 2
        if fits(mid):
            hi = mid
        else:
            lo = mid
        if hi - lo < tol:
            break
    
    return hi

def check_packing(inner, outer_center, outer_side, outer_angle, margin=0.01):
    """Verify that the packing is valid: no overlaps, all inside outer hexagon."""
    # Check outer containment
    for i in range(len(inner)):
        x, y, ang = inner[i]
        inner_verts = hexagon_vertices(x, y, 1.0, ang)
        for v in inner_verts:
            if not contains_point_in_hexagon(v, outer_center, outer_side, outer_angle):
                return False
    
    # Check non-overlap between inner hexagons
    for i in range(len(inner)):
        for j in range(i + 1, len(inner)):
            x1, y1, a1 = inner[i]
            x2, y2, a2 = inner[j]
            v1 = hexagon_vertices(x1, y1, 1.0, a1)
            v2 = hexagon_vertices(x2, y2, 1.0, a2)
            if polygons_overlap(v1, v2):
                return False
    
    return True

def generate_hexagonal_lattice_config(N):
    """Generate a hexagonal lattice arrangement of N unit hexagons."""
    # For N hexagons, arrange in a roughly circular hexagonal pattern
    # Use a hexagonal close-packed arrangement
    
    positions = []
    angles = [0.0] * N
    
    # For a hexagonal lattice, the distance between centers of adjacent hexagons is 2*inradius = 2*sqrt(3)
    dist = 2 * np.sqrt(3)
    
    # Generate points in hexagonal lattice coordinates
    # Start from center and add rings
    if N == 1:
        positions = [[0.0, 0.0]]
    else:
        # Hexagonal lattice: points at (i*dx, j*dy) where i,j integers
        # Actually for hex close pack: 
        # Row 0: y=0, x = 0, dist, 2*dist, ...
        # Row 1: y=dist*sqrt(3)/2, x = dist/2, 3*dist/2, ...
        # Row -1: y=-dist*sqrt(3)/2, x = dist/2, 3*dist/2, ...
        
        # Generate in order of distance from origin
        candidates = []
        max_ring = 5
        for j in range(-max_ring, max_ring + 1):
            for i in range(-max_ring, max_ring + 1):
                x = i * dist + (j % 2) * dist / 2
                y = j * dist * np.sqrt(3) / 2
                candidates.append((x, y, i**2 + j**2 + i*j))  # hex distance squared approx
        
        # Sort by distance from origin
        candidates.sort(key=lambda c: c[0]**2 + c[1]**2)
        
        # Take first N
        for k in range(min(N, len(candidates))):
            positions.append([candidates[k][0], candidates[k][1]])
    
    return np.array(positions), angles

def optimize_packing(N, deadline):
    """Main optimization routine."""
    
    # Generate initial configuration
    init_positions, init_angles = generate_hexagonal_lattice_config(N)
    
    # Initial outer side estimate
    # Compute bounding box of inner hexagons
    all_points = []
    for i in range(N):
        v = hexagon_vertices(init_positions[i][0], init_positions[i][1], 1.0, init_angles[i])
        all_points.extend(v.tolist())
    all_points = np.array(all_points)
    
    # Estimate outer side as max distance from center
    center_est = (all_points[0] + all_points[-1]) / 2 if len(all_points) > 0 else np.array([0.0, 0.0])
    max_dist = np.max(np.linalg.norm(all_points - np.array([0.0, 0.0]), axis=1))
    initial_outer_side = max_dist + 2.0  # some margin
    
    # Try several outer angles
    best_result = None
    best_side = float('inf')
    
    outer_angles_to_try = [0.0, 30.0, 15.0, 45.0]
    
    for outer_angle in outer_angles_to_try:
        if time.time() > deadline - 20:
            break
        
        # Optimize inner positions and outer center for this outer angle
        # Decision variables: 2*N positions + 2 outer center + 1 outer side
        # Minimize outer side subject to non-overlap and containment
        
        n_vars = 2 * N + 3  # 2N positions, 2 outer center, 1 outer side
        
        def unpack(x):
            positions = x[:2*N].reshape(N, 2)
            outer_cx = x[2*N]
            outer_cy = x[2*N + 1]
            outer_side = x[2*N + 2]
            return positions, (outer_cx, outer_cy), outer_side
        
        def objective(x):
            positions, outer_center, outer_side = unpack(x)
            
            # Penalties for violations
            penalty = 0.0
            
            # Containment penalty
            for i in range(N):
                inner_verts = hexagon_vertices(positions[i][0], positions[i][1], 1.0, init_angles[i])
                for v in inner_verts:
                    if not contains_point_in_hexagon(v, outer_center, outer_side, outer_angle):
                        # Distance outside
                        penalty += 1.0
            
            # Non-overlap penalty
            for i in range(N):
                for j in range(i + 1, N):
                    v1 = hexagon_vertices(positions[i][0], positions[i][1], 1.0, init_angles[i])
                    v2 = hexagon_vertices(positions[j][0], positions[j][1], 1.0, init_angles[j])
                    if polygons_overlap(v1, v2):
                        penalty += 1.0
            
            return outer_side + penalty * 10.0
        
        # Initial guess
        x0 = np.zeros(n_vars)
        x0[:2*N] = init_positions.flatten()
        x0[2*N] = 0.0
        x0[2*N + 1] = 0.0
        x0[2*N + 2] = initial_outer_side
        
        # Bounds
        bounds = []
        for i in range(2 * N):
            bounds.append((-50.0, 50.0))
        bounds.append((-50.0, 50.0))
        bounds.append((-50.0, 50.0))
        bounds.append((1.5, 50.0))
        
        # Try L-BFGS-B
        try:
            result = minimize(objective, x0, method='L-BFGS-B', bounds=bounds,
                            options={'maxiter': 300, 'ftol': 1e-6})
            if result.success or result.fun < best_side:
                positions, outer_center, outer_side = unpack(result.x)
                inner_list = [[positions[i][0], positions[i][1], init_angles[i]] for i in range(N)]
                
                if check_packing(inner_list, outer_center, outer_side, outer_angle):
                    if outer_side < best_side:
                        best_side = outer_side
                        best_result = (inner_list, outer_center, outer_side, outer_angle)
        except Exception:
            pass
    
    if best_result is not None:
        return best_result
    
    # Fallback: use initial configuration with computed outer side
    inner_list = [[init_positions[i][0], init_positions[i][1], init_angles[i]] for i in range(N)]
    
    # Find minimal outer side for this configuration
    outer_side = compute_max_containing_side(init_positions, init_angles, (0.0, 0.0), 0.0, 2.0, 100.0)
    
    return (inner_list, (0.0, 0.0), outer_side, 0.0)

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin
    
    try:
        inner, outer_center, outer_side, outer_angle = optimize_packing(N, deadline)
        
        # Final validation
        if not check_packing(inner, outer_center, outer_side, outer_angle):
            # Fallback to a simple valid configuration
            # Place hexagons in a row
            inner_fallback = [[2.0 * np.sqrt(3) * i, 0.0, 0.0] for i in range(N)]
            # Compute bounding
            all_pts = []
            for x, y, a in inner_fallback:
                v = hexagon_vertices(x, y, 1.0, a)
                all_pts.extend(v.tolist())
            all_pts = np.array(all_pts)
            max_x = np.max(all_pts[:, 0])
            max_y = np.max(all_pts[:, 1])
            # Outer hexagon needs to contain all these points
            # Conservative estimate
            outer_side_fallback = max(max_x, max_y) + 3.0
            inner = inner_fallback
            outer_center = (0.0, 0.0)
            outer_side = outer_side_fallback
            outer_angle = 0.0
            
        return inner, outer_center, outer_side, outer_angle
    except Exception:
        # Very simple fallback
        inner = [[2.0 * np.sqrt(3) * i, 0.0, 0.0] for i in range(N)]
        return inner, (0.0, 0.0), 50.0, 0.0

# EVOLVE-BLOCK-END
