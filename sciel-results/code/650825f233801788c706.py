import numpy as np
from scipy.optimize import minimize

N = 12

def hexagon_vertices(cx, cy, side, angle_deg):
    """Return vertices of a regular hexagon centered at (cx, cy) with given side length and rotation."""
    angle = np.radians(angle_deg)
    # Regular hexagon vertices at angles 0, 60, 120, 180, 240, 300 degrees
    base_angles = np.array([0, 60, 120, 180, 240, 300]) * np.pi / 180.0
    angles = base_angles + angle
    # Side length s: distance from center to vertex is s (for regular hexagon, circumradius = side length)
    r = side
    xs = cx + r * np.cos(angles)
    ys = cy + r * np.sin(angles)
    return np.column_stack([xs, ys])

def hexagon_support(vertices, direction):
    """Compute support function value of polygon in given direction."""
    return np.max(vertices @ direction)

def min_enclosing_hexagon_side(inner_hexes, outer_center, outer_angle_deg):
    """
    Given inner hexagons (each as (cx, cy, angle_deg) with side=1), 
    find the minimum side length of a regular hexagon centered at outer_center 
    with rotation outer_angle_deg that contains all inner hexagons.
    
    Actually, we need to find the minimal side length such that all inner hexagons
    fit inside. For a given outer hexagon orientation and center, the required side
    length is determined by the maximum extent in each of the 6 normal directions.
    """
    outer_angle = np.radians(outer_angle_deg)
    # The 6 outward normal directions of the outer hexagon
    # For a regular hexagon with a vertex at angle 0, the normals to the sides are at 30, 90, 150, 210, 270, 330
    # But we need to be careful: the support function in the normal direction gives the distance from center to the side
    # For a regular hexagon of side s, the inradius (distance from center to side) is s * sqrt(3)/2
    # So the side length s = 2 * h / sqrt(3) where h is the max support in normal direction
    
    # Normal directions: perpendicular to each side
    # If hexagon has vertices at angles theta_i, the sides are between consecutive vertices
    # The normal to the side between vertex at angle a and vertex at angle b is at (a+b)/2 + 90 degrees? 
    # Let me think differently.
    
    # For a regular hexagon centered at origin with one vertex at angle 0:
    # Vertices at 0, 60, 120, 180, 240, 300 degrees
    # The sides connect consecutive vertices. The outward normal to the side between vertex at angle 0 and 60 
    # points in direction 30 degrees.
    # So normals are at 30, 90, 150, 210, 270, 330 degrees (i.e., 30 + k*60 for k=0..5)
    
    normal_angles = (30 + np.arange(6) * 60) * np.pi / 180.0
    normal_angles += outer_angle
    
    normals = np.column_stack([np.cos(normal_angles), np.sin(normal_angles)])
    
    # For each inner hexagon, compute its support in each normal direction
    # The union's support in direction d is max over all inner hexagons of their support in d
    # But we need to account for the outer center offset
    
    max_supports = []
    for (cx, cy, angle_deg) in inner_hexes:
        verts = hexagon_vertices(cx, cy, 1.0, angle_deg)
        # Translate to outer center coordinate system
        verts_rel = verts - np.array(outer_center)
        for normal in normals:
            s = np.max(verts_rel @ normal)
            max_supports.append(s)
    
    max_supports = np.array(max_supports)
    # The required inradius is the max over all 6 directions
    # But we need the max for each direction separately, then the overall max
    # Reshape: 6 inner hexagons... wait, we have N inner hexagons, 6 normals each
    max_supports = max_supports.reshape(-1, 6)
    # For each normal direction, take the max over all inner hexagons
    max_per_direction = np.max(max_supports, axis=0)
    # The required inradius is the max over all directions
    h = np.max(max_per_direction)
    # Side length = 2 * h / sqrt(3)
    side = 2.0 * h / np.sqrt(3.0)
    return side

def objective(params):
    """
    params: flattened array of [x1, y1, a1, x2, y2, a2, ..., x12, y12, a12, cx, cy, outer_angle]
    Returns the outer hexagon side length.
    """
    inner_hexes = []
    for i in range(N):
        x = params[3*i]
        y = params[3*i+1]
        a = params[3*i+2]
        inner_hexes.append((x, y, a))
    cx = params[3*N]
    cy = params[3*N+1]
    outer_angle = params[3*N+2]
    
    return min_enclosing_hexagon_side(inner_hexes, (cx, cy), outer_angle)

def construct():
    # Initial guess: hexagonal lattice arrangement
    # For 12 hexagons, use a compact arrangement
    # Hexagonal lattice spacing: for unit hexagons, center-to-center distance is sqrt(3) * side = sqrt(3)
    # Actually, for regular hexagons of side 1, the distance between centers of adjacent hexagons in a honeycomb is 2 * (sqrt(3)/2) = sqrt(3)
    
    # Let's use a hexagonal cluster: 
    # Ring 0: 1 hexagon at center
    # Ring 1: 6 hexagons around it
    # Ring 2: need 5 more to make 12
    # Total: 1 + 6 + 5 = 12
    
    d = np.sqrt(3.0)  # center-to-center distance for adjacent hexagons
    
    inner_hexes = []
    # Center
    inner_hexes.append([0.0, 0.0, 0.0])
    
    # Ring 1: 6 hexagons at distance d from center, at angles 0, 60, 120, 180, 240, 300
    for k in range(6):
        angle = k * 60.0
        x = d * np.cos(np.radians(angle))
        y = d * np.sin(np.radians(angle))
        inner_hexes.append([x, y, 0.0])
    
    # Ring 2: 5 hexagons. Place them at distance 2*d from center in directions 30, 90, 150, 210, 270
    # But we only need 5, so pick 5 of the 6 possible positions
    for k in range(5):
        angle = 30.0 + k * 60.0
        x = 2.0 * d * np.cos(np.radians(angle))
        y = 2.0 * d * np.sin(np.radians(angle))
        inner_hexes.append([x, y, 0.0])
    
    # Initial outer hexagon: centered at origin, angle 0
    cx, cy = 0.0, 0.0
    outer_angle = 0.0
    
    # Compute initial side length
    inner_list = [(x, y, a) for (x, y, a) in inner_hexes]
    init_side = min_enclosing_hexagon_side(inner_list, (cx, cy), outer_angle)
    
    # Prepare parameters for optimization
    params = np.zeros(3*N + 3)
    for i in range(N):
        params[3*i] = inner_hexes[i][0]
        params[3*i+1] = inner_hexes[i][1]
        params[3*i+2] = inner_hexes[i][2]
    params[3*N] = cx
    params[3*N+1] = cy
    params[3*N+2] = outer_angle
    
    # Use simulated annealing-like approach with scipy minimize
    # Try multiple starting points and local optimizations
    
    best_params = params.copy()
    best_side = init_side
    
    # Try different initial arrangements
    arrangements = []
    
    # Arrangement 1: Current hexagonal cluster (above)
    arrangements.append(params.copy())
    
    # Arrangement 2: Two rows of 6
    params2 = np.zeros(3*N + 3)
    d = np.sqrt(3.0)
    for i in range(6):
        # Row 1
        x = (i - 2.5) * d
        y = 0.0
        params2[3*i] = x
        params2[3*i+1] = y
        params2[3*i+2] = 0.0
    for i in range(6, 12):
        # Row 2
        x = (i - 6 - 2.5) * d
        y = d * 0.866  # offset by half the height
        params2[3*i] = x
        params2[3*i+1] = y
        params2[3*i+2] = 0.0
    arrangements.append(params2)
    
    # Arrangement 3: Three rows of 4
    params3 = np.zeros(3*N + 3)
    for row in range(3):
        for col in range(4):
            idx = row * 4 + col
            x = (col - 1.5) * d
            y = row * d * 0.866
            params3[3*idx] = x
            params3[3*idx+1] = y
            params3[3*idx+2] = 0.0
    arrangements.append(params3)
    
    # For each arrangement, run local optimization
    for arr in arrangements:
        # Try local optimization
        try:
            result = minimize(objective, arr, method='Nelder-Mead', 
                            options={'maxiter': 500, 'xatol': 1e-6, 'fatol': 1e-8})
            if result.fun < best_side:
                best_side = result.fun
                best_params = result.x
        except:
            pass
        
        # Also try with different outer angles
        for test_angle in [0, 30, 60, 90]:
            arr2 = arr.copy()
            arr2[3*N+2] = np.radians(test_angle)
            try:
                result = minimize(objective, arr2, method='Nelder-Mead', 
                                options={'maxiter': 300, 'xatol': 1e-6, 'fatol': 1e-8})
                if result.fun < best_side:
                    best_side = result.fun
                    best_params = result.x
            except:
                pass
    
    # Extract final parameters
    inner = []
    for i in range(N):
        x = best_params[3*i]
        y = best_params[3*i+1]
        a = np.degrees(best_params[3*i+2]) % 360.0
        inner.append([x, y, a])
    
    cx = best_params[3*N]
    cy = best_params[3*N+1]
    outer_angle = np.degrees(best_params[3*N+2]) % 360.0
    
    return inner, (cx, cy), best_side, outer_angle

# EVOLVE-BLOCK-START
# The construct function is defined above
# EVOLVE-BLOCK-END
