import numpy as np
import time

N = 12
SIDE = 1.0  # unit hexagons

def hex_verts(cx, cy, angle_deg):
    """Return 6 vertices of a unit regular hexagon centered at (cx,cy) with given rotation."""
    angle = np.radians(angle_deg)
    c, s = np.cos(angle), np.sin(angle)
    # vertices of unit regular hexagon (flat-top, side length 1)
    base = np.array([
        [1.0, 0.0],
        [0.5, np.sqrt(3)/2],
        [-0.5, np.sqrt(3)/2],
        [-1.0, 0.0],
        [-0.5, -np.sqrt(3)/2],
        [0.5, -np.sqrt(3)/2]
    ])
    R = np.array([[c, -s], [s, c]])
    return base @ R.T + np.array([cx, cy])

def hex_normals(cx, cy, angle_deg):
    """Return 6 outward normals of a unit regular hexagon."""
    angle = np.radians(angle_deg)
    # outward normals for flat-top hexagon
    base_normals = np.array([
        [1.0, 0.0],
        [0.5, -np.sqrt(3)/2],
        [-0.5, -np.sqrt(3)/2],
        [-1.0, 0.0],
        [-0.5, np.sqrt(3)/2],
        [0.5, np.sqrt(3)/2]
    ])
    R = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    return base_normals @ R.T

def support_function(n, cx, cy, angle_deg):
    """Support function in direction n."""
    normals = hex_normals(0, 0, angle_deg)
    verts = hex_verts(0, 0, angle_deg)
    # support = max over vertices of dot(vertex, n)
    return np.max(np.dot(verts, n)) + np.dot(np.array([cx, cy]), n)

def min_distance_between_hexes(p1, p2):
    """Compute minimum distance between two unit hexagons.
    p1, p2: (cx, cy, angle_deg)
    Returns negative value if overlapping (violation), positive if separated.
    Uses support function approach: min over all normal directions.
    """
    c1, s1, a1 = p1
    c2, s2, a2 = p2
    
    # Get all normals from both hexagons
    n1 = hex_normals(0, 0, a1)
    n2 = hex_normals(0, 0, a2)
    
    all_normals = np.vstack([n1, n2, -n1, -n2])
    
    min_dist = float('inf')
    for n in all_normals:
        n_norm = np.linalg.norm(n)
        if n_norm < 1e-12:
            continue
        n = n / n_norm
        s1_val = support_function(n, c1, s1, a1)
        s2_val = support_function(n, c2, s2, a2)
        # Distance in this direction
        d = s2_val - s1_val  # can be negative
        min_dist = min(min_dist, -d)  # we want separation, so -d gives gap
    
    # Also check reverse
    min_dist = min(min_dist, min_dist)  # symmetric
    
    return min_dist

def max_distance_to_origin(p):
    """Maximum distance from origin to any vertex of the hexagon."""
    c, s, a = p
    verts = hex_verts(c, s, a)
    return np.max(np.linalg.norm(verts, axis=1))

def objective(params):
    """Objective: outer hexagon side length needed.
    params: array of length 3*N: [x0, y0, angle0, x1, y1, angle1, ...]
    Returns the minimum outer side length (with outer hex centered at origin, rotated 0).
    """
    inner = []
    for i in range(N):
        x = params[3*i]
        y = params[3*i + 1]
        angle = params[3*i + 2]
        inner.append((x, y, angle))
    
    # Check pairwise distances
    min_pair_dist = float('inf')
    for i in range(N):
        for j in range(i+1, N):
            d = min_distance_between_hexes(inner[i], inner[j])
            min_pair_dist = min(min_pair_dist, d)
    
    # Check containment: all vertices must be inside outer hex
    max_r = 0.0
    for i in range(N):
        r = max_distance_to_origin(inner[i])
        max_r = max(max_r, r)
    
    # For a regular hexagon with side length S, the apothem is S*sqrt(3)/2
    # The maximum distance from center to a vertex is S
    # So we need S >= max_r / cos(angle_from_vertex_to_nearest_edge)
    # Actually for a regular hexagon, the circumradius = S
    # So outer side length must be at least max_r
    # But also need to account for rotation of outer hex
    # With outer hex rotated 0 (flat-top), the constraint is more complex
    
    # Simplified: use the support function approach for containment
    # The outer hexagon (centered at origin, side length S, rotated 0) has 6 faces
    # Each face has a normal direction
    # For containment, for each outer face normal n, the support of inner hex in -n direction <= S*apothem
    
    # Actually let's just use: the outer hex must contain all inner hex vertices
    # The outer hex (side S, rotated 0, centered at origin) is the intersection of 6 half-planes
    # Each half-plane: dot(p, n_i) <= S * sqrt(3)/2 for i=0..5 where n_i are outward normals
    
    outer_normals = hex_normals(0, 0, 0.0)
    S_needed = 0.0
    for i in range(N):
        verts = hex_verts(inner[i][0], inner[i][1], inner[i][2])
        for n in outer_normals:
            n_norm = np.linalg.norm(n)
            if n_norm < 1e-12:
                continue
            n = n / n_norm
            max_dot = np.max(np.dot(verts, n))
            # Need max_dot <= S * sqrt(3)/2
            S_candidate = max_dot / (np.sqrt(3) / 2)
            S_needed = max(S_needed, S_candidate)
    
    return S_needed

def honeycomb_init():
    """Generate a honeycomb-like initial configuration."""
    # Place hexagons in a roughly hexagonal arrangement
    # For N=12, use a pattern like:
    # Row 0: 4 hexagons
    # Row 1: 4 hexagons
    # Row 2: 4 hexagons
    # Or a more compact arrangement
    
    # Let's try a 3x4 grid with hexagonal offset
    positions = []
    dx = 2.0 * SIDE  # horizontal spacing for flat-top hexes
    dy = np.sqrt(3) * SIDE  # vertical spacing
    
    # 3 rows of 4
    for row in range(3):
        for col in range(4):
            x = col * dx + (row % 2) * dx / 2
            y = row * dy
            positions.append((x, y, 0.0))
    
    # Center the configuration
    xs = [p[0] for p in positions]
    ys = [p[1] for p in positions]
    cx = (max(xs) + min(xs)) / 2
    cy = (max(ys) + min(ys)) / 2
    
    centered = [(x - cx, y - cy, a) for x, y, a in positions]
    
    # Flatten to array
    params = []
    for p in centered:
        params.extend([p[0], p[1], p[2]])
    return np.array(params)

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    # Initial configuration
    params = honeycomb_init()
    
    best_params = params.copy()
    best_score = objective(best_params)
    
    # Simulated annealing
    np.random.seed(42)
    temperature = 0.5
    cooling = 0.995
    max_iter = 50000
    
    for iteration in range(max_iter):
        if time.time() > deadline - 10:
            break
        
        # Generate a random perturbation
        new_params = params + np.random.normal(0, temperature * np.array([1.0, 1.0, 0.1] * N))
        
        new_score = objective(new_params)
        old_score = objective(params)
        
        if new_score < old_score or np.random.random() < np.exp((old_score - new_score) / temperature):
            params = new_params
            if new_score < best_score:
                best_score = new_score
                best_params = new_params.copy()
        
        temperature *= cooling
        if temperature < 0.001:
            temperature = 0.001
    
    # Final polish with Nelder-Mead
    if time.time() < deadline - 20:
        from scipy.optimize import minimize
        try:
            result = minimize(objective, best_params, method='Nelder-Mead', 
                             options={'maxiter': 1000, 'xatol': 1e-8, 'fatol': 1e-10})
            if result.fun < best_score:
                best_params = result.x
                best_score = result.fun
        except:
            pass
    
    # Convert back to format
    inner = []
    for i in range(N):
        x = best_params[3*i]
        y = best_params[3*i + 1]
        angle = best_params[3*i + 2]
        inner.append([x, y, angle])
    
    # The outer hexagon is centered at origin with side length best_score, rotated 0
    return inner, (0.0, 0.0), best_score, 0.0
