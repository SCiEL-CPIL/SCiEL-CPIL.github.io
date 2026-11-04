import numpy as np
from scipy.optimize import minimize

N = 12
sqrt3 = np.sqrt(3)

def hex_vertices(cx, cy, angle_deg):
    """Return 6 vertices of a unit regular hexagon centered at (cx,cy) rotated by angle_deg."""
    a = np.radians(angle_deg)
    cos_a, sin_a = np.cos(a), np.sin(a)
    # Unit hexagon vertices at angles 0, 60, 120, 180, 240, 300
    base_angles = np.array([0, 60, 120, 180, 240, 300]) * np.pi / 180
    vx = np.cos(base_angles)
    vy = np.sin(base_angles)
    # Rotate
    rx = vx * cos_a - vy * sin_a
    ry = vx * sin_a + vy * cos_a
    return np.column_stack([cx + rx, cy + ry])

def required_side_length(positions, angles, center, container_angle_deg):
    """
    Compute the minimum side length s of a regular hexagon centered at `center`
    with rotation `container_angle_deg` that contains all hexagons at `positions`
    with their respective rotations `angles`.
    
    A regular hexagon with side s, centered at origin, rotated by angle phi,
    has support function h(n) = s * |n · u| where u are the 6 vertex directions... 
    Actually, for a regular hexagon with side s and flat-to-flat orientation (vertices at 0,60,...),
    the support function in direction n is s * max over 6 vertices of (n · v_i / |v_i|)... 
    
    More precisely: a regular hexagon with side s centered at origin has vertices at distance s from center
    at angles 0, 60, 120, 180, 240, 300 (for flat-to-flat, i.e., one vertex pointing right).
    Wait, let me be careful. A regular hexagon with side length s:
    - The distance from center to each vertex is s.
    - The distance from center to each edge midpoint is s * sqrt(3)/2.
    
    The support function h(n) of the outer hexagon in direction n is:
    h(n) = s * max_i (n · v_i) where v_i are unit vectors to vertices.
    But actually, for a convex polygon, h(n) = max_i (n · p_i) where p_i are vertices.
    For a regular hexagon with vertices at s*(cos(θ_i), sin(θ_i)) for θ_i = 0, 60, ..., 300:
    h(n) = s * max_i (n · (cos θ_i, sin θ_i))
    
    So for a point p to be inside the outer hexagon (centered at c, rotated by phi), we need:
    For all 6 directions, the projection of (p - c) onto each normal direction must be <= h(normal).
    
    Actually, it's easier to check: a point p is inside the outer hexagon iff for each of the 6 edge
    normals, the dot product with (p - c) is <= s * sqrt(3)/2 (the apothem).
    
    The 6 edge normals for a flat-to-flat hexagon (vertices at 0,60,...) point at angles 30, 90, 150, 210, 270, 330.
    The apothem is s * sqrt(3)/2.
    
    So: for each normal direction n_k (unit vector), we need n_k · (p - c) <= s * sqrt(3)/2.
    This gives s >= (n_k · (p - c)) / (sqrt(3)/2) for each k where the dot product is positive.
    The required s is the max over all points p and all 6 normals of (n_k · (p - c)) / (sqrt(3)/2).
    """
    phi = np.radians(container_angle_deg)
    # Edge normals at angles 30, 90, 150, 210, 270, 330 degrees, rotated by phi
    base_normals = np.array([30, 90, 150, 210, 270, 330]) * np.pi / 180 + phi
    normals = np.column_stack([np.cos(base_normals), np.sin(base_normals)])
    
    max_s = 0.0
    for i in range(len(positions)):
        verts = hex_vertices(positions[i][0], positions[i][1], angles[i])
        for v in verts:
            rel = np.array([v[0] - center[0], v[1] - center[1]])
            projections = normals @ rel
            # For each normal, s must be >= projection / (sqrt(3)/2)
            s_needed = np.max(projections) / (sqrt3 / 2)
            if s_needed > max_s:
                max_s = s_needed
    return max_s

def pairwise_distances(positions):
    """Compute pairwise Euclidean distances."""
    n = len(positions)
    d = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            d[i, j] = np.sqrt((positions[i][0] - positions[j][0])**2 + (positions[i][1] - positions[j][1])**2)
    return d

def min_pairwise_distance(positions):
    """Minimum distance between any two hexagon centers."""
    n = len(positions)
    min_d = float('inf')
    for i in range(n):
        for j in range(i+1, n):
            d = np.sqrt((positions[i][0] - positions[j][0])**2 + (positions[i][1] - positions[j][1])**2)
            if d < min_d:
                min_d = d
    return min_d

def objective(params):
    """
    Objective: minimize the required side length of the container.
    params: [x0, y0, a0, x1, y1, a1, ..., x11, y11, a11, cx, cy, cphi]
    """
    n_params_per_hex = 3
    pos = []
    angles = []
    for i in range(N):
        idx = i * n_params_per_hex
        x, y, a = params[idx], params[idx+1], params[idx+2]
        pos.append((x, y))
        angles.append(a)
    
    cx = params[N * n_params_per_hex]
    cy = params[N * n_params_per_hex + 1]
    cphi = params[N * n_params_per_hex + 2]
    
    center = (cx, cy)
    
    # Penalty for overlapping hexagons: minimum center distance should be >= 2 (for non-overlapping unit hexagons)
    # Actually, for two unit hexagons to be disjoint, the minimum distance between centers depends on orientation.
    # A safe lower bound: if centers are closer than 2, they might overlap. Let's use a soft penalty.
    min_d = min_pairwise_distance(pos)
    overlap_penalty = 0.0
    if min_d < 2.0:
        overlap_penalty = (2.0 - min_d) ** 2 * 100
    
    s = required_side_length(pos, angles, center, cphi)
    
    return s + overlap_penalty

def construct():
    # Initial positions: 3x4 grid
    np.random.seed(42)
    initial_params = []
    for row in range(3):
        y = (row - 1) * sqrt3
        for col in range(4):
            x = (col - 1.5) * 2.0
            initial_params.extend([x, y, 0.0])
    initial_params.extend([0.0, 0.0, 0.0])
    
    # Try multiple random initializations and Nelder-Mead
    best_params = initial_params
    best_val = objective(initial_params)
    
    # Also try a few random perturbations
    for trial in range(5):
        params = list(initial_params)
        for i in range(len(params)):
            if i < N * 3:
                params[i] += np.random.uniform(-0.5, 0.5)
        val = objective(params)
        if val < best_val:
            best_val = val
            best_params = params
    
    # Nelder-Mead optimization
    result = minimize(objective, best_params, method='Nelder-Mead', 
                      options={'maxiter': 2000, 'xatol': 1e-6, 'fatol': 1e-6, 'adaptive': True})
    
    if result.fun < best_val:
        best_params = result.x
        best_val = result.fun
    
    # Extract final positions
    pos = []
    angles = []
    for i in range(N):
        idx = i * 3
        x, y, a = best_params[idx], best_params[idx+1], best_params[idx+2]
        pos.append((x, y))
        angles.append(a)
    
    cx = best_params[N * 3]
    cy = best_params[N * 3 + 1]
    cphi = best_params[N * 3 + 2]
    
    center = (cx, cy)
    
    # Compute the actual required side length without penalty
    s = required_side_length(pos, angles, center, cphi)
    
    # Add tiny margin for numerical safety
    side = s + 1e-7
    
    # Build inner list
    inner = [(pos[i][0], pos[i][1], angles[i]) for i in range(N)]
    
    return inner, center, side, cphi

# EVOLVE-BLOCK-END
