import math
import time
import numpy as np
from scipy.optimize import minimize

N = 12
sqrt3 = math.sqrt(3.0)

# Precompute normal angles for the outer hexagon (flat-top)
# The outer hexagon is fixed at angle 0.
# Normals are at 30, 90, 150, 210, 270, 330 degrees.
normal_angles = [math.radians(30 + 60 * j) for j in range(6)]
normal_cos = np.array([math.cos(a) for a in normal_angles])
normal_sin = np.array([math.sin(a) for a in normal_angles])

def get_hex_vertices(cx, cy, theta):
    """Returns 6 vertices of a unit hexagon at (cx, cy) with rotation theta."""
    # Vertices of a unit hexagon centered at origin, flat-top (angle 0)
    # Angles: 0, 60, 120, 180, 240, 300
    angles = np.array([0, 60, 120, 180, 240, 300]) * math.pi / 180.0 + theta
    vx = cx + np.cos(angles)
    vy = cy + np.sin(angles)
    return np.column_stack((vx, vy))

def calc_L_bound(positions, angles):
    """
    Calculate the minimum side length L of the outer hexagon 
    (centered at 0,0, angle 0) to contain all inner hexagons.
    """
    max_proj = 0.0
    for i in range(N):
        cx, cy = positions[i]
        theta = angles[i]
        verts = get_hex_vertices(cx, cy, theta)
        # Project onto 6 normals
        projs = verts[:, 0] * normal_cos + verts[:, 1] * normal_sin
        # Max projection for this hexagon
        current_max = np.max(projs)
        if current_max > max_proj:
            max_proj = current_max
    # L is related to max_proj.
    # For a regular hexagon of side L, the distance from center to side (apothem) is L * sqrt(3) / 2.
    # The projection of a vertex onto the normal must be <= apothem.
    # So max_proj <= L * sqrt(3) / 2
    # L >= max_proj * 2 / sqrt(3)
    return max_proj * 2.0 / sqrt3

def exact_min_distance(v1, v2):
    """
    Calculate the exact minimum distance between two convex polygons (hexagons).
    Returns positive distance if disjoint, negative if overlapping (penetration depth).
    v1, v2: (6, 2) arrays of vertices.
    """
    # Get all edge normals from both polygons
    def get_edge_normals(verts):
        normals = []
        for i in range(6):
            p1 = verts[i]
            p2 = verts[(i + 1) % 6]
            edge = p2 - p1
            # Normal perpendicular to edge (for CCW vertices, inward normal)
            nx = edge[1]
            ny = -edge[0]
            norm = math.sqrt(nx*nx + ny*ny)
            if norm > 1e-9:
                normals.append(np.array([nx/norm, ny/norm]))
        return normals

    normals1 = get_edge_normals(v1)
    normals2 = get_edge_normals(v2)
    
    min_gap = float('inf')
    all_overlap = True
    
    for n in normals1 + normals2:
        # Project both polygons onto normal n
        proj1 = np.dot(v1, n)
        proj2 = np.dot(v2, n)
        
        min1, max1 = np.min(proj1), np.max(proj1)
        min2, max2 = np.min(proj2), np.max(proj2)
        
        # Check for gap or overlap
        if max1 < min2 or max2 < min1:
            # Disjoint on this axis
            gap = min(abs(min2 - max1), abs(min1 - max2))
            if gap < min_gap:
                min_gap = gap
            all_overlap = False
        else:
            # Overlap on this axis
            overlap = min(max1, max2) - max(min1, min2)
            if overlap < min_gap:
                min_gap = overlap

    if all_overlap:
        return -min_gap  # Penetration depth (negative)
    else:
        return min_gap  # Gap distance (positive)

def check_no_overlap(positions, angles, tol=1e-9):
    """Check if all hexagons are non-overlapping."""
    verts_list = [get_hex_vertices(positions[i][0], positions[i][1], angles[i]) for i in range(N)]
    for i in range(N):
        for j in range(i + 1, N):
            d = exact_min_distance(verts_list[i], verts_list[j])
            if d < -tol:  # Overlapping
                return False
    return True

def objective(params):
    """
    Objective function for optimization.
    params: array of shape (3*N,) -> [x0, y0, theta0, x1, y1, theta1, ...]
    Returns: Outer side length L + penalty for overlaps.
    """
    positions = params[0:2*N].reshape(N, 2)
    angles = params[2*N:3*N]
    
    L = calc_L_bound(positions, angles)
    
    # Calculate overlap penalty
    overlap_penalty = 0.0
    verts_list = [get_hex_vertices(positions[i][0], positions[i][1], angles[i]) for i in range(N)]
    for i in range(N):
        for j in range(i + 1, N):
            d = exact_min_distance(verts_list[i], verts_list[j])
            if d < 0:
                overlap_penalty += d * d  # Quadratic penalty
    
    return L + 1000.0 * overlap_penalty

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin

    # Generate a few diverse initial configurations
    initial_configs = []
    
    # Config 1: Hexagonal cluster (centered)
    centers = []
    for q in range(-2, 3):
        for r in range(-2, 3):
            if abs(q + r) <= 2:
                x = 1.5 * q
                y = sqrt3 * (r + q / 2.0)
                centers.append((x, y))
    
    # Select 12 closest to center for initial guess
    centers.sort(key=lambda p: p[0]**2 + p[1]**2)
    init_pos = np.array(centers[:12])
    init_ang = np.zeros(N)
    initial_configs.append(np.concatenate([init_pos.flatten(), init_ang]))
    
    # Config 2: Grid
    grid_pos = []
    for i in range(12):
        q = i % 4 - 1.5
        r = i // 4 - 1.5
        x = 1.5 * q
        y = sqrt3 * (r + q / 2.0)
        grid_pos.append([x, y])
    init_pos = np.array(grid_pos)
    init_ang = np.zeros(N)
    initial_configs.append(np.concatenate([init_pos.flatten(), init_ang]))
    
    # Config 3: Random perturbation of Config 1
    np.random.seed(42)
    rand_pos = init_pos + np.random.normal(0, 0.5, size=(N, 2))
    rand_ang = np.random.normal(0, 10, size=N) * math.pi / 180.0
    initial_configs.append(np.concatenate([rand_pos.flatten(), rand_ang]))
    
    best_params = None
    best_score = float('inf')
    
    # Run local optimization from each initial config
    for init in initial_configs:
        if time.time() > deadline - 10:
            break
            
        try:
            res = minimize(objective, init, method='Nelder-Mead', options={'maxiter': 500, 'xatol': 1e-4, 'fatol': 1e-4})
            score = res.fun
            if score < best_score:
                best_score = score
                best_params = res.x
        except Exception:
            continue
            
    if best_params is None:
        # Fallback if optimization failed
        best_params = initial_configs[0]
        best_score = objective(best_params)
        
    pos = best_params[0:2*N].reshape(N, 2)
    angs = best_params[2*N:3*N]
    
    # Final validation and safety margin
    if not check_no_overlap(pos, angs, tol=1e-8):
        # If still overlapping, use a known valid configuration
        # Simple grid with larger spacing
        pos = []
        for i in range(12):
            q = (i % 4) - 1.5
            r = (i // 4) - 1.5
            x = 1.5 * q * 1.05
            y = sqrt3 * (r + q / 2.0) * 1.05
            pos.append([x, y])
        pos = np.array(pos)
        angs = np.zeros(N)
        L_val = calc_L_bound(pos, angs)
    else:
        L_val = calc_L_bound(pos, angs)
    
    inner = []
    for i in range(N):
        inner.append([pos[i][0], pos[i][1], math.degrees(angs[i])])
    
    return inner, (0.0, 0.0), L_val, 0.0

# EVOLVE-BLOCK-END
