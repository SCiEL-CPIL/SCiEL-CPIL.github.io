"""Hexagon packing: 11 unit hexagons inside a regular hexagon."""
import math
import time
import numpy as np
from scipy.optimize import minimize, differential_evolution

N = 11

# EVOLVE-BLOCK-START

# Precompute base hexagon vertices (unit side length, centered at origin, angle 0)
# Vertices are at angles 0, 60, 120, 180, 240, 300 degrees
_BASE_VERTS = np.array([
    [math.cos(i * math.pi / 3.0), math.sin(i * math.pi / 3.0)] 
    for i in range(6)
])

def _get_hex_vertices(angle_rad, center):
    """
    Compute the 6 vertices of a unit regular hexagon 
    rotated by angle_rad and translated by center.
    Returns (6, 2) array.
    """
    c = math.cos(angle_rad)
    s = math.sin(angle_rad)
    R = np.array([[c, -s], [s, c]])
    return _BASE_VERTS @ R.T + center

def _poly_dist(v1, v2):
    """
    Compute the minimum distance between two convex polygons (hexagons) using SAT.
    v1, v2: (6, 2) arrays of vertices in counter-clockwise order.
    Returns 0.0 if overlapping, otherwise the minimum gap.
    """
    def get_outward_normals(verts):
        normals = np.empty((6, 2))
        for i in range(6):
            edge = verts[(i + 1) % 6] - verts[i]
            norm = np.array([-edge[1], edge[0]])
            norm_len = np.linalg.norm(norm)
            if norm_len > 1e-12:
                normals[i] = norm / norm_len
            else:
                normals[i] = np.array([1.0, 0.0])
        return normals

    n1 = get_outward_normals(v1)
    n2 = get_outward_normals(v2)
    
    min_dist = float('inf')
    
    for k in range(6):
        n = n1[k]
        proj1 = v1 @ n
        proj2 = v2 @ n
        min1, max1 = np.min(proj1), np.max(proj1)
        min2, max2 = np.min(proj2), np.max(proj2)
        
        if max1 < min2:
            gap = min2 - max1
            if gap < min_dist:
                min_dist = gap
        elif max2 < min1:
            gap = min1 - max2
            if gap < min_dist:
                min_dist = gap

    for k in range(6):
        n = n2[k]
        proj1 = v1 @ n
        proj2 = v2 @ n
        min1, max1 = np.min(proj1), np.max(proj1)
        min2, max2 = np.min(proj2), np.max(proj2)
        
        if max1 < min2:
            gap = min2 - max1
            if gap < min_dist:
                min_dist = gap
        elif max2 < min1:
            gap = min1 - max2
            if gap < min_dist:
                min_dist = gap

    if min_dist == float('inf'):
        return 0.0
    return min_dist

def _compute_side_length(centers, angles, outer_angle_rad):
    """
    Compute the minimum side length S of the outer hexagon 
    (centered at origin, rotated by outer_angle_rad) 
    that contains all inner hexagons.
    """
    outer_normals = np.array([
        [math.cos(outer_angle_rad + k * math.pi / 3.0), 
         math.sin(outer_angle_rad + k * math.pi / 3.0)]
        for k in range(6)
    ])
    
    max_extent = 0.0
    for i in range(N):
        verts = _get_hex_vertices(angles[i], centers[i])
        projs = verts @ outer_normals.T
        max_abs = np.max(np.abs(projs))
        if max_abs > max_extent:
            max_extent = max_abs
            
    S = max_extent * 2.0 / math.sqrt(3.0)
    return S

def _is_valid(centers, angles, S, outer_angle_rad, tol=1e-4):
    """
    Check if the configuration is valid:
    1. All inner hexagons are inside the outer hexagon of side S.
    2. No inner hexagons overlap.
    """
    outer_normals = np.array([
        [math.cos(outer_angle_rad + k * math.pi / 3.0), 
         math.sin(outer_angle_rad + k * math.pi / 3.0)]
        for k in range(6)
    ])
    limit = S * math.sqrt(3) / 2.0
    
    for i in range(N):
        verts = _get_hex_vertices(angles[i], centers[i])
        projs = verts @ outer_normals.T
        if np.max(np.abs(projs)) > limit + tol:
            return False
            
    verts_list = [_get_hex_vertices(angles[i], centers[i]) for i in range(N)]
    for i in range(N):
        for j in range(i + 1, N):
            d = _poly_dist(verts_list[i], verts_list[j])
            if d < tol:
                return False
    return True

def _seed_solution():
    """
    Generate a valid seed solution.
    Hexagonal lattice arrangement.
    """
    d = math.sqrt(3)  # Distance between centers of touching hexagons
    centers = []
    angles = []
    
    # Central
    centers.append([0, 0])
    angles.append(0.0)
    
    # Ring 1 (6 hexagons)
    for i in range(6):
        theta = i * math.pi / 3.0
        cx = d * math.cos(theta)
        cy = d * math.sin(theta)
        centers.append([cx, cy])
        angles.append(0.0)
        
    # Ring 2 (12 hexagons, pick 4)
    # Positions in axial coordinates (q, r)
    ring2_pos = []
    for q in range(-2, 3):
        for r in range(-2, 3):
            s = -q - r
            if max(abs(q), abs(r), abs(s)) == 2:
                # Convert axial to pixel
                x = d * (q + r / 2.0)
                y = d * (math.sqrt(3) / 2.0) * r
                ring2_pos.append((x, y))
                
    # Pick 4 well-separated ones
    selected = [ring2_pos[0], ring2_pos[3], ring2_pos[6], ring2_pos[9]]
    for pos in selected:
        centers.append(list(pos))
        angles.append(0.0)
        
    centers = np.array(centers)
    angles = np.array(angles)
    
    S = _compute_side_length(centers, angles, 0.0)
    S *= 1.05 # Add margin
    return centers, angles, S

def construct():
    start_time = time.time()
    deadline = 145.0 # Safety margin
    
    # Initialize with seed
    centers0, angles0, S0 = _seed_solution()
    best_S = S0
    best_centers = centers0.copy()
    best_angles = angles0.copy()
    best_outer_angle = 0.0
    
    # Verify seed
    if not _is_valid(best_centers, best_angles, best_S, best_outer_angle):
        # Fallback: spread out more
        d = math.sqrt(3) * 1.2
        centers_fb = []
        angles_fb = []
        centers_fb.append([0, 0])
        angles_fb.append(0.0)
        for i in range(6):
            theta = i * math.pi / 3.0
            centers_fb.append([d * math.cos(theta), d * math.sin(theta)])
            angles_fb.append(0.0)
        for i in range(4):
            theta = (2*i + 1) * math.pi / 12.0
            centers_fb.append([2*d * math.cos(theta), 2*d * math.sin(theta)])
            angles_fb.append(0.0)
        best_centers = np.array(centers_fb)
        best_angles = np.array(angles_fb)
        best_S = _compute_side_length(best_centers, best_angles, 0.0)
        best_S *= 1.1

    def objective(params):
        """
        params: 3N + 1 array
        [x1, y1, a1, x2, y2, a2, ..., xN, yN, aN, outer_angle]
        """
        if time.time() > deadline - 2.0:
            return 1e10
            
        # Unpack
        pos = params[:3*N].reshape(N, 3)
        centers = pos[:, :2]
        angles = pos[:, 2]
        outer_angle = params[3*N]
        
        # Compute S
        S = _compute_side_length(centers, angles, outer_angle)
        
        # Compute penalty
        penalty = 0.0
        # Pre-compute vertices
        verts = [_get_hex_vertices(angles[i], centers[i]) for i in range(N)]
        
        for i in range(N):
            for j in range(i + 1, N):
                d = _poly_dist(verts[i], verts[j])
                if d < 0.01:
                    penalty += (0.01 - d) ** 2
                    
        return S + 100.0 * penalty

    # Optimization 1: Differential Evolution
    # Bounds: x, y in [-5, 5], angle in [0, 2*pi]
    bounds = [(-5, 5)] * (2*N) + [(0, 2*math.pi)] * N + [(0, math.pi/3)]
    
    x0 = np.hstack([best_centers.flatten(), best_angles, [best_outer_angle]])
    
    try:
        if time.time() < deadline - 40.0:
            # DE is slow for 34 vars. Use smaller popsize.
            res = differential_evolution(
                objective, bounds, 
                seed=42, 
                maxiter=100, 
                popsize=15, 
                tol=1e-5, 
                polish=False
            )
            if res.fun < best_S:
                pos = res.x[:3*N].reshape(N, 3)
                centers = pos[:, :2]
                angles = pos[:, 2]
                outer_angle = res.x[3*N]
                S_val = _compute_side_length(centers, angles, outer_angle)
                if _is_valid(centers, angles, S_val * 1.001, outer_angle):
                    best_S = S_val
                    best_centers = centers
                    best_angles = angles
                    best_outer_angle = outer_angle
    except Exception:
        pass

    # Optimization 2: Nelder-Mead Local Refinement
    if time.time() < deadline - 10.0:
        x0 = np.hstack([best_centers.flatten(), best_angles, [best_outer_angle]])
        try:
            res = minimize(
                objective, x0, 
                method='Nelder-Mead',
                options={'xatol': 1e-6, 'fatol': 1e-6, 'maxiter': 5000, 'maxfev': 10000}
            )
            if res.fun < best_S:
                pos = res.x[:3*N].reshape(N, 3)
                centers = pos[:, :2]
                angles = pos[:, 2]
                outer_angle = res.x[3*N]
                S_val = _compute_side_length(centers, angles, outer_angle)
                if _is_valid(centers, angles, S_val * 1.001, outer_angle):
                    best_S = S_val
                    best_centers = centers
                    best_angles = angles
                    best_outer_angle = outer_angle
        except Exception:
            pass

    # Optimization 3: Second round of NM with different start if time permits
    if time.time() < deadline - 5.0:
        # Try a perturbed start
        x0_pert = x0.copy()
        x0_pert[:2*N] += np.random.normal(0, 0.1, 2*N)
        try:
            res = minimize(
                objective, x0_pert, 
                method='Nelder-Mead',
                options={'xatol': 1e-7, 'fatol': 1e-7, 'maxiter': 2000, 'maxfev': 5000}
            )
            if res.fun < best_S:
                pos = res.x[:3*N].reshape(N, 3)
                centers = pos[:, :2]
                angles = pos[:, 2]
                outer_angle = res.x[3*N]
                S_val = _compute_side_length(centers, angles, outer_angle)
                if _is_valid(centers, angles, S_val * 1.001, outer_angle):
                    best_S = S_val
                    best_centers = centers
                    best_angles = angles
                    best_outer_angle = outer_angle
        except Exception:
            pass

    # Format output
    inner = []
    for i in range(N):
        inner.append([best_centers[i][0], best_centers[i][1], math.degrees(best_angles[i])])
        
    return inner, (0.0, 0.0), best_S, math.degrees(best_outer_angle)

# EVOLVE-BLOCK-END
