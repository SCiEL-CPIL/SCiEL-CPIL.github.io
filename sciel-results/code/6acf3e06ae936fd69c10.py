# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time

def _hex_vertices(cx, cy, s, angle_deg):
    """Return 6 vertices of a regular hexagon centered at (cx,cy) with side s and angle."""
    a = np.radians(angle_deg)
    angles = a + np.radians(60) * np.arange(6)
    return np.column_stack([cx + s * np.cos(angles), cy + s * np.sin(angles)])

def _edge_normals(vertices):
    """Return 12 edge normals (6 edges x 2 directions) for a hexagon."""
    n = len(vertices)
    normals = []
    for i in range(n):
        v1 = vertices[i]
        v2 = vertices[(i + 1) % n]
        edge = v2 - v1
        # Perpendicular (normal) to edge
        normal = np.array([-edge[1], edge[0]])
        normal = normal / (np.linalg.norm(normal) + 1e-15)
        normals.append(normal)
        normals.append(-normal)
    return np.array(normals)

def _project_on_normal(vertices, normal):
    """Project all vertices onto a normal direction."""
    return vertices @ normal

def _min_gap_between_hexes(vertices1, vertices2):
    """
    Check separation: for each of the 12 normals of hex1, project both hexes.
    The gap is the minimum over all normals of the separation.
    Actually, we need: on at least one of 12 edge normals, the projected intervals are apart by > 1e-6.
    This means the hexagons are separated (not overlapping).
    We return the maximum gap over all 12 normals (positive means separated).
    """
    normals1 = _edge_normals(vertices1)
    max_gap = -np.inf
    for normal in normals1:
        proj1 = _project_on_normal(vertices1, normal)
        proj2 = _project_on_normal(vertices2, normal)
        # Interval 1: [min1, max1], Interval 2: [min2, max2]
        # Gap = max(min1 - max2, min2 - max1)
        gap = max(np.min(proj1) - np.max(proj2), np.min(proj2) - np.max(proj1))
        max_gap = max(max_gap, gap)
    return max_gap

def _cluster_config(cluster_params):
    """
    Generate cluster of 11 hexagons from parameters.
    cluster_params: (11, 3) array of (dx, dy, angle_offset) relative to cluster center.
    We'll use a base pattern and optimize.
    """
    n = 11
    # Base hexagonal pattern: center + 6 surrounding + 4 extra
    # For 11 hexagons, a common compact arrangement:
    # Row 0: 3 hexagons
    # Row 1: 3 hexagons (offset)
    # Row 2: 3 hexagons (offset)
    # Row 3: 2 hexagons
    # Or use the "hexagonal number" style: 1 in center, 6 around, then 4 more
    
    # Let's use a pattern that's known to be compact:
    # Positions for unit hexagons in a tight cluster
    # Using axial coordinates for hex grid
    base_positions = np.array([
        [0, 0],           # center
        [1, 0],           # right
        [0, 1],           # top-right
        [-1, 1],          # top-left
        [-1, 0],          # left
        [0, -1],          # bottom-left
        [1, -1],          # bottom-right
        [2, 0],           # far right
        [2, -1],          # far bottom-right
        [-2, 1],          # far top-left
        [-2, 0],          # far left
    ])
    
    # Convert axial to cartesian for hex grid (pointy-top)
    # x = size * (sqrt(3) * q + sqrt(3)/2 * r)
    # y = size * (3/2 * r)
    sqrt3 = np.sqrt(3)
    q = base_positions[:, 0]
    r = base_positions[:, 1]
    x = sqrt3 * (q + 0.5 * r)
    y = 1.5 * r
    
    # For unit hexagons, the distance between centers of adjacent hexes is 2*sqrt(3)/2 * something
    # Actually for unit side hexagons, the distance between centers of adjacent hexagons is 2 * apothem = 2 * sqrt(3)/2 * 1 = sqrt(3)
    # Wait, apothem of unit hexagon = sqrt(3)/2, so distance between adjacent centers = 2 * sqrt(3)/2 = sqrt(3)
    # But in our axial coords, we need to scale so that adjacent hexes are sqrt(3) apart
    
    # In axial coords with the above formula, adjacent hexes (e.g., (0,0) and (1,0)) are:
    # (0,0) -> (0, 0)
    # (1,0) -> (sqrt(3), 0)
    # Distance = sqrt(3), which is correct for unit hexagons!
    
    return np.column_stack([x, y])

def _compute_outer_side(cluster_centered, angle_deg):
    """
    Given cluster vertices (centered at origin) and outer hexagon angle,
    compute the minimum side length of a regular hexagon (centered at origin, with given angle)
    that contains all cluster vertices.
    
    For a regular hexagon with side s and angle a, centered at origin:
    A point p is inside if for all 6 edge normals n_i, p @ n_i <= s * apothem = s * sqrt(3)/2
    Actually, the support function: the hexagon contains p if max_i (p @ n_i) <= s * sqrt(3)/2
    where n_i are the 6 outward normals.
    
    So s >= max_i (p @ n_i) / (sqrt(3)/2) for all points p.
    Thus s = max over all points p and all 6 normals n_i of (p @ n_i) / (sqrt(3)/2)
    """
    # Outer hexagon vertices to get normals
    outer_verts = _hex_vertices(0, 0, 1.0, angle_deg)  # unit side, just to get normals
    normals = []
    for i in range(6):
        v1 = outer_verts[i]
        v2 = outer_verts[(i + 1) % 6]
        edge = v2 - v1
        normal = np.array([-edge[1], edge[0]])
        normal = normal / (np.linalg.norm(normal) + 1e-15)
        # Ensure normal points outward (away from center)
        mid = (v1 + v2) / 2
        if np.dot(normal, mid) < 0:
            normal = -normal
        normals.append(normal)
    normals = np.array(normals)
    
    # For each point in cluster, project onto each normal
    # The required s is max(p @ n) / (sqrt(3)/2)
    max_proj = 0.0
    for p in cluster_centered:
        projs = p @ normals.T
        max_proj = max(max_proj, np.max(projs))
    
    return max_proj / (np.sqrt(3) / 2)

def _get_all_cluster_vertices(cluster_positions, base_angle=0):
    """Get all vertices of all 11 hexagons in the cluster."""
    all_verts = []
    for i in range(len(cluster_positions)):
        cx, cy = cluster_positions[i]
        # Each inner hexagon has angle 0 (or we could optimize individual angles, but let's keep them aligned for simplicity)
        verts = _hex_vertices(cx, cy, 1.0, base_angle)
        all_verts.append(verts)
    return all_verts

def _check_validity(cluster_positions, outer_angle, outer_side):
    """Check if the configuration is valid."""
    # 1. All inner hexagons must be separated
    all_verts = _get_all_cluster_vertices(cluster_positions)
    for i in range(11):
        for j in range(i + 1, 11):
            gap = _min_gap_between_hexes(all_verts[i], all_verts[j])
            if gap <= 1e-6:
                return False
    
    # 2. All inner vertices must be inside the outer hexagon
    outer_verts = _hex_vertices(0, 0, outer_side, outer_angle)
    # Check each inner vertex against each outer edge
    for verts in all_verts:
        for p in verts:
            for i in range(6):
                p1 = outer_verts[i]
                p2 = outer_verts[(i + 1) % 6]
                edge = p2 - p1
                normal = np.array([-edge[1], edge[0]])
                normal = normal / (np.linalg.norm(normal) + 1e-15)
                # Ensure normal points inward (toward center)
                mid = (p1 + p2) / 2
                if np.dot(normal, -mid) < 0:
                    normal = -normal
                # Cross product check: cross(p2 - p1, p - p1) >= -1e-6
                cross = edge[0] * (p[1] - p1[1]) - edge[1] * (p[0] - p1[0])
                if cross < -1e-6:
                    return False
    return True

def _objective(params):
    """
    Objective function to minimize.
    params: [dx0, dy0, dx1, dy1, ..., dx10, dy10, global_angle]
    We optimize the cluster positions and global rotation.
    """
    n = 11
    positions = params[:2 * n].reshape(n, 2)
    global_angle = params[2 * n]
    
    # Center the cluster
    center = np.mean(positions, axis=0)
    centered = positions - center
    
    # Compute outer side length needed
    outer_side = _compute_outer_side(centered, global_angle)
    
    # Check validity (with some tolerance)
    if not _check_validity(centered, global_angle, outer_side):
        return 1e6 + outer_side  # Penalize invalid configs
    
    return outer_side

def hexagon_packing_11():
    """
    Constructs a packing of 11 disjoint unit regular hexagons inside a larger regular hexagon.
    """
    start_time = time.time()
    deadline = start_time + 580  # Safety margin
    
    # Initial cluster: hexagonal pattern
    base_positions = _cluster_config(None)
    
    # Optimize positions and global angle
    n = 11
    x0 = np.zeros(2 * n + 1)
    x0[:2 * n] = base_positions.flatten()
    x0[2 * n] = 0  # initial global angle
    
    # Use Nelder-Mead for local optimization
    best_result = None
    best_side = np.inf
    
    # Try multiple starting angles
    for start_angle in [0, 15, 30, 45, 60, 75, 90]:
        x0_trial = x0.copy()
        x0_trial[2 * n] = np.radians(start_angle)
        
        try:
            result = minimize(_objective, x0_trial, method='Nelder-Mead',
                             options={'maxiter': 2000, 'xatol': 1e-8, 'fatol': 1e-8})
            
            if result.fun < best_side:
                best_side = result.fun
                best_result = result.x
        except Exception:
            continue
        
        if time.time() > deadline:
            break
    
    if best_result is None:
        # Fallback: return the initial configuration with a large enough outer hexagon
        positions = base_positions
        center = np.mean(positions, axis=0)
        centered = positions - center
        outer_side = _compute_outer_side(centered, 0)
        # Add margin
        outer_side *= 1.05
        inner_data = np.column_stack([centered[:, 0], centered[:, 1], np.zeros(n)])
        outer_data = np.array([0, 0, 0])
        return inner_data, outer_data, outer_side
    
    # Extract best configuration
    positions = best_result[:2 * n].reshape(n, 2)
    global_angle = best_result[2 * n]
    
    # Center the cluster
    center = np.mean(positions, axis=0)
    centered = positions - center
    
    # Compute outer side
    outer_side = _compute_outer_side(centered, global_angle)
    
    # Verify and add small margin
    if not _check_validity(centered, global_angle, outer_side):
        outer_side *= 1.01
    
    # Build output
    inner_data = np.column_stack([centered[:, 0], centered[:, 1], np.zeros(n)])
    outer_data = np.array([0, 0, 0])
    
    return inner_data, outer_data, outer_side

# EVOLVE-BLOCK-END
