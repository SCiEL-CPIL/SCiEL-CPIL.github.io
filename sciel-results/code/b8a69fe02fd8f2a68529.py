# EVOLVE-BLOCK-START
import numpy as np
import time
import math
from scipy.optimize import minimize

def _hex_vertices(cx, cy, d, angle_deg):
    """Vertices of a regular hexagon centered at (cx,cy) with side d and angle (deg)."""
    a = np.radians(angle_deg)
    thetas = a + np.radians(60 * np.arange(6))
    return np.column_stack([cx + d * np.cos(thetas),
                            cy + d * np.sin(thetas)])

def _hex_edge_normals(vertices):
    """Outward edge normals for a hexagon given its 6 vertices (CCW)."""
    v = vertices
    edges = np.roll(v, -1, axis=0) - v
    normals = np.column_stack([edges[:, 1], -edges[:, 0]])
    norms = np.linalg.norm(normals, axis=1, keepdims=True)
    normals = normals / np.where(norms == 0, 1.0, norms)
    return normals

def _separation_gap(verts1, verts2):
    """
    Separating axis test: returns the minimum gap (positive = separated, negative = overlap).
    """
    n1 = _hex_edge_normals(verts1)
    n2 = _hex_edge_normals(verts2)
    all_normals = np.vstack([n1, n2])
    
    min_gap = np.inf
    for normal in all_normals:
        proj1 = verts1 @ normal
        proj2 = verts2 @ normal
        gap1 = proj2.min() - proj1.max()
        gap2 = proj1.min() - proj2.max()
        min_gap = min(min_gap, gap1, gap2)
    return min_gap

def _compute_min_s(centers, angles, outer_center, outer_angle):
    """
    Compute minimum s of a regular hexagon (centered at outer_center, angle outer_angle) containing all inner hexagons.
    """
    a = np.radians(outer_angle)
    normals = []
    for k in range(6):
        theta = a + np.radians(30 + 60 * k)
        normals.append(np.array([np.cos(theta), np.sin(theta)]))
    
    max_proj = 0.0
    for i in range(12):
        verts = _hex_vertices(centers[i, 0], centers[i, 1], 1.0, angles[i])
        for v in verts:
            rel = v - outer_center
            for n in normals:
                proj = abs(rel @ n)
                if proj > max_proj:
                    max_proj = proj
    return max_proj / (np.sqrt(3) / 2)

def _check_validity(centers, angles, s, outer_center, outer_angle):
    """Check if configuration is valid: no overlaps, all vertices inside outer hexagon."""
    if s < np.sqrt(12) - 1e-8:
        return False
    
    # Check overlaps
    for i in range(12):
        verts_i = _hex_vertices(centers[i, 0], centers[i, 1], 1.0, angles[i])
        for j in range(i + 1, 12):
            verts_j = _hex_vertices(centers[j, 0], centers[j, 1], 1.0, angles[j])
            gap = _separation_gap(verts_i, verts_j)
            if gap < -1e-6:
                return False
    
    # Check boundary
    outer_verts = _hex_vertices(outer_center[0], outer_center[1], s, outer_angle)
    for i in range(12):
        verts_i = _hex_vertices(centers[i, 0], centers[i, 1], 1.0, angles[i])
        for v in verts_i:
            for k in range(6):
                p1 = outer_verts[k]
                p2 = outer_verts[(k + 1) % 6]
                cross = (p2[0] - p1[0]) * (v[1] - p1[1]) - (p2[1] - p1[1]) * (v[0] - p1[0])
                if cross < -1e-6:
                    return False
    return True

def _objective_lbf(x, deadline, outer_center, outer_angle):
    """
    L-BFGS objective: minimize s + penalty for constraint violations.
    x: 24-vector = 12*(cx, cy)
    """
    if time.time() > deadline:
        return 1e10
    
    try:
        centers = x.reshape(12, 2)
    except ValueError:
        return 1e10

    angles = np.zeros(12)
    
    s = _compute_min_s(centers, angles, outer_center, outer_angle)
    if s < np.sqrt(12):
        s = np.sqrt(12) + 1e-6
    
    # Overlap penalty
    overlap_pen = 0.0
    for i in range(12):
        verts_i = _hex_vertices(centers[i, 0], centers[i, 1], 1.0, angles[i])
        for j in range(i + 1, 12):
            verts_j = _hex_vertices(centers[j, 0], centers[j, 1], 1.0, angles[j])
            gap = _separation_gap(verts_i, verts_j)
            if gap < 0:
                overlap_pen += (-gap) ** 2
    
    # Boundary penalty
    boundary_pen = 0.0
    outer_verts = _hex_vertices(outer_center[0], outer_center[1], s, outer_angle)
    for i in range(12):
        verts_i = _hex_vertices(centers[i, 0], centers[i, 1], 1.0, angles[i])
        for v in verts_i:
            for k in range(6):
                p1 = outer_verts[k]
                p2 = outer_verts[(k + 1) % 6]
                cross = (p2[0] - p1[0]) * (v[1] - p1[1]) - (p2[1] - p1[1]) * (v[0] - p1[0])
                if cross < 0:
                    boundary_pen += (-cross) ** 2
    
    return s + 1000 * (overlap_pen + boundary_pen)

def _initial_flower():
    """Generate initial 12-hexagon pattern: 6 in inner ring, 6 in outer ring."""
    d_inner = 2 * np.sqrt(3)
    d_outer = 4 * np.sqrt(3)
    centers = []
    # Inner ring: 6 hexagons
    for k in range(6):
        angle = np.radians(60 * k)
        centers.append([d_inner * np.cos(angle), d_inner * np.sin(angle)])
    # Outer ring: 6 hexagons, offset by 30 degrees
    for k in range(6):
        angle = np.radians(60 * k + 30)
        centers.append([d_outer * np.cos(angle), d_outer * np.sin(angle)])
    
    centers = np.array(centers)
    centers -= centers.mean(axis=0)
    return centers

def _initial_grid():
    """Generate initial hexagonal grid arrangement."""
    d = 2 * np.sqrt(3)
    centers = []
    for row in range(3):
        for col in range(4):
            x = (col - 1.5) * d
            y = (row - 1) * d * np.sqrt(3)
            centers.append([x, y])
    
    centers = np.array(centers)
    centers -= centers.mean(axis=0)
    return centers

def hexagon_packing_12():
    start_time = time.time()
    deadline = start_time + 590  # Safety margin
    
    best_score = 0.0
    best_result = None
    
    np.random.seed(42)
    
    # Try multiple initial configurations and outer angles
    initial_configs = [_initial_flower(), _initial_grid()]
    outer_angles = np.linspace(0, 60, 13)
    
    for config_idx, init_centers in enumerate(initial_configs):
        for outer_angle in outer_angles:
            if time.time() > deadline - 30:
                break
            
            outer_center = np.array([0.0, 0.0])
            
            # L-BFGS optimization
            x0 = init_centers.flatten()
            
            def obj(x):
                return _objective_lbf(x, deadline, outer_center, outer_angle)
            
            try:
                result = minimize(obj, x0, method='L-BFGS-B', 
                                 options={'maxiter': 500, 'ftol': 1e-10, 'gtol': 1e-8})
                
                if result.x.size == 24:
                    centers = result.x.reshape(12, 2)
                    angles = np.zeros(12)
                    
                    # Recompute optimal s
                    s = _compute_min_s(centers, angles, outer_center, outer_angle)
                    
                    # Validate
                    if _check_validity(centers, angles, s, outer_center, outer_angle):
                        score = 1.0 / s
                        if score > best_score:
                            best_score = score
                            best_result = (centers, angles, s, outer_center, outer_angle)
            except Exception as e:
                pass
    
    # If we found a good solution, use it
    if best_result is not None:
        centers, angles, s, outer_center, outer_angle = best_result
        
        inner_hex_data = np.column_stack([centers, angles])
        outer_hex_data = np.array([outer_center[0], outer_center[1], outer_angle])
        
        return inner_hex_data, outer_hex_data, float(s)
    
    # Fallback: use the seed configuration
    inner_hex_data = np.array(
        [
            [0, 0, 0],
            [-2.5, 0, 0],
            [2.5, 0, 0],
            [-1.25, 2.17, 0],
            [1.25, 2.17, 0],
            [-1.25, -2.17, 0],
            [1.25, -2.17, 0],
            [-3.75, 2.17, 0],
            [3.75, 2.17, 0],
            [-3.75, -2.17, 0],
            [3.75, -2.17, 0],
            [0, -4, 0],
        ]
    )
    outer_hex_data = np.array([0, 0, 0])
    outer_hex_side_length = 8
    
    return inner_hex_data, outer_hex_data, outer_hex_side_length

# EVOLVE-BLOCK-END
