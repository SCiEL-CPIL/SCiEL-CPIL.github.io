# EVOLVE-BLOCK-START
import numpy as np
import time
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

def _compute_min_s(centers, angles):
    """
    Compute minimum s of a regular hexagon (centered at origin, angle 0) containing all inner hexagons.
    """
    max_s = 0.0
    for i in range(12):
        verts = _hex_vertices(centers[i, 0], centers[i, 1], 1.0, angles[i])
        for v in verts:
            px, py = v[0], v[1]
            for k in range(6):
                angle = np.radians(30 + 60 * k)
                val = px * np.cos(angle) + py * np.sin(angle)
                s_required = val / (np.sqrt(3) / 2)
                if s_required > max_s:
                    max_s = s_required
    return max_s

def _check_validity(centers, angles, s):
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
    outer_verts = _hex_vertices(0, 0, s, 0)
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

def _objective_sa(x, deadline):
    """
    Simulated annealing objective: minimize s + penalty for constraint violations.
    x: 36-vector = 12*(cx, cy, angle_deg)
    """
    if time.time() > deadline:
        return 1e10
    
    centers = x[:24].reshape(12, 2)
    angles = x[24:36]
    
    s = _compute_min_s(centers, angles)
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
    outer_verts = _hex_vertices(0, 0, s, 0)
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

def _sa_step(x, temp, deadline):
    """One step of simulated annealing."""
    # Generate perturbation
    x_new = x.copy()
    
    # Move a random hexagon
    idx = np.random.randint(12)
    x_new[idx*2] += np.random.randn() * temp * 0.5
    x_new[idx*2+1] += np.random.randn() * temp * 0.5
    x_new[24+idx] += np.random.randn() * temp * 0.1
    
    # Occasionally rotate the whole configuration
    if np.random.rand() < 0.1:
        theta = np.random.randn() * temp * 0.05
        c, s_ = np.cos(theta), np.sin(theta)
        for i in range(12):
            cx, cy = x_new[i*2], x_new[i*2+1]
            x_new[i*2] = c*cx - s_*cy
            x_new[i*2+1] = s_*cx + c*cy
    
    return x_new

def hexagon_packing_12():
    start_time = time.time()
    deadline = start_time + 600
    
    best_score = 0.0
    best_x = None
    
    # Initial guess: hexagonal grid arrangement
    d = 2 * np.sqrt(3)
    initial_centers = []
    for k in range(4):
        initial_centers.append([(k - 1.5) * d, 0])
    for k in range(4):
        initial_centers.append([(k - 1.5) * d, d * np.sqrt(3)])
    for k in range(4):
        initial_centers.append([(k - 1.5) * d, 2 * d * np.sqrt(3)])
    
    initial_centers = np.array(initial_centers)
    initial_angles = np.zeros(12)
    initial_centers -= initial_centers.mean(axis=0)
    
    x0 = np.concatenate([initial_centers.flatten(), initial_angles])
    
    np.random.seed(42)
    
    # Run multiple SA restarts
    n_restarts = 30
    steps_per_restart = 5000
    
    for restart in range(n_restarts):
        if time.time() > deadline - 30:
            break
        
        if restart == 0:
            x = x0.copy()
        else:
            x = x0 + np.random.randn(36) * 0.3
            x[24:36] = np.random.uniform(0, 60, 12)
        
        # Simulated annealing
        temp_start = 1.0
        temp_end = 0.001
        cooling = (temp_end / temp_start) ** (1.0 / steps_per_restart)
        temp = temp_start
        
        current_obj = _objective_sa(x, deadline)
        
        for step in range(steps_per_restart):
            if time.time() > deadline - 30:
                break
            
            x_new = _sa_step(x, temp, deadline)
            obj_new = _objective_sa(x_new, deadline)
            
            # Acceptance criterion
            if obj_new < current_obj or np.random.rand() < np.exp((current_obj - obj_new) / temp):
                x = x_new
                current_obj = obj_new
            
            temp *= cooling
        
        # Evaluate final configuration
        centers = x[:24].reshape(12, 2)
        angles = x[24:36]
        s = _compute_min_s(centers, angles)
        
        if _check_validity(centers, angles, s):
            score = 1.0 / s
            if score > best_score:
                best_score = score
                best_x = x.copy()
    
    # If we found a good solution, use it
    if best_x is not None:
        centers = best_x[:24].reshape(12, 2)
        angles = best_x[24:36]
        s = _compute_min_s(centers, angles)
        
        inner_hex_data = np.column_stack([centers, angles])
        outer_hex_data = np.array([0.0, 0.0, 0.0])
        
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
