"""
Hexagon packing: 12 unit hexagons in a regular hexagon, minimize outer side length.
Strategy: Individual hexagon optimization with force-based relaxation and diverse initializations.
"""
import numpy as np
import time
from scipy.optimize import minimize

N = 12
DEADLINE = 145.0  # Safety margin before 150s kill

# Unit hexagon vertices (pointy-top, side length 1)
UNIT_VERTS = np.array([
    [np.cos(np.radians(60 * i + 30)), np.sin(np.radians(60 * i + 30))] for i in range(6)
])

# Edge normals (outward) for containment check
EDGE_NORMALS = np.array([
    [np.cos(np.radians(60 * i)), np.sin(np.radians(60 * i))] for i in range(6)
])


def get_hex_vertices(center, angle_deg):
    """Get vertices of a unit hexagon at given center and angle."""
    a_rad = np.radians(angle_deg)
    c = np.cos(a_rad)
    s = np.sin(a_rad)
    R = np.array([[c, -s], [s, c]])
    return UNIT_VERTS @ R.T + center


def required_outer_side(centers, angles, outer_center, outer_angle_deg):
    """Compute minimum outer side length to contain all hexagons."""
    all_verts = []
    for i in range(N):
        all_verts.append(get_hex_vertices(centers[i], angles[i]))
    all_verts = np.array(all_verts)  # (N, 6, 2)
    
    oa_rad = np.radians(outer_angle_deg)
    c_o = np.cos(-oa_rad)
    s_o = np.sin(-oa_rad)
    R_o = np.array([[c_o, -s_o], [s_o, c_o]])
    
    flat_verts = all_verts.reshape(-1, 2)
    v_rel = flat_verts - outer_center
    v_rot = v_rel @ R_o.T
    projs = v_rot @ EDGE_NORMALS.T
    abs_projs = np.abs(projs)
    max_proj = np.max(abs_projs, axis=1)
    max_extent = np.max(max_proj)
    
    return max_extent / (np.sqrt(3) / 2)


def sat_separation(v1, v2):
    """Compute minimum separation between two hexagons using SAT. Returns negative if overlapping."""
    edges1 = np.roll(v1, -1, axis=0) - v1
    edges2 = np.roll(v2, -1, axis=0) - v2
    axes = np.vstack([edges1, edges2])
    norms = np.linalg.norm(axes, axis=1, keepdims=True)
    axes = axes / (norms + 1e-15)
    
    proj1 = v1 @ axes.T
    proj2 = v2 @ axes.T
    
    sep = np.minimum(proj1.min(axis=0) - proj2.max(axis=0),
                    proj2.min(axis=0) - proj1.max(axis=0))
    
    return np.max(sep)


def check_validity(centers, angles, outer_center, outer_side, outer_angle_deg, tol=1e-6):
    """Check if configuration is valid (no overlaps, contained in outer hexagon)."""
    all_verts = []
    for i in range(N):
        all_verts.append(get_hex_vertices(centers[i], angles[i]))
    all_verts = np.array(all_verts)
    
    # Check internal overlaps
    for i in range(N):
        for j in range(i + 1, N):
            sep = sat_separation(all_verts[i], all_verts[j])
            if sep < -tol:
                return False
    
    # Check containment
    apothem = outer_side * np.sqrt(3) / 2
    oa_rad = np.radians(outer_angle_deg)
    c_o = np.cos(-oa_rad)
    s_o = np.sin(-oa_rad)
    R_o = np.array([[c_o, -s_o], [s_o, c_o]])
    
    flat_verts = all_verts.reshape(-1, 2)
    v_rel = flat_verts - outer_center
    v_rot = v_rel @ R_o.T
    projs = v_rot @ EDGE_NORMALS.T
    abs_projs = np.abs(projs)
    max_proj = np.max(abs_projs, axis=1)
    
    if np.max(max_proj) > apothem + tol:
        return False
    
    return True


def generate_hex_cluster_config():
    """Generate a hexagonal cluster configuration."""
    a1 = np.array([np.sqrt(3), 0])
    a2 = np.array([np.sqrt(3) / 2, 1.5])
    
    coords = [
        (0, 0),
        (1, 0), (-1, 0), (0, 1), (0, -1), (1, -1), (-1, 1),
        (2, -1), (-2, 1), (0, 2), (0, -2), (2, 0)
    ]
    
    centers = []
    for q, r in coords:
        x = np.sqrt(3) * (q + r / 2.0)
        y = 1.5 * r
        centers.append([x, y])
    
    centers = np.array(centers)
    cx = np.mean(centers[:, 0])
    cy = np.mean(centers[:, 1])
    centers -= np.array([cx, cy])
    
    angles = np.zeros(N)
    return centers, angles


def generate_grid_config():
    """Generate a grid configuration."""
    centers = []
    for i in range(3):
        for j in range(4):
            x = (i - 1) * np.sqrt(3) * 1.01
            y = (j - 1.5) * 1.5 * 1.01
            centers.append([x, y])
    
    centers = np.array(centers)
    cx = np.mean(centers[:, 0])
    cy = np.mean(centers[:, 1])
    centers -= np.array([cx, cy])
    
    angles = np.zeros(N)
    return centers, angles


def generate_staggered_config():
    """Generate a staggered configuration."""
    centers = []
    for i in range(3):
        for j in range(4):
            x = (i - 1) * np.sqrt(3) * 1.02 + (j % 2) * np.sqrt(3) * 0.51
            y = (j - 1.5) * 1.5 * 1.02
            centers.append([x, y])
    
    centers = np.array(centers)
    cx = np.mean(centers[:, 0])
    cy = np.mean(centers[:, 1])
    centers -= np.array([cx, cy])
    
    angles = np.zeros(N)
    return centers, angles


def generate_circular_config():
    """Generate a circular configuration."""
    centers = []
    for i in range(N):
        theta = 2 * np.pi * i / N
        x = 2.5 * np.cos(theta)
        y = 2.5 * np.sin(theta)
        centers.append([x, y])
    
    centers = np.array(centers)
    cx = np.mean(centers[:, 0])
    cy = np.mean(centers[:, 1])
    centers -= np.array([cx, cy])
    
    angles = np.zeros(N)
    return centers, angles


def force_based_relaxation(centers, angles, outer_center, outer_angle_deg, max_iters=500, dt=0.01):
    """
    Relax configuration using force-based approach.
    Forces: 
    1. Repulsion between overlapping hexagons
    2. Attraction toward outer hexagon center
    3. Containment constraint
    """
    centers = centers.copy()
    angles = angles.copy()
    
    for iter in range(max_iters):
        if time.time() > DEADLINE - 5:
            break
        
        # Compute forces
        forces = np.zeros_like(centers)
        
        # Repulsion between pairs
        for i in range(N):
            v_i = get_hex_vertices(centers[i], angles[i])
            for j in range(i + 1, N):
                v_j = get_hex_vertices(centers[j], angles[j])
                sep = sat_separation(v_i, v_j)
                if sep < 0:  # Overlapping
                    # Compute separation vector
                    center_i = np.mean(v_i, axis=0)
                    center_j = np.mean(v_j, axis=0)
                    sep_vec = center_i - center_j
                    sep_norm = np.linalg.norm(sep_vec)
                    if sep_norm > 1e-10:
                        sep_vec = sep_vec / sep_norm
                    else:
                        sep_vec = np.array([1.0, 0.0])
                    
                    force_mag = -sep * 10.0  # Strong repulsion
                    forces[i] += sep_vec * force_mag
                    forces[j] -= sep_vec * force_mag
        
        # Attraction toward center (weak)
        for i in range(N):
            to_center = outer_center - centers[i]
            forces[i] += to_center * 0.1
        
        # Update positions
        for i in range(N):
            centers[i] += forces[i] * dt
        
        # Enforce containment
        apothem_target = 4.0  # Target apothem
        for i in range(N):
            v_i = get_hex_vertices(centers[i], angles[i])
            flat_v = v_i.reshape(-1, 2)
            v_rel = flat_v - outer_center
            projs = v_rel @ EDGE_NORMALS.T
            abs_projs = np.abs(projs)
            max_proj = np.max(abs_projs, axis=1)
            
            if np.max(max_proj) > apothem_target:
                # Push back toward center
                to_center = outer_center - centers[i]
                centers[i] += to_center * 0.1
    
    return centers, angles


def objective_individual(params, outer_center, outer_angle_deg):
    """
    Objective function for individual hexagon optimization.
    params: (N*3,) array of [x1, y1, a1, x2, y2, a2, ...]
    """
    centers = params[:N*2].reshape(N, 2)
    angles = params[N*2:]
    
    # Compute required outer side
    side = required_outer_side(centers, angles, outer_center, outer_angle_deg)
    
    # Check for overlaps and add penalty
    all_verts = []
    for i in range(N):
        all_verts.append(get_hex_vertices(centers[i], angles[i]))
    all_verts = np.array(all_verts)
    
    overlap_penalty = 0.0
    for i in range(N):
        for j in range(i + 1, N):
            sep = sat_separation(all_verts[i], all_verts[j])
            if sep < 0:
                overlap_penalty += (-sep) ** 2 * 100.0
    
    return side + overlap_penalty


def construct():
    start_time = time.time()
    np.random.seed(42)
    
    outer_center = np.array([0.0, 0.0])
    outer_angle = 0.0
    
    best_side = float('inf')
    best_centers = None
    best_angles = None
    
    # Generate diverse initial configurations
    initial_configs = [
        generate_hex_cluster_config(),
        generate_grid_config(),
        generate_staggered_config(),
        generate_circular_config(),
    ]
    
    # Add some random configurations
    for _ in range(3):
        centers = np.random.uniform(-3, 3, (N, 2))
        angles = np.random.uniform(0, 60, N)
        initial_configs.append((centers, angles))
    
    for centers, angles in initial_configs:
        if time.time() > DEADLINE - 30:
            break
        
        # Force-based relaxation
        relaxed_centers, relaxed_angles = force_based_relaxation(
            centers, angles, outer_center, outer_angle, 
            max_iters=200, dt=0.02
        )
        
        # Compute side length
        side = required_outer_side(relaxed_centers, relaxed_angles, outer_center, outer_angle)
        
        # Check validity
        if check_validity(relaxed_centers, relaxed_angles, outer_center, side, outer_angle):
            if side < best_side:
                best_side = side
                best_centers = relaxed_centers.copy()
                best_angles = relaxed_angles.copy()
        
        # Nelder-Mead refinement on best candidate
        if side < best_side or best_side == float('inf'):
            params = np.concatenate([relaxed_centers.flatten(), relaxed_angles])
            
            res = minimize(
                objective_individual,
                params,
                args=(outer_center, outer_angle),
                method='Nelder-Mead',
                options={'maxiter': 500, 'xatol': 1e-5, 'fatol': 1e-5}
            )
            
            if res.fun < best_side:
                refined_centers = res.x[:N*2].reshape(N, 2)
                refined_angles = res.x[N*2:]
                
                # Check validity
                refined_side = required_outer_side(refined_centers, refined_angles, outer_center, outer_angle)
                if check_validity(refined_centers, refined_angles, outer_center, refined_side, outer_angle):
                    if refined_side < best_side:
                        best_side = refined_side
                        best_centers = refined_centers
                        best_angles = refined_angles
    
    # Final refinement if we have a good candidate
    if best_centers is not None and time.time() < DEADLINE - 15:
        params = np.concatenate([best_centers.flatten(), best_angles])
        
        res = minimize(
            objective_individual,
            params,
            args=(outer_center, outer_angle),
            method='Nelder-Mead',
            options={'maxiter': 1000, 'xatol': 1e-6, 'fatol': 1e-6}
        )
        
        if res.fun < best_side:
            refined_centers = res.x[:N*2].reshape(N, 2)
            refined_angles = res.x[N*2:]
            
            refined_side = required_outer_side(refined_centers, refined_angles, outer_center, outer_angle)
            if check_validity(refined_centers, refined_angles, outer_center, refined_side, outer_angle):
                best_side = refined_side
                best_centers = refined_centers
                best_angles = refined_angles
    
    # Fallback if no valid configuration found
    if best_centers is None:
        centers, angles = generate_hex_cluster_config()
        best_side = required_outer_side(centers, angles, outer_center, outer_angle)
        best_centers = centers
        best_angles = angles
    
    # Construct return format
    inner_list = []
    for i in range(N):
        inner_list.append([best_centers[i, 0], best_centers[i, 1], best_angles[i]])
    
    return inner_list, (0.0, 0.0), best_side, outer_angle


if __name__ == "__main__":
    result = construct()
    print(f"Best outer side length: {result[2]:.6f}")
    print(f"Number of inner hexagons: {len(result[0])}")
