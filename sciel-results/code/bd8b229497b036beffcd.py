import numpy as np
import time
from scipy.optimize import differential_evolution
from scipy.spatial import ConvexHull
import itertools

N = 12
DEADLINE = 145.0

# Unit hexagon vertices (flat-top, side length 1)
UNIT_VERTS = np.array([
    [np.cos(np.radians(60 * i)), np.sin(np.radians(60 * i))] for i in range(6)
])

# Outward normals for the 6 edges of a regular hexagon
OUTER_NORMALS = np.array([
    [np.cos(np.radians(60 * i + 30)), np.sin(np.radians(60 * i + 30))] for i in range(6)
])


def get_vertices(centers):
    """
    centers: (N, 3) array of [x, y, angle_deg]
    Returns: (N, 6, 2) array of vertices
    """
    n = len(centers)
    verts = np.zeros((n, 6, 2))
    for i in range(n):
        x, y, a_deg = centers[i]
        a_rad = np.radians(a_deg)
        c = np.cos(a_rad)
        s = np.sin(a_rad)
        R = np.array([[c, -s], [s, c]])
        verts[i] = UNIT_VERTS @ R.T + np.array([x, y])
    return verts


def required_outer_side(centers, outer_center, outer_angle):
    """Compute the minimum outer side length needed to contain all inner hexagons."""
    verts = get_vertices(np.array(centers, dtype=float))
    all_verts = verts.reshape(-1, 2)
    
    oa_rad = np.radians(outer_angle)
    c_o = np.cos(-oa_rad)
    s_o = np.sin(-oa_rad)
    R_o = np.array([[c_o, -s_o], [s_o, c_o]])
    
    v_rel = all_verts - outer_center
    v_rot = v_rel @ R_o.T
    projs = v_rot @ OUTER_NORMALS.T
    abs_projs = np.abs(projs)
    max_extent = np.max(abs_projs)
    
    return max_extent / (np.sqrt(3) / 2)


def check_validity(centers, outer_center, outer_side, outer_angle, tol=1e-9):
    """Check if packing is valid: all inner hexagons inside outer, no overlaps."""
    verts = get_vertices(np.array(centers, dtype=float))
    n = verts.shape[0]
    
    # Check pairwise separation using SAT
    for i in range(n):
        v1 = verts[i]
        for j in range(i + 1, n):
            v2 = verts[j]
            edges1 = np.roll(v1, -1, axis=0) - v1
            edges2 = np.roll(v2, -1, axis=0) - v2
            axes = np.vstack([edges1, edges2])
            norms = np.linalg.norm(axes, axis=1, keepdims=True)
            axes = axes / (norms + 1e-15)
            
            proj1 = v1 @ axes.T
            proj2 = v2 @ axes.T
            
            sep = np.minimum(proj1.min(axis=0) - proj2.max(axis=0),
                            proj2.min(axis=0) - proj1.max(axis=0))
            
            if np.max(sep) < -tol:
                return False
    
    # Check containment in outer hexagon
    apothem = outer_side * np.sqrt(3) / 2
    oa_rad = np.radians(outer_angle)
    c_o = np.cos(-oa_rad)
    s_o = np.sin(-oa_rad)
    R_o = np.array([[c_o, -s_o], [s_o, c_o]])
    
    all_verts = verts.reshape(-1, 2)
    v_rel = all_verts - outer_center
    v_rot = v_rel @ R_o.T
    projs = v_rot @ OUTER_NORMALS.T
    abs_projs = np.abs(projs)
    
    if np.max(abs_projs) > apothem + tol:
        return False
    
    return True


def generate_hex_lattice_arrangements():
    """
    Generate diverse, compact initial configurations using hexagonal lattice patterns.
    Each configuration is a list of (i, j) coordinates in a hex lattice basis.
    """
    configs = []
    
    # Hex lattice basis vectors
    a1 = np.array([np.sqrt(3), 0.0])
    a2 = np.array([np.sqrt(3) / 2.0, 1.5])
    
    def make_config(coords, angle=0.0):
        centers = []
        for i, j in coords:
            pos = i * a1 + j * a2
            centers.append([pos[0], pos[1], angle])
        return centers
    
    # Pattern 1: 3-4-3-2
    p1 = [
        (0, 0), (1, 0), (2, 0),
        (-1, 1), (0, 1), (1, 1), (2, 1),
        (0, 2), (1, 2), (2, 2),
        (0, 3), (1, 3)
    ]
    configs.append(make_config(p1))
    
    # Pattern 2: 2-4-4-2
    p2 = [
        (0, 0), (1, 0),
        (-1, 1), (0, 1), (1, 1), (2, 1),
        (-1, 2), (0, 2), (1, 2), (2, 2),
        (0, 3), (1, 3)
    ]
    configs.append(make_config(p2))
    
    # Pattern 3: 1-3-4-3-1
    p3 = [
        (0, 0),
        (-1, 1), (0, 1), (1, 1),
        (-2, 2), (-1, 2), (0, 2), (1, 2),
        (-1, 3), (0, 3), (1, 3),
        (0, 4)
    ]
    configs.append(make_config(p3))
    
    # Pattern 4: 4-4-4
    p4 = [
        (-1, 0), (0, 0), (1, 0), (2, 0),
        (-1, 1), (0, 1), (1, 1), (2, 1),
        (-1, 2), (0, 2), (1, 2), (2, 2)
    ]
    configs.append(make_config(p4))
    
    # Pattern 5: 2-3-4-3
    p5 = [
        (0, 0), (1, 0),
        (-1, 1), (0, 1), (1, 1),
        (-2, 2), (-1, 2), (0, 2), (1, 2),
        (-1, 3), (0, 3), (1, 3)
    ]
    configs.append(make_config(p5))
    
    # Pattern 6: 3-3-3-3
    p6 = [
        (-1, 0), (0, 0), (1, 0),
        (-1, 1), (0, 1), (1, 1),
        (-1, 2), (0, 2), (1, 2),
        (-1, 3), (0, 3), (1, 3)
    ]
    configs.append(make_config(p6))
    
    # Pattern 7: Center + ring
    p7 = [
        (0, 0),
        (1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (-1, 1),
        (2, 0), (-2, 0), (0, 2), (0, -2), (1, 2)
    ]
    configs.append(make_config(p7))
    
    # Pattern 8: 3-4-3-2 variant
    p8 = [
        (-1, 0), (0, 0), (1, 0),
        (-1, 1), (0, 1), (1, 1), (2, 1),
        (0, 2), (1, 2), (2, 2), (3, 2),
        (0, 3), (1, 3)
    ]
    configs.append(make_config(p8))
    
    # Add perturbed versions
    for i in range(4):
        base = configs[i % len(configs)]
        np.random.seed(100 + i)
        perturbed = []
        for c in base:
            dx = np.random.uniform(-0.15, 0.15)
            dy = np.random.uniform(-0.15, 0.15)
            da = np.random.uniform(-10, 10)
            perturbed.append([c[0] + dx, c[1] + dy, c[2] + da])
        configs.append(perturbed)
        
    return configs


def compute_cluster_side_length(centers, outer_center, outer_angle):
    """
    Compute the required outer side length for a given configuration.
    """
    verts = get_vertices(np.array(centers, dtype=float))
    all_verts = verts.reshape(-1, 2)
    
    oa_rad = np.radians(outer_angle)
    c_o = np.cos(-oa_rad)
    s_o = np.sin(-oa_rad)
    R_o = np.array([[c_o, -s_o], [s_o, c_o]])
    
    v_rel = all_verts - outer_center
    v_rot = v_rel @ R_o.T
    projs = v_rot @ OUTER_NORMALS.T
    abs_projs = np.abs(projs)
    max_extent = np.max(abs_projs)
    
    return max_extent / (np.sqrt(3) / 2)


def optimize_rotation(centers, outer_center, num_steps=120):
    """
    Find the best global rotation of the inner arrangement to minimize outer side length.
    """
    best_side = float('inf')
    best_angle = 0.0
    
    angles = np.linspace(0, 60, num_steps)
    
    for rot_deg in angles:
        rotated_centers = []
        for c in centers:
            rotated_centers.append([c[0], c[1], c[2] + rot_deg])
        
        side = compute_cluster_side_length(rotated_centers, outer_center, 0.0)
        
        if side < best_side:
            best_side = side
            best_angle = rot_deg
    
    return best_side, best_angle


def refine_rotation(centers, outer_center, center_angle, num_steps=50, range_deg=5.0):
    """Fine-tune the rotation angle around a given center angle."""
    best_side = float('inf')
    best_angle = center_angle
    
    angles = np.linspace(center_angle - range_deg, center_angle + range_deg, num_steps)
    
    for rot_deg in angles:
        rotated_centers = []
        for c in centers:
            rotated_centers.append([c[0], c[1], c[2] + rot_deg])
        
        side = compute_cluster_side_length(rotated_centers, outer_center, 0.0)
        
        if side < best_side:
            best_side = side
            best_angle = rot_deg
    
    return best_side, best_angle


def optimize_position(centers, outer_center, num_steps=100):
    """
    Optimize the global translation of the cluster to minimize the required outer side length.
    """
    best_side = float('inf')
    best_tx, best_ty = 0.0, 0.0
    
    # Try a grid of translations
    tx_range = np.linspace(-2.0, 2.0, num_steps)
    ty_range = np.linspace(-2.0, 2.0, num_steps)
    
    for tx in tx_range:
        for ty in ty_range:
            translated_centers = []
            for c in centers:
                translated_centers.append([c[0] + tx, c[1] + ty, c[2]])
            
            side = compute_cluster_side_length(translated_centers, outer_center, 0.0)
            
            if side < best_side:
                best_side = side
                best_tx, best_ty = tx, ty
    
    return best_side, best_tx, best_ty


def construct():
    start_time = time.time()
    
    best_score = float('inf')
    best_result = None
    
    configs = generate_hex_lattice_arrangements()
    
    # Shuffle for diversity
    np.random.shuffle(configs)
    
    outer_center = np.array([0.0, 0.0])
    
    for i, config in enumerate(configs):
        elapsed = time.time() - start_time
        if elapsed > DEADLINE - 10:
            break
        
        # Optimize rotation
        elapsed = time.time() - start_time
        if elapsed > DEADLINE - 5:
            break
        
        rot_time = min(8.0, DEADLINE - elapsed - 2)
        if rot_time > 1.0:
            side_rot, angle_rot = optimize_rotation(config, outer_center, num_steps=60)
            # Apply the best rotation
            rotated_config = []
            for c in config:
                rotated_config.append([c[0], c[1], c[2] + angle_rot])
            
            # Fine-tune rotation
            side_refined, angle_refined = refine_rotation(rotated_config, outer_center, angle_rot, num_steps=30, range_deg=3.0)
            rotated_config = []
            for c in config:
                rotated_config.append([c[0], c[1], c[2] + angle_refined])
            
            # Check validity of rotated version
            req_side = compute_cluster_side_length(rotated_config, outer_center, 0.0)
            if check_validity(rotated_config, outer_center, req_side, 0.0):
                if req_side < best_score:
                    best_score = req_side
                    best_result = (rotated_config, (0.0, 0.0), req_side, 0.0)
        else:
            # Just use the unrotated version
            req_side = compute_cluster_side_length(config, outer_center, 0.0)
            if check_validity(config, outer_center, req_side, 0.0):
                if req_side < best_score:
                    best_score = req_side
                    best_result = (config, (0.0, 0.0), req_side, 0.0)
        
        # Also check the original config
        orig_side = compute_cluster_side_length(config, outer_center, 0.0)
        if check_validity(config, outer_center, orig_side, 0.0):
            if orig_side < best_score:
                best_score = orig_side
                best_result = (config, (0.0, 0.0), orig_side, 0.0)
    
    if best_result is not None:
        return best_result
    
    # Fallback: Simple circular arrangement
    inner_default = []
    for i in range(N):
        theta = 2 * np.pi * i / N
        x = 3.5 * np.cos(theta)
        y = 3.5 * np.sin(theta)
        inner_default.append([x, y, 0.0])
    
    required_side = compute_cluster_side_length(inner_default, outer_center, 0.0)
    return inner_default, (0.0, 0.0), required_side, 0.0


if __name__ == "__main__":
    result = construct()
    print(f"Best Side: {result[2]}")
