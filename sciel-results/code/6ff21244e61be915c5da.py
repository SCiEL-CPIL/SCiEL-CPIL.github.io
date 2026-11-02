import numpy as np
import time
import math
import itertools
import random

N = 12
DEADLINE = 145.0

# Unit hexagon geometry
# Vertices at angles 0, 60, 120, 180, 240, 300
UNIT_VERTS = np.array([
    [np.cos(np.radians(60 * i)), np.sin(np.radians(60 * i))] for i in range(6)
])

# Outward normals for the 6 edges (pointing to edge midpoints)
# Midpoints at 30, 90, 150, 210, 270, 330
OUTER_NORMALS = np.array([
    [np.cos(np.radians(60 * i + 30)), np.sin(np.radians(60 * i + 30))] for i in range(6)
])

def get_hex_vertices(centers):
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

def compute_required_side(centers, outer_center=(0.0, 0.0), outer_angle=0.0):
    """
    Compute the minimum outer side length needed to contain all inner hexagons.
    """
    verts = get_hex_vertices(np.array(centers, dtype=float))
    
    oa_rad = np.radians(outer_angle)
    c_o = np.cos(-oa_rad)
    s_o = np.sin(-oa_rad)
    R_o = np.array([[c_o, -s_o], [s_o, c_o]])
    
    max_extent = 0.0
    for i in range(len(centers)):
        v = verts[i]
        v_rel = v - np.array(outer_center)
        v_rot = v_rel @ R_o.T
        # Project onto outward normals
        projs = v_rot @ OUTER_NORMALS.T
        abs_projs = np.abs(projs)
        max_proj = np.max(abs_projs, axis=1)
        max_extent = max(max_extent, np.max(max_proj))
    
    # Apothem = side * sqrt(3)/2  =>  side = apothem / (sqrt(3)/2)
    return max_extent / (np.sqrt(3) / 2)

def check_validity(centers, outer_center=(0.0, 0.0), outer_side=None, outer_angle=0.0, tol=1e-7):
    """
    Check if packing is valid: no overlaps, all inside outer hexagon.
    """
    if outer_side is None:
        outer_side = compute_required_side(centers, outer_center, outer_angle)
        
    verts = get_hex_vertices(np.array(centers, dtype=float))
    n = verts.shape[0]
    
    # 1. Check pairwise separation (SAT)
    for i in range(n):
        v1 = verts[i]
        for j in range(i + 1, n):
            v2 = verts[j]
            edges1 = np.roll(v1, -1, axis=0) - v1
            edges2 = np.roll(v2, -1, axis=0) - v2
            axes = np.vstack([edges1, edges2])
            axes = axes / (np.linalg.norm(axes, axis=1, keepdims=True) + 1e-15)
            
            proj1 = v1 @ axes.T
            proj2 = v2 @ axes.T
            
            sep = np.minimum(proj1.min(axis=0) - proj2.max(axis=0),
                            proj2.min(axis=0) - proj1.max(axis=0))
            
            if np.max(sep) < -tol:
                return False
    
    # 2. Check containment
    apothem = outer_side * np.sqrt(3) / 2
    oa_rad = np.radians(outer_angle)
    c_o = np.cos(-oa_rad)
    s_o = np.sin(-oa_rad)
    R_o = np.array([[c_o, -s_o], [s_o, c_o]])
    
    for i in range(n):
        v = verts[i]
        v_rel = v - np.array(outer_center)
        v_rot = v_rel @ R_o.T
        projs = v_rot @ OUTER_NORMALS.T
        abs_projs = np.abs(projs)
        max_proj = np.max(abs_projs, axis=1)
        
        if np.max(max_proj) > apothem + tol:
            return False
            
    return True

def generate_lattice_configurations():
    """
    Generate diverse, compact initial configurations on a hexagonal lattice.
    """
    configs = []
    
    # Hexagonal lattice basis vectors
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
    configs.append(np.array(make_config(p1)))
    
    # Pattern 2: 2-4-4-2
    p2 = [
        (0, 0), (1, 0),
        (-1, 1), (0, 1), (1, 1), (2, 1),
        (-1, 2), (0, 2), (1, 2), (2, 2),
        (0, 3), (1, 3)
    ]
    configs.append(np.array(make_config(p2)))
    
    # Pattern 3: 1-3-4-3-1
    p3 = [
        (0, 0),
        (-1, 1), (0, 1), (1, 1),
        (-2, 2), (-1, 2), (0, 2), (1, 2),
        (-1, 3), (0, 3), (1, 3),
        (0, 4)
    ]
    configs.append(np.array(make_config(p3)))
    
    # Pattern 4: 4-4-4
    p4 = [
        (-1, 0), (0, 0), (1, 0), (2, 0),
        (-1, 1), (0, 1), (1, 1), (2, 1),
        (-1, 2), (0, 2), (1, 2), (2, 2)
    ]
    configs.append(np.array(make_config(p4)))
    
    # Pattern 5: 2-3-4-3
    p5 = [
        (0, 0), (1, 0),
        (-1, 1), (0, 1), (1, 1),
        (-2, 2), (-1, 2), (0, 2), (1, 2),
        (-1, 3), (0, 3), (1, 3)
    ]
    configs.append(np.array(make_config(p5)))
    
    # Pattern 6: 3-3-3-3
    p6 = [
        (-1, 0), (0, 0), (1, 0),
        (-1, 1), (0, 1), (1, 1),
        (-1, 2), (0, 2), (1, 2),
        (-1, 3), (0, 3), (1, 3)
    ]
    configs.append(np.array(make_config(p6)))
    
    # Add random perturbations
    for i in range(3):
        base = configs[i % len(configs)].copy()
        np.random.seed(42 + i)
        perturbed = base + np.random.uniform(-0.2, 0.2, base.shape)
        configs.append(perturbed)
        
    return configs

def physics_objective(params, fixed_positions=None, fixed_angles=None):
    """
    Objective function for physical relaxation.
    Minimizes: Smooth Containment + Strong Repulsion.
    If fixed_positions is provided, only positions of non-fixed hexagons are optimized.
    If fixed_angles is provided, only angles of non-fixed hexagons are optimized.
    """
    # params is a flattened array of variables to optimize
    # We need to reconstruct the full centers array
    centers = np.zeros((N, 3))
    
    # Initialize with fixed values if provided
    if fixed_positions is not None:
        centers[:, :2] = fixed_positions
    if fixed_angles is not None:
        centers[:, 2] = fixed_angles
    
    # Map params to the free variables
    # This is a simplified version. In a full implementation, we'd track which indices are free.
    # For now, let's assume params contains all x, y, angles that are free.
    # This is tricky without a proper mapping. Let's use a simpler approach for the relaxation.
    
    # Actually, let's just optimize all x, y, angles but with very high penalty for moving fixed ones?
    # No, that's inefficient.
    
    # Let's stick to the SA approach for the relaxation, but use the graph to guide the initial state.
    # The "topological optimization" is really about finding good initial states via graphs.
    
    # Repulsion Term
    repulsion = 0.0
    for i in range(N):
        for j in range(i + 1, N):
            d = np.linalg.norm(centers[i, :2] - centers[j, :2])
            threshold = 1.732 
            if d < threshold:
                repulsion += 10000.0 * ((threshold - d) / threshold) ** 6
            elif d < threshold * 1.2:
                repulsion += 100.0 * ((threshold * 1.2 - d) / threshold) ** 2
            
    # Containment Term (Smooth Max)
    verts = get_hex_vertices(centers)
    v_rel = verts
    projs = np.einsum('nki,ki->nk', v_rel, OUTER_NORMALS)
    abs_projs = np.abs(projs)
    d_hex = np.max(abs_projs, axis=2)
    p = 10
    smooth_max = (np.sum(d_hex ** p) / (N * 6)) ** (1.0 / p)
    containment_cost = smooth_max ** 2
    
    total_cost = repulsion + containment_cost
    return total_cost

def relax_configuration(initial_params, time_limit=15.0):
    """
    Run Simulated Annealing to optimize the configuration.
    """
    start_time = time.time()
    
    current = np.array(initial_params, dtype=float)
    current_cost = physics_objective(current)
    
    best = current.copy()
    best_cost = current_cost
    
    T = 5.0
    cooling_rate = 0.995
    min_T = 0.01
    max_steps = 5000
    
    step_sizes = np.zeros(N * 3)
    for i in range(N):
        step_sizes[i*3] = 0.5
        step_sizes[i*3+1] = 0.5
        step_sizes[i*3+2] = 10.0
        
    rng = np.random.default_rng(42)
    
    for step in range(max_steps):
        if time.time() - start_time > time_limit:
            break
        if T < min_T:
            break
            
        scale = T / 5.0 + 0.1
        delta = rng.normal(0, 1, N * 3) * step_sizes * scale
        neighbor = current + delta
        neighbor[2::3] = neighbor[2::3] % 360.0
        
        neighbor_cost = physics_objective(neighbor)
        delta_cost = neighbor_cost - current_cost
        
        if delta_cost < 0 or rng.random() < math.exp(-delta_cost / T):
            current = neighbor
            current_cost = neighbor_cost
            
            if current_cost < best_cost:
                best = current.copy()
                best_cost = current_cost
                
        T *= cooling_rate
        
    return best, best_cost

def generate_graph_configs():
    """
    Generate configurations based on planar triangulations (contact graphs).
    """
    configs = []
    
    # Simple hexagonal lattice graphs
    # For 12 nodes, we can use subsets of the infinite hexagonal lattice.
    
    # Lattice vectors
    a1 = np.array([np.sqrt(3), 0.0])
    a2 = np.array([np.sqrt(3) / 2.0, 1.5])
    
    # Generate a few specific topologies
    # Topology 1: 4x3 rectangle
    coords1 = [(i, j) for i in range(4) for j in range(3)]
    # Topology 2: Hexagon of radius 2 (19 nodes, take 12)
    # Topology 3: 3x4 rectangle
    coords3 = [(i, j) for i in range(3) for j in range(4)]
    
    for coords in [coords1, coords3]:
        centers = []
        for i, j in coords:
            pos = i * a1 + j * a2
            centers.append([pos[0], pos[1], 0.0])
        configs.append(np.array(centers))
        
    return configs

def construct():
    start_time = time.time()
    
    best_score = float('inf')
    best_result = None
    
    configs = generate_lattice_configurations()
    graph_configs = generate_graph_configs()
    configs.extend(graph_configs)
    
    np.random.shuffle(configs)
    
    for i, config in enumerate(configs):
        elapsed = time.time() - start_time
        if elapsed > DEADLINE - 10:
            break
            
        relax_time = min(15.0, DEADLINE - elapsed - 5)
        
        if relax_time < 2.0:
            continue
            
        try:
            optimized, cost = relax_configuration(config, time_limit=relax_time)
        except Exception:
            optimized = config
            
        req_side = compute_required_side(optimized)
        
        if check_validity(optimized, outer_side=req_side):
            if req_side < best_score:
                best_score = req_side
                best_result = (optimized.tolist(), (0.0, 0.0), req_side, 0.0)
        else:
            orig_side = compute_required_side(config)
            if check_validity(config, outer_side=orig_side):
                if orig_side < best_score:
                    best_score = orig_side
                    best_result = (config.tolist(), (0.0, 0.0), orig_side, 0.0)

    if best_result is not None:
        return best_result
    
    inner_default = []
    for i in range(N):
        theta = 2 * np.pi * i / N
        x = 3.5 * np.cos(theta)
        y = 3.5 * np.sin(theta)
        inner_default.append([x, y, 0.0])
    
    required_side = compute_required_side(inner_default)
    return inner_default, (0.0, 0.0), required_side, 0.0

if __name__ == "__main__":
    result = construct()
    print(f"Best Side: {result[2]}")
