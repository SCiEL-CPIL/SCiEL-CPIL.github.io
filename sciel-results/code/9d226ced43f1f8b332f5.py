import numpy as np
import time
import math
from scipy.optimize import minimize

N = 12
DEADLINE = 145.0  # Safety margin for 150s limit

# Precompute unit hexagon vertices (flat-top orientation)
# Vertices at angles 0, 60, 120, 180, 240, 300
UNIT_VERTS = np.array([
    [np.cos(np.radians(60 * i)), np.sin(np.radians(60 * i))] for i in range(6)
])

# Outward normals for the 6 edges of a regular hexagon (pointing to midpoints)
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

def sat_min_separation(v1, v2):
    """
    Compute minimum separation between two convex polygons using SAT.
    Returns negative value if overlapping.
    """
    edges1 = np.roll(v1, -1, axis=0) - v1
    edges2 = np.roll(v2, -1, axis=0) - v2
    axes = np.vstack([edges1, edges2])
    axes = axes / (np.linalg.norm(axes, axis=1, keepdims=True) + 1e-15)
    
    proj1 = v1 @ axes.T
    proj2 = v2 @ axes.T
    
    sep = np.minimum(proj1.min(axis=0) - proj2.max(axis=0),
                    proj2.min(axis=0) - proj1.max(axis=0))
    
    return np.max(sep)

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

def check_validity(centers, outer_side=None, outer_center=(0.0, 0.0), outer_angle=0.0, tol=1e-7):
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
            sep = sat_min_separation(v1, v2)
            if sep < -tol:
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

def objective(params):
    """
    Objective function for Simulated Annealing.
    Minimizes: outer_side + overlap_penalty
    """
    centers = np.array(params).reshape(N, 3)
    verts = get_hex_vertices(centers)
    n = verts.shape[0]
    
    # 1. Overlap Penalty
    min_sep = float('inf')
    for i in range(n):
        for j in range(i + 1, n):
            sep = sat_min_separation(verts[i], verts[j])
            if sep < min_sep:
                min_sep = sep
    
    overlap_penalty = 0.0
    if min_sep < 0:
        overlap_penalty = -min_sep * 1000.0
    
    # 2. Containment (Outer Side Length)
    req_side = compute_required_side(centers)
    
    return req_side + overlap_penalty

def generate_initial_config(seed):
    """
    Generate a diverse initial configuration.
    We use a mix of lattice-based and random starts.
    """
    np.random.seed(seed)
    
    # Option 1: Hexagonal lattice with random jitter
    # Centers in a hex grid
    a1 = np.array([np.sqrt(3), 0.0])
    a2 = np.array([np.sqrt(3)/2, 1.5])
    
    # Pick 12 positions from a larger grid
    positions = []
    for i in range(-3, 4):
        for j in range(-3, 4):
            pos = i * a1 + j * a2
            positions.append(pos)
            
    # Shuffle and take first 12
    idx = np.random.choice(len(positions), 12, replace=False)
    centers = []
    for i in idx:
        pos = positions[i]
        angle = np.random.uniform(0, 60) # Random rotation
        centers.append([pos[0], pos[1], angle])
        
    # Center of mass correction
    com = np.mean([c[:2] for c in centers], axis=0)
    centered = []
    for c in centers:
        centered.append([c[0] - com[0], c[1] - com[1], c[2]])
        
    return centered

def simulated_annealing(initial_params, time_limit=10.0):
    """
    Perform Simulated Annealing to optimize the configuration.
    """
    start_time = time.time()
    x = np.array(initial_params).flatten()
    current_cost = objective(x)
    best_x = x.copy()
    best_cost = current_cost
    
    # SA Parameters
    T_start = 10.0
    T_end = 0.01
    alpha = 0.999 # Cooling rate per step
    
    # Estimate number of steps based on time
    # Assume ~1ms per evaluation (optimistic, actually might be 5-10ms)
    # Let's do a fixed number of steps and check time
    max_steps = 5000
    
    for step in range(max_steps):
        if time.time() - start_time > time_limit:
            break
            
        # Generate neighbor
        # Perturb positions and angles
        dx = np.random.normal(0, 0.5, N)
        dy = np.random.normal(0, 0.5, N)
        da = np.random.uniform(-15, 15, N)
        
        x_new = x.copy()
        x_new[0::3] += dx
        x_new[1::3] += dy
        x_new[2::3] += da
        
        # Keep angles bounded (optional, but helps with periodicity if we cared)
        # x_new[2::3] = np.mod(x_new[2::3], 60)
        
        new_cost = objective(x_new)
        
        # Acceptance probability
        if new_cost < current_cost:
            x = x_new
            current_cost = new_cost
            if new_cost < best_cost:
                best_cost = new_cost
                best_x = x_new.copy()
        else:
            # Boltzmann acceptance
            delta = new_cost - current_cost
            T = T_start * (T_end / T_start) ** (step / max_steps)
            if T > 0:
                prob = np.exp(-delta / T)
                if np.random.random() < prob:
                    x = x_new
                    current_cost = new_cost
        
    return best_x.reshape(N, 3)

def construct():
    start_time = time.time()
    
    best_score = float('inf')
    best_result = None
    
    # Multiple restarts with different seeds
    num_restarts = 20
    
    for i in range(num_restarts):
        elapsed = time.time() - start_time
        if elapsed > DEADLINE - 5:
            break
            
        # Time budget for this restart
        # Leave 5s for final validation and return
        remaining = DEADLINE - elapsed - 5
        if remaining < 2.0:
            break
            
        # Allocate time: 80% SA, 20% buffer
        sa_time = remaining * 0.8
        
        # Generate initial config
        init_config = generate_initial_config(seed=1000 + i)
        
        # Run Simulated Annealing
        try:
            optimized = simulated_annealing(init_config, time_limit=sa_time)
        except Exception:
            optimized = np.array(init_config)
            
        # Validate and update best
        req_side = compute_required_side(optimized)
        
        if check_validity(optimized, outer_side=req_side):
            if req_side < best_score:
                best_score = req_side
                best_result = (optimized.tolist(), (0.0, 0.0), req_side, 0.0)
                
        # Also check if the initial config was better (unlikely but possible)
        orig_side = compute_required_side(init_config)
        if check_validity(init_config, outer_side=orig_side):
            if orig_side < best_score:
                best_score = orig_side
                best_result = (init_config, (0.0, 0.0), orig_side, 0.0)

    if best_result is not None:
        return best_result
    
    # Fallback: Simple circular arrangement
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
