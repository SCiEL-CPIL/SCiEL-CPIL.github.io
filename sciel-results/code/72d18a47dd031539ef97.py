"""Improved program using vectorized smooth surrogate + L-BFGS-B optimization."""
import numpy as np
import time
from scipy.optimize import minimize

N = 12
DEADLINE = 145.0

# Precompute unit hexagon vertices (side length 1, angle 0)
UNIT_VERTS = np.array([
    [np.cos(np.radians(60*i)), np.sin(np.radians(60*i))] for i in range(6)
])
OUTER_NORMALS = np.array([
    [np.cos(np.radians(60*i)), np.sin(np.radians(60*i))] for i in range(6)
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

def smooth_min(values, beta=50.0):
    """Soft-min using log-sum-exp."""
    values = np.array(values)
    if len(values) == 0:
        return 0.0
    m = np.min(values)
    if m < -100:
        return m
    shifted = values - m
    exp_vals = np.exp(-beta * shifted)
    return m - np.log(np.sum(exp_vals)) / beta

def smooth_max(values, beta=50.0):
    """Soft-max using log-sum-exp."""
    return -smooth_min(-values, beta)

def compute_clearances_fast_vec(verts, outer_center, outer_angle, outer_side):
    """
    verts: (N, 6, 2)
    Returns min_pair_sep, min_outer_clearance
    """
    n = verts.shape[0]
    min_pair = float('inf')
    
    # Precompute edge normals for all hexagons
    # edges: (N, 6, 2)
    edges = np.roll(verts, -1, axis=1) - verts
    norms = np.linalg.norm(edges, axis=2) # (N, 6)
    norms[norms < 1e-12] = 1.0
    # normals: (N, 6, 2)
    normals = np.stack((-edges[:, :, 1], edges[:, :, 0]), axis=2) / norms[:, :, np.newaxis]

    for i in range(n):
        v1 = verts[i]
        n1 = normals[i]
        
        for j in range(i+1, n):
            v2 = verts[j]
            n2 = normals[j]
            
            axes = np.vstack([n1, n2])
            
            proj1 = v1 @ axes.T
            proj2 = v2 @ axes.T
            
            sep = np.maximum(proj1.min(axis=0) - proj2.max(axis=0), 
                             proj2.min(axis=0) - proj1.max(axis=0))
            
            min_sep = np.min(sep)
            if min_sep < min_pair:
                min_pair = min_sep
                
    # Outer Clearance
    apothem = outer_side * np.sqrt(3) / 2
    
    oa_rad = np.radians(outer_angle)
    c_o = np.cos(-oa_rad)
    s_o = np.sin(-oa_rad)
    R_o = np.array([[c_o, -s_o], [s_o, c_o]])
    
    min_outer = float('inf')
    
    for i in range(n):
        v = verts[i]
        v_rel = v - outer_center
        v_rot = v_rel @ R_o.T
        projs = v_rot @ OUTER_NORMALS.T
        abs_projs = np.abs(projs)
        max_proj = np.max(abs_projs, axis=1)
        
        clearances = apothem - max_proj
        min_c = np.min(clearances)
        if min_c < min_outer:
            min_outer = min_c
            
    return min_pair, min_outer

def required_outer_side(centers, outer_center, outer_angle):
    """Compute the minimum outer side length needed to contain all inner hexagons."""
    verts = get_vertices(np.array(centers))
    
    oa_rad = np.radians(outer_angle)
    c_o = np.cos(-oa_rad)
    s_o = np.sin(-oa_rad)
    R_o = np.array([[c_o, -s_o], [s_o, c_o]])
    
    max_extent = 0.0
    for i in range(len(centers)):
        v = verts[i]
        v_rel = v - outer_center
        v_rot = v_rel @ R_o.T
        projs = v_rot @ OUTER_NORMALS.T
        abs_projs = np.abs(projs)
        max_proj = np.max(abs_projs, axis=1)
        max_extent = max(max_extent, np.max(max_proj))
    
    return max_extent / (np.sqrt(3) / 2)

def check_validity(centers, outer_center, outer_side, outer_angle):
    """Check if packing is valid: all inner hexagons inside outer, no overlaps."""
    verts = get_vertices(np.array(centers))
    min_pair, min_outer = compute_clearances_fast_vec(verts, outer_center, outer_angle, outer_side)
    return min_pair >= -1e-9 and min_outer >= -1e-9

def initial_config_hexagonal():
    """Create a hexagonal lattice initial configuration."""
    centers = []
    dx = 2.0 * np.sqrt(3)
    dy = 3.0
    
    positions = [
        (0, 0),
        (dx, 0), (-dx, 0),
        (dx/2, dy*0.866), (-dx/2, dy*0.866),
        (dx/2, -dy*0.866), (-dx/2, -dy*0.866),
        (3*dx/2, dy*0.866), (-3*dx/2, dy*0.866),
        (3*dx/2, -dy*0.866), (-3*dx/2, -dy*0.866),
        (0, dy*1.732), (0, -dy*1.732)
    ]
    
    for (x, y) in positions:
        centers.append([x, y, 0.0])
    
    return centers

def initial_config_circular():
    """Create a circular arrangement."""
    centers = []
    for i in range(N):
        theta = 2 * np.pi * i / N
        r = 2.5
        x = r * np.cos(theta)
        y = r * np.sin(theta)
        a = np.degrees(theta)
        centers.append([x, y, a])
    return centers

def initial_config_random(seed=42):
    """Create a random initial configuration."""
    np.random.seed(seed)
    centers = []
    for i in range(N):
        x = np.random.uniform(-4, 4)
        y = np.random.uniform(-4, 4)
        a = np.random.uniform(0, 60)
        centers.append([x, y, a])
    return centers

def initial_config_staggered():
    """Staggered grid arrangement."""
    centers = []
    dx = 2.0 * np.sqrt(3)
    dy = 3.0
    # 4x3 grid with stagger
    for row in range(3):
        for col in range(4):
            x = col * dx
            y = row * dy
            if row % 2 == 1:
                x += dx / 2
            x -= 1.5 * dx
            y -= dy
            centers.append([x, y, 0.0])
    return centers

def smooth_objective_compact(params, outer_center, outer_angle, beta=30.0):
    """
    Objective that minimizes the 'smooth' outer side length and overlaps.
    """
    n = N
    centers = np.zeros((n, 3))
    for i in range(n):
        centers[i] = [params[3*i], params[3*i+1], params[3*i+2]]
    
    verts = get_vertices(centers)
    
    # Calculate smooth outer side
    oa_rad = np.radians(outer_angle)
    c_o = np.cos(-oa_rad)
    s_o = np.sin(-oa_rad)
    R_o = np.array([[c_o, -s_o], [s_o, c_o]])
    
    extents = []
    for i in range(n):
        v = verts[i]
        v_rel = v - outer_center
        v_rot = v_rel @ R_o.T
        projs = v_rot @ OUTER_NORMALS.T
        abs_projs = np.abs(projs)
        max_proj_per_vertex = np.max(abs_projs, axis=1)
        extents.extend(max_proj_per_vertex)
        
    extents = np.array(extents)
    smooth_apothem = smooth_max(extents, beta)
    smooth_side = smooth_apothem / (np.sqrt(3) / 2)
    
    # Calculate overlaps
    min_pair, min_outer = compute_clearances_fast_vec(verts, outer_center, outer_angle, smooth_side)
    
    # Penalty for negative clearances
    penalty = 0.0
    if min_pair < 0:
        penalty += min_pair ** 2 * 100.0
    if min_outer < 0:
        penalty += min_outer ** 2 * 100.0
        
    return smooth_side + penalty

def lbfghs_polish(centers, outer_center, outer_angle, outer_side, time_limit=20.0):
    """
    Polish a configuration using L-BFGS-B with the smooth objective.
    Optimizes inner hexagon positions and angles.
    """
    start_time = time.time()
    
    # Flatten parameters
    params = np.zeros(3 * N)
    for i in range(N):
        params[3*i] = centers[i][0]
        params[3*i+1] = centers[i][1]
        params[3*i+2] = centers[i][2]
    
    def obj(p):
        if time.time() - start_time > time_limit:
            return 1e6
        return smooth_objective_compact(p, outer_center, outer_angle)
    
    def grad(p):
        if time.time() - start_time > time_limit:
            return np.zeros_like(p)
        eps = 1e-5
        g = np.zeros_like(p)
        f0 = obj(p)
        for i in range(len(p)):
            p_p = p.copy()
            p_p[i] += eps
            fp = obj(p_p)
            g[i] = (fp - f0) / eps
        return g
    
    result = minimize(
        obj, params, jac=grad, method='L-BFGS-B',
        options={'maxiter': 500, 'maxfun': 10000, 'ftol': 1e-12, 'gtol': 1e-8}
    )
    
    new_centers = []
    for i in range(N):
        new_centers.append([result.x[3*i], result.x[3*i+1], result.x[3*i+2]])
    
    return new_centers

def construct():
    start_time = time.time()
    
    best_result = None
    best_score = float('inf')
    
    configs = [
        ("hexagonal", initial_config_hexagonal()),
        ("circular", initial_config_circular()),
        ("random1", initial_config_random(42)),
        ("random2", initial_config_random(123)),
        ("staggered", initial_config_staggered()),
        ("random3", initial_config_random(999)),
        ("random4", initial_config_random(1000)),
    ]
    
    for name, config in configs:
        elapsed = time.time() - start_time
        if elapsed > DEADLINE - 10:
            break
        
        # Use L-BFGS-B directly with the compacting objective
        polish_time = min(20.0, DEADLINE - elapsed - 5)
        
        try:
            polished = lbfghs_polish(config, (0.0, 0.0), 0.0, 10.0, time_limit=polish_time)
        except Exception:
            polished = config
        
        elapsed = time.time() - start_time
        if elapsed > DEADLINE - 5:
            break
        
        # Compute final required side
        final_side = required_outer_side(polished, (0.0, 0.0), 0.0)
        
        # Validate
        if check_validity(polished, (0.0, 0.0), final_side, 0.0):
            if final_side < best_score:
                best_score = final_side
                best_result = (polished, (0.0, 0.0), final_side, 0.0)
            
    if best_result is not None:
        return best_result
    
    # Fallback
    inner_default = []
    for i in range(N):
        theta = 2 * np.pi * i / N
        x = 3.5 * np.cos(theta)
        y = 3.5 * np.sin(theta)
        inner_default.append([x, y, 0.0])
    
    required_side = required_outer_side(inner_default, (0.0, 0.0), 0.0)
    return inner_default, (0.0, 0.0), required_side, 0.0
