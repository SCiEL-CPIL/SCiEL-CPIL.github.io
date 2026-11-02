"""
Strategy: Direct optimization of the outer hexagon side length using a
smooth 'soft' SAT separation penalty and a smooth containment penalty.
The objective is the outer side length S. We use a penalty method to
handle the non-overlap constraint, allowing the optimizer to explore
configurations where hexagons are very close (near-touching) without
being hard-rejected, which is crucial for finding tight packings.
We use L-BFGS-B which handles bounds and is efficient for this smooth
landscape.
"""
import numpy as np
import time
from scipy.optimize import minimize

N = 12
DEADLINE = 145.0

# Unit hexagon vertices (side length 1)
UNIT_VERTS = np.array([
    [np.cos(np.radians(60 * i)), np.sin(np.radians(60 * i))] for i in range(6)
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
    Returns the minimum distance (positive if separated, negative if overlapping).
    """
    edges1 = np.roll(v1, -1, axis=0) - v1
    edges2 = np.roll(v2, -1, axis=0) - v2
    axes = np.vstack([edges1, edges2])
    norms = np.linalg.norm(axes, axis=1, keepdims=True)
    axes = axes / (norms + 1e-15)
    
    proj1 = v1 @ axes.T
    proj2 = v2 @ axes.T
    
    sep = np.minimum(proj1.min(axis=0) - proj2.max(axis=0),
                    proj2.min(axis=0) - proj1.max(axis=0))
    
    # If max separation is negative, they overlap.
    # The value is the minimum separation along the axis of minimum separation.
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
        projs = v_rot @ np.array([
            [np.cos(np.radians(60 * i + 30)), np.sin(np.radians(60 * i + 30))] for i in range(6)
        ]).T
        abs_projs = np.abs(projs)
        max_proj = np.max(abs_projs, axis=1)
        max_extent = max(max_extent, np.max(max_proj))
    
    # Apothem = side * sqrt(3)/2  =>  side = apothem / (sqrt(3)/2)
    return max_extent / (np.sqrt(3) / 2)

def objective(params, outer_center=(0.0, 0.0), outer_angle=0.0):
    """
    Objective: Minimize outer side length S.
    Penalty: Add a penalty for overlaps (negative separation).
    """
    centers = np.array(params).reshape(N, 3)
    verts = get_hex_vertices(centers)
    
    # 1. Compute required side length
    S = compute_required_side(centers, outer_center, outer_angle)
    
    # 2. Compute overlap penalty
    # We want separation >= 0. If separation < 0, we penalize.
    # Use a smooth approximation: penalty = sum(max(0, -sep)^2)
    overlap_penalty = 0.0
    for i in range(N):
        v1 = verts[i]
        for j in range(i + 1, N):
            v2 = verts[j]
            sep = sat_min_separation(v1, v2)
            if sep < 0:
                # Quadratic penalty for overlap
                overlap_penalty += (-sep) ** 2
    
    # Scale penalty to be significant relative to S
    # S is around 3-4. Overlap sep is small.
    # Let's use a large multiplier.
    penalty_weight = 100.0
    total_cost = S + penalty_weight * overlap_penalty
    
    return total_cost

def generate_initial_configs():
    """
    Generate diverse initial configurations.
    """
    configs = []
    
    # Hexagonal lattice basis
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
    
    # Add random perturbations
    np.random.seed(42)
    for i in range(3):
        base = configs[i % len(configs)]
        perturbed = []
        for c in base:
            dx = np.random.uniform(-0.1, 0.1)
            dy = np.random.uniform(-0.1, 0.1)
            da = np.random.uniform(-5, 5)
            perturbed.append([c[0] + dx, c[1] + dy, c[2] + da])
        configs.append(perturbed)
        
    return configs

def construct():
    start_time = time.time()
    
    best_score = float('inf')
    best_result = None
    
    configs = generate_initial_configs()
    np.random.shuffle(configs)
    
    for i, config in enumerate(configs):
        elapsed = time.time() - start_time
        if elapsed > DEADLINE - 10:
            break
            
        x0 = np.array(config).flatten()
        
        # L-BFGS-B is good for bounded optimization
        # We don't have explicit bounds, but the optimizer will try to minimize.
        # We can add loose bounds if needed, but usually not necessary.
        
        def obj(x):
            if time.time() - start_time > DEADLINE - 5:
                return 1e10
            return objective(x)
            
        try:
            res = minimize(
                obj, x0, method='L-BFGS-B',
                options={'maxiter': 100, 'ftol': 1e-5, 'gtol': 1e-4, 'disp': False}
            )
            new_centers = res.x.reshape(N, 3)
        except Exception:
            new_centers = np.array(config)
            
        # Check validity of the result
        # Compute required side
        req_side = compute_required_side(new_centers)
        
        # Check for overlaps
        verts = get_hex_vertices(new_centers)
        valid = True
        for i in range(N):
            v1 = verts[i]
            for j in range(i + 1, N):
                v2 = verts[j]
                sep = sat_min_separation(v1, v2)
                if sep < -1e-5:  # Allow small numerical error
                    valid = False
                    break
            if not valid:
                break
                
        if valid:
            if req_side < best_score:
                best_score = req_side
                best_result = (new_centers.tolist(), (0.0, 0.0), req_side, 0.0)
        else:
            # If invalid, check if the original config was valid and better
            orig_side = compute_required_side(config)
            verts_orig = get_hex_vertices(np.array(config))
            valid_orig = True
            for i in range(N):
                v1 = verts_orig[i]
                for j in range(i + 1, N):
                    v2 = verts_orig[j]
                    sep = sat_min_separation(v1, v2)
                    if sep < -1e-5:
                        valid_orig = False
                        break
                if not valid_orig:
                    break
                    
            if valid_orig and orig_side < best_score:
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
    
    required_side = compute_required_side(inner_default)
    return inner_default, (0.0, 0.0), required_side, 0.0

if __name__ == "__main__":
    result = construct()
    print(f"Best Side: {result[2]}")
