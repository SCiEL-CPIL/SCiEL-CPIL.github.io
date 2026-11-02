"""
Packing 12 unit regular hexagons inside a regular hexagon, minimizing the outer side length.
Strategy:
1. Generate diverse initial configurations from hexagonal lattice patterns with rotations.
2. Use a differentiable objective function that penalizes overlaps and outer size.
3. Optimize using L-BFGS-B with careful time management.
4. Validate final configuration and return the best found.
"""
import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial import ConvexHull

N = 12
DEADLINE = 140.0

# Unit hexagon vertices (pointy-top: vertices at 30, 90, 150, ...)
# Using flat-top orientation: vertices at 0, 60, 120, ...
UNIT_VERTS = np.array([
    [np.cos(np.radians(60 * i)), np.sin(np.radians(60 * i))] for i in range(6)
])

def get_hex_vertices(params):
    """
    params: (N, 3) array of [x, y, angle_deg]
    Returns: (N, 6, 2) array of all vertices
    """
    n = len(params)
    all_verts = np.zeros((n, 6, 2))
    for i in range(n):
        x, y, a_deg = params[i]
        a_rad = np.radians(a_deg)
        c, s = np.cos(a_rad), np.sin(a_rad)
        R = np.array([[c, -s], [s, c]])
        all_verts[i] = UNIT_VERTS @ R.T + np.array([x, y])
    return all_verts

def compute_outer_side(all_verts, center=(0.0, 0.0), angle=0.0):
    """
    Compute the minimum side length of a regular hexagon centered at `center`
    with rotation `angle` that contains all vertices in `all_verts`.
    """
    oa_rad = np.radians(angle)
    c_o, s_o = np.cos(-oa_rad), np.sin(-oa_rad)
    R_o = np.array([[c_o, -s_o], [s_o, c_o]])
    
    v_rel = all_verts.reshape(-1, 2) - np.array(center)
    v_rot = v_rel @ R_o.T
    
    # Outward normals for the outer hexagon edges
    normals = np.array([
        [np.cos(np.radians(60 * i + 30)), np.sin(np.radians(60 * i + 30))] for i in range(6)
    ])
    
    projs = v_rot @ normals.T
    max_extent = np.max(np.abs(projs))
    
    # side = apothem / (sqrt(3)/2)
    return max_extent / (np.sqrt(3) / 2)

def penalty_overlap(verts_i, verts_j):
    """
    Compute a smooth penalty for overlap between two convex hexagons.
    Uses the maximum penetration depth approximation.
    """
    # Edge normals for hexagon i
    edges_i = np.roll(verts_i, -1, axis=0) - verts_i
    norms_i = np.linalg.norm(edges_i, axis=1, keepdims=True) + 1e-15
    normals_i = edges_i / norms_i
    # Flip normals to point outward
    normals_i = -normals_i
    
    # Edge normals for hexagon j
    edges_j = np.roll(verts_j, -1, axis=0) - verts_j
    norms_j = np.linalg.norm(edges_j, axis=1, keepdims=True) + 1e-15
    normals_j = edges_j / norms_j
    normals_j = -normals_j
    
    all_normals = np.vstack([normals_i, normals_j])
    
    proj_i = verts_i @ all_normals.T
    proj_j = verts_j @ all_normals.T
    
    # For each axis, check if intervals overlap
    # Overlap = min(max_i, max_j) - max(min_i, min_j)
    overlap = np.minimum(proj_i.max(axis=0), proj_j.max(axis=0)) - \
              np.maximum(proj_i.min(axis=0), proj_j.min(axis=0))
    
    # Return the maximum overlap (penetration depth)
    return np.max(overlap)

def differentiable_objective(params):
    """
    Objective: minimize outer side length + penalty for overlaps.
    """
    all_verts = get_hex_vertices(params)
    
    # Outer side
    side = compute_outer_side(all_verts, (0.0, 0.0), 0.0)
    
    # Overlap penalty
    penalty = 0.0
    n = len(params)
    for i in range(n):
        for j in range(i + 1, n):
            pen = penalty_overlap(all_verts[i], all_verts[j])
            if pen > 0:
                penalty += pen ** 2
    
    # Regularization to keep centers close to origin
    centers = params[:, :2]
    spread = np.mean(np.sum(centers ** 2, axis=1))
    
    return side + 100.0 * penalty + 0.01 * spread

def check_no_overlap(params, tol=1e-7):
    """Check that no two inner hexagons overlap using SAT."""
    all_verts = get_hex_vertices(params)
    n = len(params)
    for i in range(n):
        verts_i = all_verts[i]
        for j in range(i + 1, n):
            verts_j = all_verts[j]
            # SAT check
            edges_i = np.roll(verts_i, -1, axis=0) - verts_i
            edges_j = np.roll(verts_j, -1, axis=0) - verts_j
            
            axes = np.vstack([edges_i, edges_j])
            norms = np.linalg.norm(axes, axis=1, keepdims=True) + 1e-15
            axes = axes / norms
            
            proj_i = verts_i @ axes.T
            proj_j = verts_j @ axes.T
            
            sep = np.minimum(
                proj_i.min(axis=0) - proj_j.max(axis=0),
                proj_j.min(axis=0) - proj_i.max(axis=0)
            )
            
            if np.max(sep) < -tol:
                return False
    return True

def generate_initial_configs():
    """Generate diverse initial configurations from hexagonal lattice patterns."""
    configs = []
    
    a1 = np.array([np.sqrt(3), 0.0])
    a2 = np.array([np.sqrt(3) / 2.0, 1.5])
    
    def make_config(coords, angle=0.0):
        centers = []
        for i, j in coords:
            pos = i * a1 + j * a2
            centers.append([pos[0], pos[1], angle])
        return np.array(centers)
    
    # Various hexagonal lattice patterns for 12 hexagons
    patterns = [
        # 3-4-3-2
        [(0, 0), (1, 0), (2, 0), (-1, 1), (0, 1), (1, 1), (2, 1), (0, 2), (1, 2), (2, 2), (0, 3), (1, 3)],
        # 2-4-4-2
        [(0, 0), (1, 0), (-1, 1), (0, 1), (1, 1), (2, 1), (-1, 2), (0, 2), (1, 2), (2, 2), (0, 3), (1, 3)],
        # 1-3-4-3-1
        [(0, 0), (-1, 1), (0, 1), (1, 1), (-2, 2), (-1, 2), (0, 2), (1, 2), (-1, 3), (0, 3), (1, 3), (0, 4)],
        # 4-4-4
        [(-1, 0), (0, 0), (1, 0), (2, 0), (-1, 1), (0, 1), (1, 1), (2, 1), (-1, 2), (0, 2), (1, 2), (2, 2)],
        # 3-3-3-3
        [(-1, 0), (0, 0), (1, 0), (-1, 1), (0, 1), (1, 1), (-1, 2), (0, 2), (1, 2), (-1, 3), (0, 3), (1, 3)],
        # 2-3-4-3
        [(0, 0), (1, 0), (-1, 1), (0, 1), (1, 1), (-2, 2), (-1, 2), (0, 2), (1, 2), (-1, 3), (0, 3), (1, 3)],
        # 6-6 two rows
        [(-1, 0), (0, 0), (1, 0), (-1, 1), (0, 1), (1, 1), (-2, 2), (-1, 2), (0, 2), (-1, 3), (0, 3), (1, 3)],
        # Centered cluster
        [(0, 0), (1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (-1, -1), (1, -1), (-1, 1), (2, 0), (0, 2), (0, -2)],
    ]
    
    for p in patterns:
        configs.append(make_config(p))
    
    # Add rotated versions
    for p in patterns[:4]:
        configs.append(make_config(p, angle=30.0))
        configs.append(make_config(p, angle=60.0))
    
    # Random perturbations
    np.random.seed(42)
    for _ in range(8):
        base = configs[np.random.randint(len(configs))]
        perturbed = base + np.random.uniform(-0.3, 0.3, size=(N, 3))
        configs.append(perturbed)
    
    return configs

def construct():
    start_time = time.time()
    
    best_side = float('inf')
    best_params = None
    
    configs = generate_initial_configs()
    np.random.shuffle(configs)
    
    for i, config in enumerate(configs):
        elapsed = time.time() - start_time
        if elapsed > DEADLINE - 15:
            break
        
        remaining = DEADLINE - elapsed - 5
        budget_per = max(2.0, remaining / max(1, len(configs) - i))
        budget = min(10.0, budget_per)
        
        if budget < 1.0:
            continue
        
        def obj(x):
            if time.time() - start_time > DEADLINE - 10:
                return 1e10
            try:
                return differentiable_objective(x)
            except Exception:
                return 1e10
        
        try:
            res = minimize(
                obj, config, method='L-BFGS-B',
                options={'maxiter': 100, 'gtol': 1e-6, 'disp': False}
            )
            optimized = res.x
            
            # Check validity
            if check_no_overlap(optimized):
                all_verts = get_hex_vertices(optimized)
                side = compute_outer_side(all_verts, (0.0, 0.0), 0.0)
                if side < best_side:
                    best_side = side
                    best_params = optimized.copy()
        except Exception:
            pass
    
    # Final result
    if best_params is not None:
        inner = [[float(p[0]), float(p[1]), float(p[2])] for p in best_params]
        return inner, (0.0, 0.0), float(best_side), 0.0
    
    # Fallback: hexagonal lattice
    a1 = np.array([np.sqrt(3), 0.0])
    a2 = np.array([np.sqrt(3) / 2.0, 1.5])
    coords = [(0, 0), (1, 0), (2, 0), (-1, 1), (0, 1), (1, 1), (2, 1), (0, 2), (1, 2), (2, 2), (0, 3), (1, 3)]
    inner = []
    for i, j in coords:
        pos = i * a1 + j * a2
        inner.append([float(pos[0]), float(pos[1]), 0.0])
    
    side = compute_outer_side(get_hex_vertices(np.array(inner)), (0.0, 0.0), 0.0)
    return inner, (0.0, 0.0), float(side), 0.0

if __name__ == "__main__":
    result = construct()
    print(f"Best Side: {result[2]}")
