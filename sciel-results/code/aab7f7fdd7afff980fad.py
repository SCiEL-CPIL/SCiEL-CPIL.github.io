"""
Strategy: 6-fold Symmetric Packing with Differential Evolution.
Instead of optimizing 36 independent Cartesian coordinates, we exploit the 
6-fold rotational symmetry of the outer hexagon. We define the configuration 
by a set of "fundamental" hexagons and generate the rest via rotation. 
This reduces the search space significantly and ensures the packing is 
symmetric, which is often optimal for hexagonal containers. 
We use Differential Evolution (DE) for robust global optimization of the 
radial/angular parameters.
"""
import numpy as np
import time
from scipy.optimize import differential_evolution

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
    
    return max_extent / (np.sqrt(3) / 2)

def generate_symmetric_config(params):
    """
    params: array of shape (K, 3) where K is number of fundamental hexagons.
    Each fundamental hexagon is defined by (r, theta, alpha).
    We generate 6 copies rotated by 60 degrees each.
    To get 12 hexagons, we need K=2 fundamental hexagons.
    """
    K = len(params)
    centers = []
    for k in range(K):
        r, theta, alpha = params[k]
        base_angle_rad = np.radians(theta)
        base_pos = np.array([r * np.cos(base_angle_rad), r * np.sin(base_angle_rad)])
        base_rot = alpha
        
        for i in range(6):
            rot_angle = np.radians(60 * i)
            c = np.cos(rot_angle)
            s = np.sin(rot_angle)
            R = np.array([[c, -s], [s, c]])
            
            pos = R @ base_pos
            # The rotation of the hexagon itself also changes
            # If the hexagon is rotated by 60*i, its local frame is also rotated.
            # So the local angle adds to the global rotation.
            new_alpha = base_rot + 60 * i
            
            centers.append([pos[0], pos[1], new_alpha])
            
    return np.array(centers)

def objective_sym(params):
    """
    Objective function for symmetric configuration.
    params: flattened array of K*3 parameters.
    """
    if time.time() > DEADLINE - 5:
        return 1e10
        
    K = N // 6
    params_reshaped = params.reshape(K, 3)
    
    centers = generate_symmetric_config(params_reshaped)
    
    # Check validity and compute side
    S = compute_required_side(centers)
    
    # Check overlaps
    verts = get_hex_vertices(centers)
    overlap_penalty = 0.0
    for i in range(N):
        v1 = verts[i]
        for j in range(i + 1, N):
            v2 = verts[j]
            sep = sat_min_separation(v1, v2)
            if sep < 0:
                overlap_penalty += (-sep) ** 2
                
    penalty_weight = 100.0
    return S + penalty_weight * overlap_penalty

def construct():
    start_time = time.time()
    
    best_score = float('inf')
    best_result = None
    
    # K = 2 fundamental hexagons
    K = 2
    # Bounds for (r, theta, alpha)
    # r: 0 to 10 (generous bound)
    # theta: 0 to 60 (symmetry sector)
    # alpha: 0 to 360 (rotation)
    bounds = [
        (0.5, 10.0),  # r
        (0.0, 60.0),  # theta
        (0.0, 360.0), # alpha
        (0.5, 10.0),  # r
        (0.0, 60.0),  # theta
        (0.0, 360.0), # alpha
    ]
    
    # Initial guess
    # Try a hexagonal lattice-like start
    # Center at 0,0 is bad for K=2 symmetric unless we place them carefully.
    # Let's place one near center, one further out? 
    # Actually, with K=2, we have 12 hexagons.
    # A good start might be r1=0, r2=2*sqrt(3) approx?
    # Let's just let DE explore.
    
    try:
        # Use Differential Evolution
        # maxiter and popsize tuned for time budget
        # 120s limit. DE can be slow.
        # Let's limit maxiter.
        
        # We will run DE for a portion of the time, then refine with L-BFGS-B if time permits.
        
        # First, run DE
        # popsize=10, maxiter=50 might be enough for 6D space
        # Each eval takes ~5ms (approx). 10*50*6 = 3000 evals ~ 15s.
        
        res = differential_evolution(
            objective_sym, bounds, 
            seed=42, 
            maxiter=100, 
            popsize=15,
            tol=1e-5,
            polish=False  # We will do our own polish
        )
        
        if time.time() > DEADLINE - 10:
            # Just use the DE result if it's valid
            centers = generate_symmetric_config(res.x.reshape(K, 3))
            req_side = compute_required_side(centers)
            verts = get_hex_vertices(centers)
            valid = True
            for i in range(N):
                v1 = verts[i]
                for j in range(i + 1, N):
                    v2 = verts[j]
                    sep = sat_min_separation(v1, v2)
                    if sep < -1e-5:
                        valid = False
                        break
                if not valid:
                    break
            if valid and req_side < best_score:
                best_score = req_side
                best_result = (centers.tolist(), (0.0, 0.0), req_side, 0.0)
            return best_result if best_result else _fallback()
        
        # Refine with L-BFGS-B on the symmetric parameters
        from scipy.optimize import minimize
        
        def obj_refine(x):
            if time.time() > DEADLINE - 5:
                return 1e10
            return objective_sym(x)
            
        res_refine = minimize(
            obj_refine, res.x, method='L-BFGS-B',
            options={'maxiter': 50, 'ftol': 1e-8, 'gtol': 1e-6, 'disp': False}
        )
        
        # Check refined result
        centers_ref = generate_symmetric_config(res_refine.x.reshape(K, 3))
        req_side_ref = compute_required_side(centers_ref)
        verts_ref = get_hex_vertices(centers_ref)
        valid_ref = True
        for i in range(N):
            v1 = verts_ref[i]
            for j in range(i + 1, N):
                v2 = verts_ref[j]
                sep = sat_min_separation(v1, v2)
                if sep < -1e-5:
                    valid_ref = False
                    break
                if not valid_ref:
                    break
        if valid_ref and req_side_ref < best_score:
            best_score = req_side_ref
            best_result = (centers_ref.tolist(), (0.0, 0.0), req_side_ref, 0.0)
            
        # Also check DE result just in case
        centers_de = generate_symmetric_config(res.x.reshape(K, 3))
        req_side_de = compute_required_side(centers_de)
        verts_de = get_hex_vertices(centers_de)
        valid_de = True
        for i in range(N):
            v1 = verts_de[i]
            for j in range(i + 1, N):
                v2 = verts_de[j]
                sep = sat_min_separation(v1, v2)
                if sep < -1e-5:
                    valid_de = False
                    break
            if not valid_de:
                break
        if valid_de and req_side_de < best_score:
            best_score = req_side_de
            best_result = (centers_de.tolist(), (0.0, 0.0), req_side_de, 0.0)

    except Exception as e:
        pass

    if best_result is not None:
        return best_result
    
    return _fallback()

def _fallback():
    # Simple circular arrangement
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
