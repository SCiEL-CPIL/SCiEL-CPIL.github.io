import numpy as np
import time
from scipy.optimize import minimize

N = 12
DEADLINE = 145.0

# Unit hexagon vertices (side length 1)
# Vertices at angles 0, 60, 120, 180, 240, 300 degrees
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
        # Project onto outward normals of the regular hexagon
        # Normals are at 30, 90, 150, 210, 270, 330 degrees
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
    penalty_weight = 100.0
    total_cost = S + penalty_weight * overlap_penalty
    
    return total_cost

def generate_structural_config(r1, r2, offset_deg, global_rot_deg):
    """
    Generate a configuration based on 1+6+5 symmetry.
    r1: radius of inner ring (6 hexagons)
    r2: radius of outer ring (5 hexagons)
    offset_deg: angular offset of outer ring relative to inner ring
    global_rot_deg: global rotation of the entire structure
    """
    centers = []
    
    # 1. Center hexagon
    # Rotate to align with the "hole" of the outer ring?
    # Usually center is aligned with the lattice.
    # Let's assume center is at (0,0) and rotated by global_rot.
    centers.append([0.0, 0.0, global_rot_deg])
    
    # 2. Inner ring (6 hexagons)
    # Angles: 0, 60, 120, 180, 240, 300 + global_rot
    for i in range(6):
        theta = np.radians(60 * i + global_rot_deg)
        x = r1 * np.cos(theta)
        y = r1 * np.sin(theta)
        # Rotation of each hexagon: align "flat" side with the radius?
        # In a perfect honeycomb, the hexagon at angle theta is rotated by theta + 30?
        # Let's try rotating by theta + global_rot_deg to maintain local symmetry.
        # Actually, for a regular hexagon packing, if the center is at 0,0,
        # and a neighbor is at (d, 0), the neighbor should be rotated by 0 or 60?
        # Let's assume the rotation is simply the global rotation + local angle.
        # This is a heuristic. The optimizer can refine positions, but here we fix the structure.
        # To be safe, let's allow the rotation to be part of the structure.
        # For 1+6+5, the 6 inner ones form a ring.
        # Let's set their rotation to theta + global_rot_deg.
        centers.append([x, y, theta * 180 / np.pi + global_rot_deg])
        
    # 3. Outer ring (5 hexagons)
    # 5 hexagons. Where do they go?
    # In a 1+6+5 structure, the 5 outer hexagons typically sit in the "gaps" between the inner ring hexagons,
    # or they replace one of the 6 to make 5?
    # A common 12-hexagon packing is 1 center, 6 in a ring, and 5 more surrounding.
    # Let's place them at angles: 30, 90, 150, 210, 270 + offset + global_rot?
    # Or 0, 72, 144...? No, hexagonal symmetry suggests 60 deg steps.
    # With 5 hexagons, we break the 6-fold symmetry.
    # Let's place them at angles: 30, 90, 150, 210, 270 (offset by 30 from the inner ring vertices)
    # plus the offset_deg parameter.
    angles = [30, 90, 150, 210, 270]
    for i, ang in enumerate(angles):
        theta = np.radians(ang + offset_deg + global_rot_deg)
        x = r2 * np.cos(theta)
        y = r2 * np.sin(theta)
        # Rotation: align with local radial direction?
        # Let's use theta + global_rot_deg
        centers.append([x, y, theta * 180 / np.pi + global_rot_deg])
        
    return centers

def construct():
    start_time = time.time()
    
    best_score = float('inf')
    best_result = None
    
    # Parameters:
    # r1: inner ring radius (approx 2 for touching)
    # r2: outer ring radius (approx 3-4 for touching)
    # offset_deg: angular offset of outer ring
    # global_rot_deg: global rotation
    
    # Grid search ranges
    r1_range = np.linspace(1.8, 2.2, 20)
    r2_range = np.linspace(3.0, 4.0, 30)
    offset_range = np.linspace(0, 60, 13)
    rot_range = np.linspace(0, 60, 13)
    
    # We will do a coarse grid search, then refine the best few candidates with local optimization.
    
    candidate_configs = []
    
    # Generate grid
    for r1 in r1_range:
        for r2 in r2_range:
            for offset in offset_range:
                for rot in rot_range:
                    elapsed = time.time() - start_time
                    if elapsed > DEADLINE - 20:
                        break
                        
                    config = generate_structural_config(r1, r2, offset, rot)
                    centers = np.array(config)
                    
                    # Quick validity check: compute side and check overlaps
                    # To speed up, we can skip full SAT if we just want to rank, but we need validity.
                    # Let's compute side first.
                    S = compute_required_side(centers)
                    
                    # Check overlaps
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
                    
                    if valid:
                        if S < best_score:
                            best_score = S
                            best_result = (config, (0.0, 0.0), S, 0.0)
                            
                        # Keep top candidates for refinement
                        candidate_configs.append((S, config))
                    else:
                        # Even if invalid, if it's close, it might be a good starting point for local opt
                        # But we prioritize valid ones.
                        pass
                        
            if elapsed > DEADLINE - 20: break
        if elapsed > DEADLINE - 20: break
    if elapsed > DEADLINE - 20: break
    
    # Sort candidates by score
    candidate_configs.sort(key=lambda x: x[0])
    
    # Refine top 10 candidates with local optimization
    # We will optimize the 4 parameters: r1, r2, offset, rot
    # by converting to a 4D vector and using L-BFGS-B on the structural generator.
    
    top_candidates = candidate_configs[:10]
    
    if not top_candidates:
        # Fallback if no valid config found in grid
        # Try a random config
        np.random.seed(42)
        config = generate_structural_config(2.0, 3.5, 30, 0)
        centers = np.array(config)
        S = compute_required_side(centers)
        return config, (0.0, 0.0), S, 0.0

    for S_init, config_init in top_candidates:
        elapsed = time.time() - start_time
        if elapsed > DEADLINE - 5:
            break
            
        # Recover params from config_init? 
        # It's easier to just use the config_init as x0 for a 36D optimization?
        # No, the structural constraint is the key.
        # Let's optimize the 4 parameters directly.
        # We need to map config_init back to (r1, r2, offset, rot).
        # This is tricky because the generation is deterministic.
        # Instead, let's just run a local optimization on the 36D space starting from config_init,
        # but with a very tight tolerance to stay close to the structure?
        # Or, better, let's assume the grid found the best structure, and just do a small local search
        # on the 4 parameters.
        
        # To do that, we need to invert the generation.
        # Let's just store the params that generated the config.
        # I will modify the grid loop to store params.
        
        pass # Placeholder, will implement below

    # Re-implement grid search with param storage
    candidate_params = []
    best_score = float('inf')
    best_result = None
    
    for r1 in r1_range:
        for r2 in r2_range:
            for offset in offset_range:
                for rot in rot_range:
                    elapsed = time.time() - start_time
                    if elapsed > DEADLINE - 20:
                        break
                        
                    config = generate_structural_config(r1, r2, offset, rot)
                    centers = np.array(config)
                    
                    S = compute_required_side(centers)
                    
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
                    
                    if valid:
                        if S < best_score:
                            best_score = S
                            best_result = (config, (0.0, 0.0), S, 0.0)
                        candidate_params.append((S, [r1, r2, offset, rot]))
                    else:
                        # Store invalid ones too if they are close? 
                        # No, stick to valid for refinement start.
                        pass
                        
            if elapsed > DEADLINE - 20: break
        if elapsed > DEADLINE - 20: break
    if elapsed > DEADLINE - 20: break
    
    candidate_params.sort(key=lambda x: x[0])
    
    # Refine top 5
    top_params = candidate_params[:5]
    
    for S_init, params_init in top_params:
        elapsed = time.time() - start_time
        if elapsed > DEADLINE - 5:
            break
            
        x0 = np.array(params_init)
        
        def obj_structural(params):
            r1, r2, offset, rot = params
            # Bounds check
            if r1 < 1.5 or r1 > 2.5 or r2 < 2.5 or r2 > 4.5:
                return 1e10
            config = generate_structural_config(r1, r2, offset, rot)
            centers = np.array(config)
            
            S = compute_required_side(centers)
            
            verts = get_hex_vertices(centers)
            overlap_penalty = 0.0
            for i in range(N):
                v1 = verts[i]
                for j in range(i + 1, N):
                    v2 = verts[j]
                    sep = sat_min_separation(v1, v2)
                    if sep < 0:
                        overlap_penalty += (-sep) ** 2
                        
            return S + 100.0 * overlap_penalty
            
        try:
            res = minimize(
                obj_structural, x0, method='L-BFGS-B',
                bounds=[(1.5, 2.5), (2.5, 4.5), (0, 60), (0, 60)],
                options={'maxiter': 50, 'ftol': 1e-6, 'gtol': 1e-5, 'disp': False}
            )
            new_params = res.x
            new_config = generate_structural_config(*new_params)
            new_centers = np.array(new_config)
            
            # Validate
            new_S = compute_required_side(new_centers)
            verts = get_hex_vertices(new_centers)
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
                    
            if valid and new_S < best_score:
                best_score = new_S
                best_result = (new_config, (0.0, 0.0), new_S, 0.0)
                
        except Exception:
            pass

    if best_result is not None:
        return best_result
    
    # Fallback
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
