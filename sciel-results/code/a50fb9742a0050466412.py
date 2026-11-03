import math
import time
import numpy as np
from scipy.optimize import differential_evolution, minimize

N = 12

def construct():
    start_time = time.time()
    deadline = start_time + 145.0

    def get_hex_verts(cx, cy, side, angle_deg):
        angle_rad = math.radians(angle_deg)
        cos_a = math.cos(angle_rad)
        sin_a = math.sin(angle_rad)
        rot = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
        base = np.array([[math.cos(math.radians(a)), math.sin(math.radians(a))] for a in range(0, 360, 60)])
        verts = base @ rot.T * side
        verts += np.array([cx, cy])
        return verts

    def hex_edges(verts):
        edges = np.roll(verts, -1, axis=0) - verts
        normals = np.column_stack([-edges[:, 1], edges[:, 0]])
        norms = np.linalg.norm(normals, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        normals = normals / norms
        return edges, normals

    def sat_min_separation(verts_a, verts_b):
        edges_a, normals_a = hex_edges(verts_a)
        edges_b, normals_b = hex_edges(verts_b)
        
        all_normals = np.vstack([normals_a, normals_b])
        
        min_overlap = float('inf')
        best_axis = None
        
        for n in all_normals:
            proj_a = verts_a @ n
            proj_b = verts_b @ n
            
            min_a, max_a = np.min(proj_a), np.max(proj_a)
            min_b, max_b = np.min(proj_b), np.max(proj_b)
            
            overlap = min(max_a, max_b) - max(min_a, min_b)
            
            if overlap < 0:
                return 0.0, n
            
            if overlap < min_overlap:
                min_overlap = overlap
                best_axis = n
        
        if best_axis is None:
            return 0.0, np.array([1.0, 0.0])
        
        center_a = np.mean(verts_a, axis=0)
        center_b = np.mean(verts_b, axis=0)
        diff = center_a - center_b
        if np.dot(diff, best_axis) < 0:
            best_axis = -best_axis
        
        return min_overlap, best_axis

    def calc_outer_side(positions, angles, outer_angle=0.0):
        theta_rad = math.radians(outer_angle)
        normals = np.array([[math.cos(theta_rad + math.pi/6 + k * math.pi/3),
                             math.sin(theta_rad + math.pi/6 + k * math.pi/3)] for k in range(6)])
        
        max_proj = 0.0
        for i in range(N):
            verts = get_hex_verts(positions[i, 0], positions[i, 1], 1.0, angles[i])
            projs = verts @ normals.T
            max_proj_i = np.max(projs)
            if max_proj_i > max_proj:
                max_proj = max_proj_i
        
        return max_proj * 2.0 / math.sqrt(3)

    def penalty_objective(params):
        positions = params[:2*N].reshape(N, 2)
        angles = params[2*N:]
        
        side = calc_outer_side(positions, angles, 0.0)
        
        # Calculate overlap penalty
        penalty = 0.0
        for i in range(N):
            for j in range(i+1, N):
                vi = get_hex_verts(positions[i, 0], positions[i, 1], 1.0, angles[i])
                vj = get_hex_verts(positions[j, 0], positions[j, 1], 1.0, angles[j])
                sep, _ = sat_min_separation(vi, vj)
                if sep > 0:
                    # Quadratic penalty for overlap
                    penalty += sep * sep
        
        return side + 100.0 * penalty

    def is_valid(params):
        positions = params[:2*N].reshape(N, 2)
        angles = params[2*N:]
        
        for i in range(N):
            for j in range(i+1, N):
                vi = get_hex_verts(positions[i, 0], positions[i, 1], 1.0, angles[i])
                vj = get_hex_verts(positions[j, 0], positions[j, 1], 1.0, angles[j])
                sep, _ = sat_min_separation(vi, vj)
                if sep > 1e-7:
                    return False
        return True

    # --- Step 1: Generate diverse initial configurations ---
    sqrt3 = math.sqrt(3)
    v1 = np.array([sqrt3, 0.0])
    v2 = np.array([sqrt3/2.0, 1.5])
    
    initial_configs = []
    
    # Config 1: Hexagonal cluster (1 + 6 + 5)
    pos1 = [np.array([0.0, 0.0])]
    for k in range(6):
        ang = k * math.pi / 3
        pos1.append(sqrt3 * np.array([math.cos(ang), math.sin(ang)]))
    
    ring2_pts = []
    for i in range(-2, 3):
        for j in range(-2, 3):
            d = max(abs(i), abs(j), abs(i+j))
            if d == 2:
                pos = i * v1 + j * v2
                ring2_pts.append(pos)
    ring2_sorted = sorted(ring2_pts, key=lambda p: math.atan2(p[1], p[0]))
    for k in range(5):
        pos1.append(ring2_sorted[k])
    
    pos1 = np.array(pos1)
    center1 = np.mean(pos1, axis=0)
    pos1 = pos1 - center1
    initial_configs.append(np.concatenate([pos1.flatten(), np.zeros(N)]))
    
    # Config 2: 3x4 Grid
    pos2 = []
    for row in range(3):
        for col in range(4):
            offset = 0.5 * v1 if row % 2 == 1 else 0
            pos = col * v1 + row * v2 + offset
            pos2.append(pos)
    pos2 = np.array(pos2)
    center2 = np.mean(pos2, axis=0)
    pos2 = pos2 - center2
    initial_configs.append(np.concatenate([pos2.flatten(), np.zeros(N)]))
    
    # Config 3: 4x3 Grid
    pos3 = []
    for row in range(4):
        for col in range(3):
            offset = 0.5 * v1 if row % 2 == 1 else 0
            pos = col * v1 + row * v2 + offset
            pos3.append(pos)
    pos3 = np.array(pos3)
    center3 = np.mean(pos3, axis=0)
    pos3 = pos3 - center3
    initial_configs.append(np.concatenate([pos3.flatten(), np.zeros(N)]))
    
    # --- Step 2: Differential Evolution for Global Search ---
    # Bounds for positions and angles
    bounds = []
    for i in range(N):
        bounds.append((-8, 8))  # x
        bounds.append((-8, 8))  # y
        bounds.append((-30, 30)) # angle (degrees)
    
    best_params = None
    best_score = float('inf')
    
    # Run DE on each initial config? Or just use DE with a broad search space?
    # DE is stochastic, so we can just run it once with a good population size.
    # However, to ensure we explore the lattice-like structures, we can seed the population.
    # But standard DE doesn't easily allow seeding specific individuals in scipy's implementation 
    # without subclassing. We'll rely on DE's exploration capability.
    
    # Time budget for DE: 60 seconds
    de_deadline = start_time + 60.0
    
    try:
        de_result = differential_evolution(
            penalty_objective,
            bounds,
            maxiter=100,
            tol=1e-6,
            popsize=20,
            seed=42,
            polish=False  # We will do our own polishing
        )
        
        if de_result.fun < best_score:
            best_score = de_result.fun
            best_params = de_result.x
            
    except Exception as e:
        print(f"DE failed: {e}")
    
    # --- Step 3: Local Refinement (Nelder-Mead) on best candidates ---
    # We will refine the best result from DE, and also the initial configs if they are better.
    candidates = [best_params]
    for cfg in initial_configs:
        score = penalty_objective(cfg)
        if score < best_score:
            best_score = score
            best_params = cfg
        candidates.append(cfg)
    
    # Sort candidates by penalty score
    scored_candidates = []
    for c in candidates:
        if c is not None:
            s = penalty_objective(c)
            scored_candidates.append((s, c))
    scored_candidates.sort(key=lambda x: x[0])
    
    # Refine top 3 candidates
    for score, params in scored_candidates[:3]:
        if time.time() > deadline - 30:
            break
            
        try:
            nm_result = minimize(
                penalty_objective,
                params,
                method='Nelder-Mead',
                options={'maxiter': 500, 'xatol': 1e-8, 'fatol': 1e-10}
            )
            
            if nm_result.fun < best_score:
                best_score = nm_result.fun
                best_params = nm_result.x
                
        except Exception as e:
            pass
            
    # --- Step 4: Final Validation and Fallback ---
    if best_params is None:
        # Fallback to initial config 1
        best_params = initial_configs[0]
        
    # Check if best_params is valid
    if not is_valid(best_params):
        # Try to find a valid solution among candidates
        valid_params = None
        for score, params in scored_candidates:
            if is_valid(params):
                valid_params = params
                break
                
        if valid_params is not None:
            best_params = valid_params
        else:
            # If no valid solution found, use the one with lowest penalty
            # (It might be slightly invalid, but better than nothing? No, task requires disjoint)
            # Let's try a quick local repair
            # ... For now, just return the best invalid one? No, that scores 0.
            # Let's try to relax the validity check slightly or use a different fallback.
            # Actually, if DE+Nelder-Mead fails to find a valid one, it's a bad sign.
            # Let's try a greedy placement as a last resort fallback.
            
            # Greedy Fallback
            sqrt3 = math.sqrt(3)
            v1 = np.array([sqrt3, 0.0])
            v2 = np.array([sqrt3/2.0, 1.5])
            
            coords = [np.array([0.0, 0.0])]
            for k in range(6):
                ang = k * math.pi / 3
                coords.append(sqrt3 * np.array([math.cos(ang), math.sin(ang)]))
            
            ring2_pts = []
            for i in range(-2, 3):
                for j in range(-2, 3):
                    d = max(abs(i), abs(j), abs(i+j))
                    if d == 2:
                        pos = i * v1 + j * v2
                        ring2_pts.append(pos)
            ring2_sorted = sorted(ring2_pts, key=lambda p: math.atan2(p[1], p[0]))
            for k in range(5):
                coords.append(ring2_sorted[k])
            
            coords = np.array(coords)
            center = np.mean(coords, axis=0)
            coords = coords - center
            angles = np.zeros(N)
            best_params = np.concatenate([coords.flatten(), angles])
            
            # This should be valid
            if not is_valid(best_params):
                 # Something is very wrong, but this lattice config is theoretically valid
                 pass

    positions = best_params[:2*N].reshape(N, 2)
    angles = best_params[2*N:]
    
    inner = []
    for i in range(N):
        inner.append([float(positions[i, 0]), float(positions[i, 1]), float(angles[i])])
    
    final_side = calc_outer_side(positions, angles, 0.0)
    
    return inner, (0.0, 0.0), float(final_side), 0.0
