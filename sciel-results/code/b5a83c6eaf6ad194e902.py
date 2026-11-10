# EVOLVE-BLOCK-START
import numpy as np
import time

def _hex_vertices(cx, cy, d, angle_deg):
    """Compute 6 vertices of a regular hexagon with center (cx,cy), side d, angle in degrees."""
    a = np.radians(angle_deg)
    angles = a + np.radians(60) * np.arange(6)
    return np.column_stack([cx + d * np.cos(angles), cy + d * np.sin(angles)])

def _is_valid_fast(inner, outer_center, outer_angle, s):
    """
    Fast validity check.
    1. Check finite.
    2. Check s >= sqrt(12).
    3. Check inner hexagons don't overlap (using SAT on vertices).
    4. Check inner vertices inside outer hexagon.
    """
    if not np.all(np.isfinite(inner)) or not np.all(np.isfinite(outer_center)) or not np.isfinite(s):
        return False
    if s < np.sqrt(12) - 1e-9:
        return False
    
    outer_verts = _hex_vertices(outer_center[0], outer_center[1], s, outer_angle)
    
    # Check each inner hexagon
    for i in range(12):
        cx, cy, ang = inner[i]
        verts = _hex_vertices(cx, cy, 1.0, ang)
        
        # Check if all vertices are inside outer hexagon
        for v in verts:
            for k in range(6):
                p1 = outer_verts[k]
                p2 = outer_verts[(k+1)%6]
                cross = (p2[0] - p1[0]) * (v[1] - p1[1]) - (p2[1] - p1[1]) * (v[0] - p1[0])
                if cross < -1e-6:
                    return False

        # Check overlap with other inner hexagons
        for j in range(i+1, 12):
            cx2, cy2, ang2 = inner[j]
            verts2 = _hex_vertices(cx2, cy2, 1.0, ang2)
            
            # Quick distance check
            dist = np.linalg.norm(inner[i, :2] - inner[j, :2])
            if dist > 3.5:
                continue

            # Separating Axis Test
            normals = []
            for v in [verts, verts2]:
                for k in range(6):
                    p1 = v[k]
                    p2 = v[(k+1)%6]
                    edge = p2 - p1
                    normal = np.array([-edge[1], edge[0]])
                    norm_n = np.linalg.norm(normal)
                    if norm_n < 1e-9: continue
                    normal = normal / norm_n
                    normals.append(normal)
            
            overlap = True
            for normal in normals:
                proj_i = np.dot(verts, normal)
                proj_j = np.dot(verts2, normal)
                min_i, max_i = np.min(proj_i), np.max(proj_i)
                min_j, max_j = np.min(proj_j), np.max(proj_j)
                
                if max_i < min_j - 1e-6 or max_j < min_i - 1e-6:
                    overlap = False
                    break
            
            if overlap:
                return False

    return True

def _compute_min_outer_side(inner, outer_center, outer_angle):
    """Compute the minimum side length of outer hexagon that contains all inner hexagons."""
    shifted_centers = inner[:, :2] - outer_center
    
    all_verts = []
    for i in range(12):
        verts = _hex_vertices(shifted_centers[i, 0], shifted_centers[i, 1], 1.0, inner[i, 2])
        all_verts.append(verts)
    all_verts = np.vstack(all_verts)
    
    a_rad = np.radians(outer_angle)
    unit_outer_verts = np.column_stack([
        np.cos(a_rad + np.radians(60) * np.arange(6)),
        np.sin(a_rad + np.radians(60) * np.arange(6))
    ])
    
    max_proj = 0
    for k in range(6):
        p1 = unit_outer_verts[k]
        p2 = unit_outer_verts[(k+1)%6]
        edge = p2 - p1
        normal = np.array([edge[1], -edge[0]])
        normal = normal / (np.linalg.norm(normal) + 1e-15)
        
        # Ensure normal points outward
        mid = (p1 + p2) / 2
        if np.dot(normal, mid) < 0:
            normal = -normal
            
        projs = all_verts @ normal
        max_k = np.max(projs)
        if max_k > max_proj:
            max_proj = max_k
    
    min_s = max_proj * 2.0 / np.sqrt(3)
    return max(min_s, np.sqrt(12))

def _generate_honeycomb_seed():
    """
    Generate a compact honeycomb seed: 1 center, 6 first ring, 5 second ring.
    This is a 12-hexagon cluster.
    """
    L = 2 * np.sqrt(3) + 0.05
    points = [[0.0, 0.0, 0.0]]
    # First ring (6 hexagons)
    for k in range(6):
        angle = np.radians(60 * k)
        points.append([L * np.cos(angle), L * np.sin(angle), 0.0])
    # Second ring (5 hexagons, leaving one gap for compactness or just pick 5)
    second_ring_coords = [
        (2, 0),
        (1, 1),
        (0, 2),
        (-1, 2),
        (-2, 1)
    ]
    for (q, r) in second_ring_coords:
        x = L * (q + r/2.0)
        y = L * r * np.sqrt(3)/2
        points.append([x, y, 0.0])
        
    pts = np.array(points)
    center = np.mean(pts[:, :2], axis=0)
    pts[:, 0] -= center[0]
    pts[:, 1] -= center[1]
    return pts

def _generate_lattice_seed():
    """Generate a 3x4 lattice seed."""
    L = 2 * np.sqrt(3) + 0.1
    points = []
    for r in range(3):
        for q in range(4):
            x = (q + r/2.0) * L
            y = r * np.sqrt(3)/2 * L
            points.append([x, y, 0.0])
    pts = np.array(points)
    center = np.mean(pts[:, :2], axis=0)
    pts[:, 0] -= center[0]
    pts[:, 1] -= center[1]
    return pts

def hexagon_packing_12():
    start_time = time.time()
    deadline = start_time + 600.0

    # Initial seeds
    seeds = []
    seeds.append(_generate_honeycomb_seed())
    seeds.append(_generate_lattice_seed())

    best_inner = None
    best_s = float('inf')
    best_outer_center = np.array([0.0, 0.0])
    best_outer_angle = 0.0
    
    # Evaluate seeds
    for seed in seeds:
        for angle_deg in [0, 30]:
            s = _compute_min_outer_side(seed, best_outer_center, angle_deg)
            if s < best_s:
                best_s = s
                best_inner = seed.copy()
                best_outer_angle = angle_deg
                
    # Simulated Annealing with adaptive step sizes
    current_inner = best_inner.copy()
    current_s = best_s
    current_outer_angle = best_outer_angle
    
    T_initial = 0.5
    T_min = 0.001
    n_iterations = 50000
    
    np.random.seed(42)
    
    # Adaptive step sizes
    step_pos = 0.2
    step_ang = 0.1
    success_count = 0
    last_update = 0
    
    for it in range(n_iterations):
        if time.time() > deadline - 60.0:
            break
            
        T = T_initial * (T_min / T_initial) ** (it / n_iterations)
        
        new_inner = current_inner.copy()
        new_outer_angle = current_outer_angle
        
        # Perturb positions and angles
        n_perturb = np.random.choice([1, 2, 3], p=[0.5, 0.3, 0.2])
        indices = np.random.choice(12, size=n_perturb, replace=False)
        for idx in indices:
            new_inner[idx, 0] += np.random.randn() * step_pos
            new_inner[idx, 1] += np.random.randn() * step_pos
            new_inner[idx, 2] += np.random.randn() * step_ang
            
        # Perturb outer angle
        if np.random.rand() < 0.2:
            new_outer_angle += np.random.randn() * 0.1
            
        new_s = _compute_min_outer_side(new_inner, best_outer_center, new_outer_angle)
        
        # Check validity
        new_valid = _is_valid_fast(new_inner, best_outer_center, new_outer_angle, new_s)
        
        if new_valid:
            if new_s < current_s:
                current_inner = new_inner
                current_s = new_s
                current_outer_angle = new_outer_angle
                if current_s < best_s:
                    best_inner = current_inner.copy()
                    best_s = current_s
                    best_outer_angle = current_outer_angle
                success_count += 1
            else:
                delta = new_s - current_s
                if np.random.rand() < np.exp(-delta / max(T, 1e-10)):
                    current_inner = new_inner
                    current_s = new_s
                    current_outer_angle = new_outer_angle
                    success_count += 1
        else:
            # Penalty based on overlap
            penalty = 0
            for i in range(12):
                for j in range(i+1, 12):
                    d = np.linalg.norm(new_inner[i, :2] - new_inner[j, :2])
                    min_d = 2 * np.sqrt(3) 
                    if d < min_d:
                        penalty += (min_d - d)**2
            if penalty < 0.5 and np.random.rand() < 0.1:
                current_inner = new_inner
                current_s = new_s + penalty
                current_outer_angle = new_outer_angle
        
        # Adapt step sizes every 500 iterations
        if it - last_update >= 500:
            if success_count > 100:
                # Increase step size
                step_pos *= 1.1
                step_ang *= 1.1
            elif success_count < 20:
                # Decrease step size
                step_pos *= 0.9
                step_ang *= 0.9
            success_count = 0
            last_update = it
            # Clamp step sizes
            step_pos = np.clip(step_pos, 0.05, 1.0)
            step_ang = np.clip(step_ang, 0.02, 0.5)

    # Final refinement with Nelder-Mead
    from scipy.optimize import minimize
    
    def objective(params):
        if time.time() > deadline - 10.0:
            return 1e10
        inner = params[:36].reshape(12, 3).copy()
        outer_angle = params[36]
        
        s = _compute_min_outer_side(inner, best_outer_center, outer_angle)
        if not _is_valid_fast(inner, best_outer_center, outer_angle, s):
            return s + 10.0
        return s

    x0 = np.concatenate([best_inner.flatten(), [best_outer_angle]])
    
    try:
        result = minimize(
            objective, x0,
            method='Nelder-Mead',
            options={'maxiter': 300, 'xatol': 1e-7, 'fatol': 1e-9}
        )
        if result.fun < best_s:
            candidate_inner = result.x[:36].reshape(12, 3).copy()
            candidate_angle = result.x[36]
            candidate_s = _compute_min_outer_side(candidate_inner, best_outer_center, candidate_angle)
            if _is_valid_fast(candidate_inner, best_outer_center, candidate_angle, candidate_s):
                best_inner = candidate_inner
                best_s = candidate_s
                best_outer_angle = candidate_angle
    except Exception:
        pass

    # Final check
    if not _is_valid_fast(best_inner, best_outer_center, best_outer_angle, best_s):
        # Fallback to seed
        best_inner = seeds[1].copy()
        best_s = _compute_min_outer_side(best_inner, best_outer_center, 0)
        best_outer_angle = 0

    outer_hex_data = np.array([0.0, 0.0, best_outer_angle])
    return best_inner, outer_hex_data, float(best_s)

# EVOLVE-BLOCK-END
