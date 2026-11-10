# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def _hex_vertices(cx, cy, d, angle_deg):
    """Compute 6 vertices of a regular hexagon with center (cx,cy), side d, angle in degrees."""
    a = np.radians(angle_deg)
    angles = a + np.radians(60) * np.arange(6)
    return np.column_stack([cx + d * np.cos(angles), cy + d * np.sin(angles)])

def _compute_s_fast(inner, outer_center, outer_angle):
    """
    Compute the minimum side length of outer hexagon that contains all inner hexagons.
    Returns the side length s.
    """
    shifted = inner[:, :2] - outer_center
    a_rad = np.radians(outer_angle)
    # Unit outer hexagon vertices
    outer_unit_verts = np.column_stack([
        np.cos(a_rad + np.radians(60) * np.arange(6)),
        np.sin(a_rad + np.radians(60) * np.arange(6))
    ])
    
    # For each outer edge, compute the outward normal
    max_proj = 0.0
    for k in range(6):
        p1 = outer_unit_verts[k]
        p2 = outer_unit_verts[(k+1) % 6]
        edge = p2 - p1
        # Outward normal (pointing away from center)
        normal = np.array([edge[1], -edge[0]])
        norm_n = np.linalg.norm(normal)
        if norm_n < 1e-15:
            continue
        normal = normal / norm_n
        # Ensure outward: dot with midpoint should be positive
        mid = (p1 + p2) / 2
        if np.dot(normal, mid) < 0:
            normal = -normal
        
        # For each inner hexagon, find max projection of its vertices onto this normal
        # Inner hexagon i has center shifted[i] and angle inner[i,2]
        for i in range(12):
            cx, cy = shifted[i]
            ang = np.radians(inner[i, 2])
            # Vertices of inner hexagon i
            v_angles = ang + np.radians(60) * np.arange(6)
            vx = cx + np.cos(v_angles)
            vy = cy + np.sin(v_angles)
            proj = vx * normal[0] + vy * normal[1]
            max_p = np.max(proj)
            if max_p > max_proj:
                max_proj = max_p
    
    # The apothem of outer hexagon with side s is s * sqrt(3)/2
    # max_proj is the max distance from center to any inner vertex along outward normals
    # So s * sqrt(3)/2 >= max_proj => s >= max_proj * 2/sqrt(3)
    min_s = max_proj * 2.0 / np.sqrt(3)
    return max(min_s, np.sqrt(12))

def _is_valid_fast(inner, outer_center, outer_angle, s):
    """
    Fast validity check:
    1. Finite values
    2. s >= sqrt(12)
    3. No overlaps between inner hexagons (SAT)
    4. All inner vertices inside outer hexagon
    """
    if not np.all(np.isfinite(inner)) or not np.all(np.isfinite(outer_center)) or not np.isfinite(s):
        return False
    if s < np.sqrt(12) - 1e-9:
        return False
    
    outer_verts = _hex_vertices(outer_center[0], outer_center[1], s, outer_angle)
    
    # Precompute inner vertices
    all_inner_verts = []
    for i in range(12):
        cx, cy, ang = inner[i]
        verts = _hex_vertices(cx, cy, 1.0, ang)
        all_inner_verts.append(verts)
    
    # Check containment: all inner vertices inside outer hexagon
    for i in range(12):
        for v in all_inner_verts[i]:
            for k in range(6):
                p1 = outer_verts[k]
                p2 = outer_verts[(k+1) % 6]
                cross = (p2[0] - p1[0]) * (v[1] - p1[1]) - (p2[1] - p1[1]) * (v[0] - p1[0])
                if cross < -1e-6:
                    return False
    
    # Check non-overlap via SAT
    for i in range(12):
        for j in range(i + 1, 12):
            # Quick distance check
            dist = np.linalg.norm(inner[i, :2] - inner[j, :2])
            if dist > 3.5:
                continue
            
            v1 = all_inner_verts[i]
            v2 = all_inner_verts[j]
            
            # Collect normals from both hexagons
            normals = []
            for v in [v1, v2]:
                for k in range(6):
                    p1 = v[k]
                    p2 = v[(k+1) % 6]
                    edge = p2 - p1
                    normal = np.array([-edge[1], edge[0]])
                    norm_n = np.linalg.norm(normal)
                    if norm_n < 1e-9:
                        continue
                    normal = normal / norm_n
                    normals.append(normal)
            
            overlap = True
            for normal in normals:
                proj1 = v1 @ normal
                proj2 = v2 @ normal
                min1, max1 = np.min(proj1), np.max(proj1)
                min2, max2 = np.min(proj2), np.max(proj2)
                if max1 < min2 - 1e-6 or max2 < min1 - 1e-6:
                    overlap = False
                    break
            
            if overlap:
                return False
    
    return True

def _generate_honeycomb_seed():
    """Generate a compact honeycomb-like seed: 1 center, 6 first ring, 5 second ring."""
    L = 2 * np.sqrt(3) + 0.02
    points = [[0.0, 0.0, 0.0]]
    # First ring (6 hexagons)
    for k in range(6):
        angle = np.radians(60 * k)
        points.append([L * np.cos(angle), L * np.sin(angle), 0.0])
    # Second ring: pick 5 positions for compactness
    second_ring = [
        (2, 0), (1, 1), (0, 2), (-1, 2), (-2, 1)
    ]
    for (q, r) in second_ring:
        x = L * (q + r / 2.0)
        y = L * r * np.sqrt(3) / 2
        points.append([x, y, 0.0])
    
    pts = np.array(points)
    center = np.mean(pts[:, :2], axis=0)
    pts[:, 0] -= center[0]
    pts[:, 1] -= center[1]
    return pts

def _generate_lattice_seed():
    """Generate a 3x4 lattice seed."""
    L = 2 * np.sqrt(3) + 0.02
    points = []
    for r in range(3):
        for q in range(4):
            x = (q + r / 2.0) * L
            y = r * np.sqrt(3) / 2 * L
            points.append([x, y, 0.0])
    pts = np.array(points)
    center = np.mean(pts[:, :2], axis=0)
    pts[:, 0] -= center[0]
    pts[:, 1] -= center[1]
    return pts

def hexagon_packing_12():
    start_time = time.time()
    deadline = start_time + 580.0  # Safety margin before 600s limit
    
    # Initial seeds
    seeds = [_generate_honeycomb_seed(), _generate_lattice_seed()]
    
    best_inner = None
    best_s = float('inf')
    best_outer_angle = 0.0
    outer_center = np.array([0.0, 0.0])
    
    # Evaluate seeds with different outer angles
    for seed in seeds:
        for angle_deg in [0, 15, 30, 45]:
            s = _compute_s_fast(seed, outer_center, angle_deg)
            if s < best_s:
                best_s = s
                best_inner = seed.copy()
                best_outer_angle = angle_deg
    
    # Phase 1: Simulated Annealing with adaptive step sizes
    current_inner = best_inner.copy()
    current_s = best_s
    current_outer_angle = best_outer_angle
    
    T_initial = 0.3
    T_min = 0.0005
    n_iterations = 80000
    
    np.random.seed(42)
    
    step_pos = 0.15
    step_ang = 0.08
    success_count = 0
    last_update = 0
    
    for it in range(n_iterations):
        if time.time() > deadline - 120.0:
            break
        
        T = T_initial * (T_min / T_initial) ** (it / n_iterations)
        
        new_inner = current_inner.copy()
        new_outer_angle = current_outer_angle
        
        # Perturb positions and angles
        n_perturb = np.random.choice([1, 2, 3, 4], p=[0.35, 0.30, 0.25, 0.10])
        indices = np.random.choice(12, size=n_perturb, replace=False)
        for idx in indices:
            new_inner[idx, 0] += np.random.randn() * step_pos
            new_inner[idx, 1] += np.random.randn() * step_pos
            new_inner[idx, 2] += np.random.randn() * step_ang
        
        # Perturb outer angle
        if np.random.rand() < 0.25:
            new_outer_angle += np.random.randn() * 0.08
        
        new_s = _compute_s_fast(new_inner, outer_center, new_outer_angle)
        
        # Check validity
        new_valid = _is_valid_fast(new_inner, outer_center, new_outer_angle, new_s)
        
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
            # Accept with small probability if penalty is low
            penalty = 0.0
            for i in range(12):
                for j in range(i + 1, 12):
                    d = np.linalg.norm(new_inner[i, :2] - new_inner[j, :2])
                    min_d = 2 * np.sqrt(3)
                    if d < min_d:
                        penalty += (min_d - d) ** 2
            if penalty < 0.3 and np.random.rand() < 0.08:
                current_inner = new_inner
                current_s = new_s + penalty
                current_outer_angle = new_outer_angle
        
        # Adapt step sizes
        if it - last_update >= 400:
            if success_count > 80:
                step_pos *= 1.08
                step_ang *= 1.08
            elif success_count < 15:
                step_pos *= 0.92
                step_ang *= 0.92
            success_count = 0
            last_update = it
            step_pos = np.clip(step_pos, 0.03, 0.8)
            step_ang = np.clip(step_ang, 0.01, 0.4)
    
    # Phase 2: Nelder-Mead local optimization
    def objective(params):
        if time.time() > deadline - 30.0:
            return 1e10
        inner = params[:36].reshape(12, 3).copy()
        outer_angle = params[36]
        
        s = _compute_s_fast(inner, outer_center, outer_angle)
        if not _is_valid_fast(inner, outer_center, outer_angle, s):
            # Compute penalty
            penalty = 0.0
            for i in range(12):
                for j in range(i + 1, 12):
                    d = np.linalg.norm(inner[i, :2] - inner[j, :2])
                    min_d = 2 * np.sqrt(3)
                    if d < min_d:
                        penalty += (min_d - d) ** 2
            # Check containment
            outer_verts = _hex_vertices(0, 0, s, outer_angle)
            for i in range(12):
                cx, cy, ang = inner[i]
                verts = _hex_vertices(cx, cy, 1.0, ang)
                for v in verts:
                    for k in range(6):
                        p1 = outer_verts[k]
                        p2 = outer_verts[(k+1) % 6]
                        cross = (p2[0] - p1[0]) * (v[1] - p1[1]) - (p2[1] - p1[1]) * (v[0] - p1[0])
                        if cross < -1e-6:
                            penalty += abs(cross) * 10
            return s + penalty * 5.0
        return s
    
    x0 = np.concatenate([best_inner.flatten(), [best_outer_angle]])
    
    try:
        result = minimize(
            objective, x0,
            method='Nelder-Mead',
            options={'maxiter': 500, 'xatol': 1e-8, 'fatol': 1e-10, 'adaptive': True}
        )
        if result.fun < best_s:
            candidate_inner = result.x[:36].reshape(12, 3).copy()
            candidate_angle = result.x[36]
            candidate_s = _compute_s_fast(candidate_inner, outer_center, candidate_angle)
            if _is_valid_fast(candidate_inner, outer_center, candidate_angle, candidate_s):
                best_inner = candidate_inner
                best_s = candidate_s
                best_outer_angle = candidate_angle
    except Exception:
        pass
    
    # Phase 3: Multiple restarts of Nelder-Mead from perturbed best
    if time.time() < deadline - 60.0:
        np.random.seed(123)
        for restart in range(5):
            if time.time() > deadline - 30.0:
                break
            x0_perturbed = x0.copy()
            x0_perturbed[:36] += np.random.randn(36) * 0.05
            x0_perturbed[36] += np.random.randn() * 0.05
            
            try:
                result = minimize(
                    objective, x0_perturbed,
                    method='Nelder-Mead',
                    options={'maxiter': 300, 'xatol': 1e-8, 'fatol': 1e-10, 'adaptive': True}
                )
                if result.fun < best_s:
                    candidate_inner = result.x[:36].reshape(12, 3).copy()
                    candidate_angle = result.x[36]
                    candidate_s = _compute_s_fast(candidate_inner, outer_center, candidate_angle)
                    if _is_valid_fast(candidate_inner, outer_center, candidate_angle, candidate_s):
                        best_inner = candidate_inner
                        best_s = candidate_s
                        best_outer_angle = candidate_angle
                        x0 = result.x.copy()
            except Exception:
                pass
    
    # Final validation and fallback
    if not _is_valid_fast(best_inner, outer_center, best_outer_angle, best_s):
        best_inner = seeds[1].copy()
        best_s = _compute_s_fast(best_inner, outer_center, 0)
        best_outer_angle = 0
    
    outer_hex_data = np.array([0.0, 0.0, best_outer_angle])
    return best_inner, outer_hex_data, float(best_s)

# EVOLVE-BLOCK-END
