# EVOLVE-BLOCK-START
import numpy as np
import time

def _hex_vertices(cx, cy, d, angle_deg):
    a = np.radians(angle_deg)
    angles = a + np.radians(60) * np.arange(6)
    return np.column_stack([cx + d * np.cos(angles), cy + d * np.sin(angles)])

def _edge_normals(verts):
    n = len(verts)
    normals = np.zeros((n, 2))
    for i in range(n):
        p1 = verts[i]
        p2 = verts[(i+1) % n]
        edge = p2 - p1
        normal = np.array([edge[1], -edge[0]])
        norm = np.linalg.norm(normal)
        if norm > 1e-12:
            normals[i] = normal / norm
    return normals

def _sat_overlap(verts1, normals1, verts2, normals2):
    all_normals = np.vstack([normals1, normals2])
    min_overlap = float('inf')
    best_axis = None
    best_sign = 0
    
    for normal in all_normals:
        proj1 = np.dot(verts1, normal)
        proj2 = np.dot(verts2, normal)
        min1, max1 = np.min(proj1), np.max(proj1)
        min2, max2 = np.min(proj2), np.max(proj2)
        
        if max1 < min2 - 1e-6:
            return 0.0, None, 0
        elif max2 < min1 - 1e-6:
            return 0.0, None, 0
        
        if max1 < min2:
            overlap = min2 - max1
            sign = 1
        elif max2 < min1:
            overlap = min1 - max2
            sign = -1
        else:
            overlap = min(max1, max2) - max(min1, min2)
            if overlap < 0:
                return 0.0, None, 0
            c1 = (min1 + max1) / 2
            c2 = (min2 + max2) / 2
            sign = 1 if c1 > c2 else -1
        
        if overlap < min_overlap:
            min_overlap = overlap
            best_axis = normal
            best_sign = sign
    
    return min_overlap, best_axis, best_sign

def _compute_all_verts_normals(inner):
    centers = inner[:, :2]
    angles = inner[:, 2]
    all_verts = np.zeros((12, 6, 2))
    all_normals = np.zeros((12, 6, 2))
    for i in range(12):
        verts = _hex_vertices(centers[i, 0], centers[i, 1], 1.0, angles[i])
        all_verts[i] = verts
        all_normals[i] = _edge_normals(verts)
    return all_verts, all_normals

def _compute_s_and_overlap_fast(inner, outer_angle):
    centers = inner[:, :2]
    all_verts, all_normals = _compute_all_verts_normals(inner)
    
    a_rad = np.radians(outer_angle)
    outer_unit_verts = np.column_stack([
        np.cos(a_rad + np.radians(60) * np.arange(6)),
        np.sin(a_rad + np.radians(60) * np.arange(6))
    ])
    outer_normals = _edge_normals(outer_unit_verts)
    
    all_inner_verts_flat = all_verts.reshape(72, 2)
    
    max_proj = 0.0
    for normal in outer_normals:
        projs = np.dot(all_inner_verts_flat, normal)
        max_k = np.max(projs)
        if max_k > max_proj:
            max_proj = max_k
    
    s = max_proj * 2.0 / np.sqrt(3)
    s = max(s, np.sqrt(12))
    
    is_valid = True
    for i in range(12):
        for j in range(i+1, 12):
            dist = np.linalg.norm(centers[i] - centers[j])
            if dist > 2.0 * np.sqrt(3) + 0.1:
                continue
            overlap, _, _ = _sat_overlap(all_verts[i], all_normals[i], all_verts[j], all_normals[j])
            if overlap > 1e-8:
                is_valid = False
                break
        if not is_valid:
            break
    
    return s, is_valid

def _compute_total_overlap(inner):
    """Compute total overlap penalty for all pairs (smooth surrogate)."""
    centers = inner[:, :2]
    all_verts, all_normals = _compute_all_verts_normals(inner)
    total = 0.0
    for i in range(12):
        for j in range(i+1, 12):
            dist = np.linalg.norm(centers[i] - centers[j])
            if dist > 2.0 * np.sqrt(3) + 0.1:
                continue
            overlap, _, _ = _sat_overlap(all_verts[i], all_normals[i], all_verts[j], all_normals[j])
            if overlap > 0:
                # Soft penalty: use overlap^2 for smoothness
                total += overlap * overlap
    return total

def _compute_s_smooth(inner, outer_angle):
    """
    Smooth objective: s + penalty for overlaps.
    Uses a soft formulation to give better signals to the optimizer.
    """
    s, is_valid = _compute_s_and_overlap_fast(inner, outer_angle)
    if is_valid:
        return s
    else:
        centers = inner[:, :2]
        all_verts, all_normals = _compute_all_verts_normals(inner)
        penalty = 0.0
        for i in range(12):
            for j in range(i+1, 12):
                dist = np.linalg.norm(centers[i] - centers[j])
                if dist > 2.0 * np.sqrt(3) + 0.1:
                    continue
                overlap, _, _ = _sat_overlap(all_verts[i], all_normals[i], all_verts[j], all_normals[j])
                if overlap > 0:
                    # Smooth penalty: quadratic in overlap
                    penalty += overlap * overlap
        return s + 10.0 * penalty + 1.0

def _resolve_collisions(inner, outer_angle, max_iters=50, push_factor=0.6):
    inner = inner.copy()
    
    for _ in range(max_iters):
        all_verts, all_normals = _compute_all_verts_normals(inner)
        centers = inner[:, :2]
        
        any_overlap = False
        for i in range(12):
            for j in range(i+1, 12):
                dist = np.linalg.norm(centers[i] - centers[j])
                if dist > 2.0 * np.sqrt(3) + 0.1:
                    continue
                overlap, axis, sign = _sat_overlap(all_verts[i], all_normals[i], all_verts[j], all_normals[j])
                if overlap > 1e-8 and axis is not None:
                    any_overlap = True
                    push = sign * axis * overlap * push_factor
                    inner[i, 0] += push[0]
                    inner[i, 1] += push[1]
                    inner[j, 0] -= push[0]
                    inner[j, 1] -= push[1]
        
        if not any_overlap:
            break
    
    return inner

def _generate_seeds():
    seeds = []
    L = 2 * np.sqrt(3) + 0.01
    
    # Seed 1: Hexagonal cluster (center + 6 ring + 5 second ring)
    points = [[0.0, 0.0, 0.0]]
    for k in range(6):
        angle = np.radians(60 * k)
        points.append([L * np.cos(angle), L * np.sin(angle), 0.0])
    second_ring = [(2, 0), (1, 1), (0, 2), (-1, 1), (-2, 0)]
    for (q, r) in second_ring:
        x = L * (q + r/2.0)
        y = L * r * np.sqrt(3)/2
        points.append([x, y, 0.0])
    pts = np.array(points)
    center = np.mean(pts[:, :2], axis=0)
    pts[:, 0] -= center[0]
    pts[:, 1] -= center[1]
    seeds.append(pts)
    
    # Seed 2: Triangular lattice 3x4
    points = []
    for row in range(3):
        for col in range(4):
            x = (col + row/2.0) * L
            y = row * L * np.sqrt(3)/2
            points.append([x, y, 0.0])
    pts = np.array(points)
    center = np.mean(pts[:, :2], axis=0)
    pts[:, 0] -= center[0]
    pts[:, 1] -= center[1]
    seeds.append(pts)
    
    # Seed 3: Two rows of 6
    points = []
    for row in range(2):
        for col in range(6):
            x = col * L
            y = row * L * np.sqrt(3)
            points.append([x, y, 0.0])
    pts = np.array(points)
    center = np.mean(pts[:, :2], axis=0)
    pts[:, 0] -= center[0]
    pts[:, 1] -= center[1]
    seeds.append(pts)
    
    # Seed 4: Circular arrangement with center
    points = [[0.0, 0.0, 0.0]]
    for k in range(11):
        angle = np.radians(360.0 * k / 11.0)
        r = L * 1.5
        points.append([r * np.cos(angle), r * np.sin(angle), 0.0])
    pts = np.array(points)
    center = np.mean(pts[:, :2], axis=0)
    pts[:, 0] -= center[0]
    pts[:, 1] -= center[1]
    seeds.append(pts)
    
    # Seed 5: 4-4-4 grid
    points = []
    for row in range(3):
        for col in range(4):
            x = (col + row/2.0) * L
            y = row * L * np.sqrt(3)/2
            points.append([x, y, 0.0])
    pts = np.array(points)
    center = np.mean(pts[:, :2], axis=0)
    pts[:, 0] -= center[0]
    pts[:, 1] -= center[1]
    seeds.append(pts)
    
    # Seed 6: Alternating rotations
    points = []
    for col in range(4):
        x = col * L
        y = 0.0
        points.append([x, y, 15.0])
    for col in range(4):
        x = (col + 0.5) * L
        y = L * np.sqrt(3)/2
        points.append([x, y, -15.0])
    for col in range(4):
        x = col * L
        y = L * np.sqrt(3)
        points.append([x, y, 15.0])
    pts = np.array(points)
    center = np.mean(pts[:, :2], axis=0)
    pts[:, 0] -= center[0]
    pts[:, 1] -= center[1]
    seeds.append(pts)
    
    # Seed 7: Hex cluster with varied rotations
    points = [[0.0, 0.0, 0.0]]
    for k in range(6):
        angle = np.radians(60 * k)
        rot = 30.0 if k % 2 == 0 else -30.0
        points.append([L * np.cos(angle), L * np.sin(angle), rot])
    second_ring = [(2, 0), (1, 1), (0, 2), (-1, 1), (-2, 0)]
    for idx, (q, r) in enumerate(second_ring):
        x = L * (q + r/2.0)
        y = L * r * np.sqrt(3)/2
        rot = 15.0 * (idx % 3 - 1)
        points.append([x, y, rot])
    pts = np.array(points)
    center = np.mean(pts[:, :2], axis=0)
    pts[:, 0] -= center[0]
    pts[:, 1] -= center[1]
    seeds.append(pts)
    
    # Seed 8: 2-4-4-2 pyramid
    points = []
    # Row 0: 2 hexes
    for col in range(2):
        x = (col + 1.0) * L
        y = 0.0
        points.append([x, y, 0.0])
    # Row 1: 4 hexes
    for col in range(4):
        x = (col + 0.5) * L
        y = L * np.sqrt(3)/2
        points.append([x, y, 0.0])
    # Row 2: 4 hexes
    for col in range(4):
        x = (col + 0.5) * L
        y = L * np.sqrt(3)
        points.append([x, y, 0.0])
    # Row 3: 2 hexes
    for col in range(2):
        x = (col + 1.0) * L
        y = L * np.sqrt(3) * 1.5
        points.append([x, y, 0.0])
    pts = np.array(points)
    center = np.mean(pts[:, :2], axis=0)
    pts[:, 0] -= center[0]
    pts[:, 1] -= center[1]
    seeds.append(pts)
    
    # Seed 9: Offset honeycomb
    points = []
    for row in range(3):
        for col in range(4):
            x = (col + row/2.0) * L
            y = row * L * np.sqrt(3)/2
            rot = 10.0 * ((row + col) % 2 - 0.5)
            points.append([x, y, rot])
    pts = np.array(points)
    center = np.mean(pts[:, :2], axis=0)
    pts[:, 0] -= center[0]
    pts[:, 1] -= center[1]
    seeds.append(pts)
    
    # Seed 10: Tighter hex cluster
    L2 = 2 * np.sqrt(3) + 0.001
    points = [[0.0, 0.0, 0.0]]
    for k in range(6):
        angle = np.radians(60 * k)
        points.append([L2 * np.cos(angle), L2 * np.sin(angle), 0.0])
    second_ring = [(2, 0), (1, 1), (0, 2), (-1, 1), (-2, 0)]
    for (q, r) in second_ring:
        x = L2 * (q + r/2.0)
        y = L2 * r * np.sqrt(3)/2
        points.append([x, y, 0.0])
    pts = np.array(points)
    center = np.mean(pts[:, :2], axis=0)
    pts[:, 0] -= center[0]
    pts[:, 1] -= center[1]
    seeds.append(pts)
    
    return seeds

def _compute_s_rigid(x, y, theta_c_deg, theta_o_deg, base_positions, base_angles):
    theta_c = np.radians(theta_c_deg)
    cos_c = np.cos(theta_c)
    sin_c = np.sin(theta_c)
    
    rot_positions = np.zeros_like(base_positions)
    rot_positions[:, 0] = base_positions[:, 0] * cos_c - base_positions[:, 1] * sin_c + x
    rot_positions[:, 1] = base_positions[:, 0] * sin_c + base_positions[:, 1] * cos_c + y
    
    angles = base_angles + theta_c_deg
    
    inner = np.zeros((12, 3))
    inner[:, 0] = rot_positions[:, 0]
    inner[:, 1] = rot_positions[:, 1]
    inner[:, 2] = angles
    
    s, valid = _compute_s_and_overlap_fast(inner, theta_o_deg)
    return s, valid

def _optimize_rigid_pattern(base_positions, base_angles, deadline):
    from scipy.optimize import minimize
    
    def objective(params):
        if time.time() > deadline - 10.0:
            return 1e10
        x, y, theta_c, theta_o = params
        s, valid = _compute_s_rigid(x, y, theta_c, theta_o, base_positions, base_angles)
        if valid:
            return s
        else:
            return s + 10.0
    
    x0 = np.array([0.0, 0.0, 0.0, 0.0])
    best_result = None
    best_s = float('inf')
    
    starts = [
        np.array([0.0, 0.0, 0.0, 0.0]),
        np.array([0.0, 0.0, 15.0, 0.0]),
        np.array([0.0, 0.0, 30.0, 0.0]),
        np.array([0.0, 0.0, 0.0, 15.0]),
        np.array([0.0, 0.0, 0.0, 30.0]),
        np.array([0.5, 0.5, 10.0, 10.0]),
        np.array([-0.5, 0.3, -15.0, 5.0]),
    ]
    
    for start in starts:
        if time.time() > deadline - 20.0:
            break
        try:
            res = minimize(objective, start, method='Nelder-Mead', options={'maxiter': 400, 'xatol': 1e-7, 'fatol': 1e-9})
            if res.fun < best_s:
                best_s = res.fun
                best_result = res.x
        except Exception:
            pass
            
    if best_result is not None:
        x, y, theta_c, theta_o = best_result
        s, valid = _compute_s_rigid(x, y, theta_c, theta_o, base_positions, base_angles)
        if valid:
            theta_c_rad = np.radians(theta_c)
            cos_c = np.cos(theta_c_rad)
            sin_c = np.sin(theta_c_rad)
            rot_positions = np.zeros_like(base_positions)
            rot_positions[:, 0] = base_positions[:, 0] * cos_c - base_positions[:, 1] * sin_c + x
            rot_positions[:, 1] = base_positions[:, 0] * sin_c + base_positions[:, 1] * cos_c + y
            angles = base_angles + theta_c
            
            inner = np.zeros((12, 3))
            inner[:, 0] = rot_positions[:, 0]
            inner[:, 1] = rot_positions[:, 1]
            inner[:, 2] = angles
            return inner, theta_o, s
    return None

def _generate_rigid_seeds(deadline):
    seeds = []
    L = 2 * np.sqrt(3)
    
    # Pattern 1: 3x4 triangular
    base_pos = []
    base_ang = []
    for r in range(3):
        for c in range(4):
            x = (c + r/2.0) * L
            y = r * L * np.sqrt(3)/2
            base_pos.append([x, y])
            base_ang.append(0.0)
    base_pos = np.array(base_pos)
    base_ang = np.array(base_ang)
    base_pos -= np.mean(base_pos, axis=0)
    
    res = _optimize_rigid_pattern(base_pos, base_ang, deadline)
    if res:
        seeds.append(res[0])
        
    # Pattern 2: 2x6 triangular
    base_pos = []
    base_ang = []
    for r in range(2):
        for c in range(6):
            x = (c + r/2.0) * L
            y = r * L * np.sqrt(3)/2
            base_pos.append([x, y])
            base_ang.append(0.0)
    base_pos = np.array(base_pos)
    base_ang = np.array(base_ang)
    base_pos -= np.mean(base_pos, axis=0)
    
    res = _optimize_rigid_pattern(base_pos, base_ang, deadline)
    if res:
        seeds.append(res[0])
        
    # Pattern 3: 4x3 triangular
    base_pos = []
    base_ang = []
    for r in range(4):
        for c in range(3):
            x = (c + r/2.0) * L
            y = r * L * np.sqrt(3)/2
            base_pos.append([x, y])
            base_ang.append(0.0)
    base_pos = np.array(base_pos)
    base_ang = np.array(base_ang)
    base_pos -= np.mean(base_pos, axis=0)
    
    res = _optimize_rigid_pattern(base_pos, base_ang, deadline)
    if res:
        seeds.append(res[0])
    
    # Pattern 4: Hex cluster (center + 6 + 5)
    base_pos = [[0.0, 0.0]]
    base_ang = [0.0]
    for k in range(6):
        angle = np.radians(60 * k)
        base_pos.append([L * np.cos(angle), L * np.sin(angle)])
        base_ang.append(0.0)
    second_ring = [(2, 0), (1, 1), (0, 2), (-1, 1), (-2, 0)]
    for (q, r) in second_ring:
        x = L * (q + r/2.0)
        y = L * r * np.sqrt(3)/2
        base_pos.append([x, y])
        base_ang.append(0.0)
    base_pos = np.array(base_pos)
    base_ang = np.array(base_ang)
    base_pos -= np.mean(base_pos, axis=0)
    
    res = _optimize_rigid_pattern(base_pos, base_ang, deadline)
    if res:
        seeds.append(res[0])
    
    return seeds

def hexagon_packing_12():
    start_time = time.time()
    deadline = start_time + 610.0  # Safety margin: 20s before 630s kill
    
    seeds = _generate_seeds()
    try:
        rigid_seeds = _generate_rigid_seeds(deadline)
        seeds.extend(rigid_seeds)
    except Exception:
        pass

    best_inner = None
    best_s = float('inf')
    best_outer_angle = 0.0
    
    # Evaluate all seeds with multiple outer angles
    for seed in seeds:
        for angle_deg in [0, 30, 15, 45, 60, -15, -30]:
            s, is_valid = _compute_s_and_overlap_fast(seed, angle_deg)
            if is_valid and s < best_s:
                best_s = s
                best_inner = seed.copy()
                best_outer_angle = angle_deg
    
    if best_inner is None:
        best_inner = seeds[0].copy()
        best_s, _ = _compute_s_and_overlap_fast(best_inner, 0)
        best_outer_angle = 0
    
    # Phase 1: Collision-resolution-based local search (simulated annealing)
    np.random.seed(42)
    current_inner = best_inner.copy()
    current_s = best_s
    current_outer_angle = best_outer_angle
    
    T_initial = 0.04
    T_min = 0.00005
    n_sa_iterations = 12000
    
    for it in range(n_sa_iterations):
        if time.time() > deadline - 180.0:
            break
        
        T = T_initial * (T_min / T_initial) ** (it / n_sa_iterations)
        new_inner = current_inner.copy()
        new_outer_angle = current_outer_angle
        
        move_type = np.random.rand()
        
        if move_type < 0.45:
            # Multi-hexagon perturbation with collision resolution
            n_perturb = np.random.choice([1, 2, 3], p=[0.35, 0.40, 0.25])
            indices = np.random.choice(12, size=n_perturb, replace=False)
            for idx in indices:
                new_inner[idx, 0] += np.random.randn() * 0.12
                new_inner[idx, 1] += np.random.randn() * 0.12
                new_inner[idx, 2] += np.random.randn() * 0.06
            # Resolve collisions to get a valid candidate
            new_inner = _resolve_collisions(new_inner, new_outer_angle, max_iters=40, push_factor=0.55)
        elif move_type < 0.75:
            # Small perturbation
            n_perturb = np.random.choice([1, 2], p=[0.6, 0.4])
            indices = np.random.choice(12, size=n_perturb, replace=False)
            for idx in indices:
                new_inner[idx, 0] += np.random.randn() * 0.05
                new_inner[idx, 1] += np.random.randn() * 0.05
                new_inner[idx, 2] += np.random.randn() * 0.025
            new_inner = _resolve_collisions(new_inner, new_outer_angle, max_iters=20, push_factor=0.4)
        else:
            # Single hexagon fine-tune
            idx = np.random.randint(0, 12)
            new_inner[idx, 2] += np.random.randn() * 0.08
            new_inner[idx, 0] += np.random.randn() * 0.03
            new_inner[idx, 1] += np.random.randn() * 0.03
            if np.random.rand() < 0.3:
                new_outer_angle += np.random.randn() * 0.05
            new_inner = _resolve_collisions(new_inner, new_outer_angle, max_iters=15, push_factor=0.35)
        
        new_s, new_valid = _compute_s_and_overlap_fast(new_inner, new_outer_angle)
        
        if new_valid:
            if new_s < current_s:
                current_inner = new_inner
                current_s = new_s
                current_outer_angle = new_outer_angle
                if current_s < best_s:
                    best_inner = current_inner.copy()
                    best_s = current_s
                    best_outer_angle = current_outer_angle
            else:
                delta = new_s - current_s
                if np.random.rand() < np.exp(-delta / max(T, 1e-10)):
                    current_inner = new_inner
                    current_s = new_s
                    current_outer_angle = new_outer_angle
        else:
            # Try harder collision resolution
            resolved = _resolve_collisions(new_inner, new_outer_angle, max_iters=60, push_factor=0.7)
            resolved_s, resolved_valid = _compute_s_and_overlap_fast(resolved, new_outer_angle)
            if resolved_valid and resolved_s < current_s:
                current_inner = resolved
                current_s = resolved_s
                current_outer_angle = new_outer_angle
                if current_s < best_s:
                    best_inner = current_inner.copy()
                    best_s = current_s
                    best_outer_angle = current_outer_angle
            elif resolved_valid and np.random.rand() < np.exp(-(resolved_s - current_s) / max(T, 1e-10)):
                current_inner = resolved
                current_s = resolved_s
                current_outer_angle = new_outer_angle
    
    # Phase 2: Nelder-Mead with smooth epigraph objective
    from scipy.optimize import minimize
    
    def objective(params):
        if time.time() > deadline - 60.0:
            return 1e10
        inner = params[:36].reshape(12, 3).copy()
        outer_angle = params[36]
        return _compute_s_smooth(inner, outer_angle)
    
    x0 = np.concatenate([best_inner.flatten(), [best_outer_angle]])
    
    for restart in range(8):
        if time.time() > deadline - 70.0:
            break
        if restart == 0:
            x_start = x0.copy()
        else:
            x_start = x0 + np.random.randn(37) * 0.01
        
        try:
            result = minimize(objective, x_start, method='Nelder-Mead',
                             options={'maxiter': 1500, 'xatol': 1e-9, 'fatol': 1e-11})
            candidate_inner = result.x[:36].reshape(12, 3).copy()
            candidate_angle = result.x[36]
            candidate_s, candidate_valid = _compute_s_and_overlap_fast(candidate_inner, candidate_angle)
            if candidate_valid and candidate_s < best_s:
                best_inner = candidate_inner
                best_s = candidate_s
                best_outer_angle = candidate_angle
                x0 = np.concatenate([best_inner.flatten(), [best_outer_angle]])
        except Exception:
            pass
    
    # Phase 3: Fine-grained collision-resolution local search
    if time.time() < deadline - 50.0:
        np.random.seed(123)
        current_inner = best_inner.copy()
        current_s = best_s
        current_outer_angle = best_outer_angle
        
        for it in range(10000):
            if time.time() > deadline - 50.0:
                break
            new_inner = current_inner.copy()
            new_outer_angle = current_outer_angle
            idx = np.random.randint(0, 12)
            new_inner[idx, 0] += np.random.randn() * 0.005
            new_inner[idx, 1] += np.random.randn() * 0.005
            new_inner[idx, 2] += np.random.randn() * 0.003
            if np.random.rand() < 0.25:
                new_outer_angle += np.random.randn() * 0.004
            
            # Resolve collisions
            new_inner = _resolve_collisions(new_inner, new_outer_angle, max_iters=20, push_factor=0.4)
            
            new_s, new_valid = _compute_s_and_overlap_fast(new_inner, new_outer_angle)
            if new_valid:
                if new_s < current_s:
                    current_inner = new_inner
                    current_s = new_s
                    current_outer_angle = new_outer_angle
                    if current_s < best_s:
                        best_inner = current_inner.copy()
                        best_s = current_s
                        best_outer_angle = current_outer_angle
                else:
                    delta = new_s - current_s
                    if np.random.rand() < np.exp(-delta / max(0.002, 1e-10)):
                        current_inner = new_inner
                        current_s = new_s
                        current_outer_angle = new_outer_angle
    
    # Phase 4: Final high-precision Nelder-Mead polish
    if time.time() < deadline - 25.0:
        def objective2(params):
            if time.time() > deadline - 10.0:
                return 1e10
            inner = params[:36].reshape(12, 3).copy()
            outer_angle = params[36]
            return _compute_s_smooth(inner, outer_angle)
        
        x0 = np.concatenate([best_inner.flatten(), [best_outer_angle]])
        for restart in range(5):
            if time.time() > deadline - 12.0:
                break
            if restart == 0:
                x_start = x0.copy()
            else:
                x_start = x0 + np.random.randn(37) * 0.004
            
            try:
                result = minimize(objective2, x_start, method='Nelder-Mead',
                                 options={'maxiter': 1000, 'xatol': 1e-10, 'fatol': 1e-12})
                candidate_inner = result.x[:36].reshape(12, 3).copy()
                candidate_angle = result.x[36]
                candidate_s, candidate_valid = _compute_s_and_overlap_fast(candidate_inner, candidate_angle)
                if candidate_valid and candidate_s < best_s:
                    best_inner = candidate_inner
                    best_s = candidate_s
                    best_outer_angle = candidate_angle
                    x0 = np.concatenate([best_inner.flatten(), [best_outer_angle]])
            except Exception:
                pass
    
    # Final validation
    s_final, valid_final = _compute_s_and_overlap_fast(best_inner, best_outer_angle)
    if not valid_final:
        # Fallback to first valid seed
        for seed in seeds:
            s_test, valid_test = _compute_s_and_overlap_fast(seed, 0)
            if valid_test:
                best_inner = seed.copy()
                best_s = s_test
                best_outer_angle = 0
                break
    
    outer_hex_data = np.array([0.0, 0.0, best_outer_angle])
    return best_inner, outer_hex_data, float(best_s)

# EVOLVE-BLOCK-END
