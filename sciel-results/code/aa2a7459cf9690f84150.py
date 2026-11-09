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
    
    # Pre-compute outer edge normals and offsets for point-in-polygon test
    # Edge k is from outer_verts[k] to outer_verts[(k+1)%6]
    # Normal pointing inwards
    outer_norms = np.zeros((6, 2))
    outer_offsets = np.zeros(6)
    for k in range(6):
        p1 = outer_verts[k]
        p2 = outer_verts[(k+1)%6]
        edge = p2 - p1
        # Inward normal
        n = np.array([edge[1], -edge[0]])
        norm_n = np.linalg.norm(n)
        if norm_n < 1e-9: return False
        n = n / norm_n
        outer_norms[k] = n
        outer_offsets[k] = np.dot(n, p1)

    # Check each inner hexagon
    for i in range(12):
        cx, cy, ang = inner[i]
        verts = _hex_vertices(cx, cy, 1.0, ang)
        
        # Check if all vertices are inside outer hexagon
        for v in verts:
            for k in range(6):
                # cross(p2-p1, v-p1) >= -1e-6
                # Equivalent to dot(inward_normal, v) <= dot(inward_normal, p1) + epsilon?
                # Let's stick to the cross product definition from the prompt to be safe
                p1 = outer_verts[k]
                p2 = outer_verts[(k+1)%6]
                cross = (p2[0] - p1[0]) * (v[1] - p1[1]) - (p2[1] - p1[1]) * (v[0] - p1[0])
                if cross < -1e-6:
                    return False

        # Check overlap with other inner hexagons
        for j in range(i+1, 12):
            cx2, cy2, ang2 = inner[j]
            verts2 = _hex_vertices(cx2, cy2, 1.0, ang2)
            
            # Quick distance check: if centers are far apart, skip SAT
            dist = np.linalg.norm(inner[i, :2] - inner[j, :2])
            if dist > 3.5: # Max distance for potential overlap is < 2*1 + 2*1 = 4? Actually, min dist for no overlap is 2*sqrt(3)?
                # For two unit hexagons, min center distance to be disjoint is 2*sqrt(3) if aligned?
                # Actually, if they touch at a vertex, dist is 2*sqrt(3) approx?
                # Let's just use SAT for all pairs to be safe, it's only 66 pairs.
                pass

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
        # Outward normal for containment check?
        # We want the max projection of vertices onto the outward normals of the unit hex.
        # The distance from center to edge of unit hex is sqrt(3)/2.
        # So s * (sqrt(3)/2) >= max_proj.
        # s >= max_proj * 2 / sqrt(3).
        
        normal = np.array([edge[1], -edge[0]])
        normal = normal / (np.linalg.norm(normal) + 1e-15)
        
        # Ensure normal points outward
        # Midpoint of edge is (p1+p2)/2. Dot with normal should be positive if outward.
        mid = (p1 + p2) / 2
        if np.dot(normal, mid) < 0:
            normal = -normal
            
        projs = all_verts @ normal
        max_k = np.max(projs)
        if max_k > max_proj:
            max_proj = max_k
    
    min_s = max_proj * 2.0 / np.sqrt(3)
    return max(min_s, np.sqrt(12))

def _generate_lattice_seed():
    """Generate a seed based on a triangular lattice."""
    # Triangular lattice basis
    # a1 = (1, 0), a2 = (0.5, sqrt(3)/2) scaled by factor L
    # We want to place 12 hexagons.
    # A 3x4 rectangle on a triangular lattice?
    # Or a hexagonal cluster?
    
    # Let's try a 3x4 grid in axial coordinates (q, r)
    # Axial to Cartesian: x = (q + r/2) * L, y = r * sqrt(3)/2 * L
    # We need to choose L such that hexagons don't overlap.
    # Min L for non-overlapping unit hexagons on triangular lattice:
    # Distance between neighbors is L.
    # Min center distance for non-overlapping unit hexagons is 2*sqrt(3) if aligned?
    # Actually, for two unit hexagons, the minimum distance between centers to be disjoint is 2*sqrt(3) if they are in a "flat" orientation?
    # Let's just use L = 2*sqrt(3) + epsilon to be safe, then optimize.
    
    L = 2 * np.sqrt(3) + 0.1
    
    # Generate 12 points in a compact shape
    # Shape: 3 rows of 4?
    # Axial coords for 3x4:
    # (0,0), (1,0), (2,0), (3,0)
    # (0,1), (1,1), (2,1), (3,1)
    # (0,2), (1,2), (2,2), (3,2)
    
    points = []
    for r in range(3):
        for q in range(4):
            x = (q + r/2.0) * L
            y = r * np.sqrt(3)/2 * L
            points.append([x, y, 0.0])
            
    # Center the cluster
    centers = np.array([p[:2] for p in points])
    center = np.mean(centers, axis=0)
    for p in points:
        p[0] -= center[0]
        p[1] -= center[1]
        
    return np.array(points)

def hexagon_packing_12():
    start_time = time.time()
    deadline = start_time + 600.0
    
    # Initial seeds
    seeds = []
    # 1. Lattice seed
    seeds.append(_generate_lattice_seed())
    
    # 2. Original grid seed
    initial_inner = np.array(
        [
            [0, 0, 0],
            [-2.5, 0, 0],
            [2.5, 0, 0],
            [-1.25, 2.1650635094610965, 0],
            [1.25, 2.1650635094610965, 0],
            [-1.25, -2.1650635094610965, 0],
            [1.25, -2.1650635094610965, 0],
            [-3.75, 2.1650635094610965, 0],
            [3.75, 2.1650635094610965, 0],
            [-3.75, -2.1650635094610965, 0],
            [3.75, -2.1650635094610965, 0],
            [0, -4.330127018922193, 0],
        ]
    )
    seeds.append(initial_inner)
    
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
                
    # Simulated Annealing
    current_inner = best_inner.copy()
    current_s = best_s
    current_outer_angle = best_outer_angle
    
    T_initial = 0.5
    T_min = 0.001
    cooling_rate = 0.9995
    n_iterations = 50000
    
    np.random.seed(42)
    
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
            new_inner[idx, 0] += np.random.randn() * 0.2
            new_inner[idx, 1] += np.random.randn() * 0.2
            new_inner[idx, 2] += np.random.randn() * 0.1
            
        # Perturb outer angle
        if np.random.rand() < 0.2:
            new_outer_angle += np.random.randn() * 0.1
            
        new_s = _compute_min_outer_side(new_inner, best_outer_center, new_outer_angle)
        
        # Check validity
        # Optimization: Only check full validity if new_s is promising
        # If new_s is much worse, skip validity check to save time?
        # No, we need to ensure we don't get stuck in invalid states.
        # But _is_valid_fast is still slow.
        
        # Heuristic: If new_s > current_s + 0.1, likely invalid or worse, skip detailed check?
        # Let's just check validity.
        
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
            else:
                delta = new_s - current_s
                if np.random.rand() < np.exp(-delta / max(T, 1e-10)):
                    current_inner = new_inner
                    current_s = new_s
                    current_outer_angle = new_outer_angle
        else:
            # Penalty based on overlap
            # Simple penalty: sum of squared distances < 2*sqrt(3)
            penalty = 0
            for i in range(12):
                for j in range(i+1, 12):
                    d = np.linalg.norm(new_inner[i, :2] - new_inner[j, :2])
                    min_d = 2 * np.sqrt(3) # Approx min dist
                    if d < min_d:
                        penalty += (min_d - d)**2
            if penalty < 0.5 and np.random.rand() < 0.1:
                current_inner = new_inner
                current_s = new_s + penalty
                current_outer_angle = new_outer_angle

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
            options={'maxiter': 100, 'xatol': 1e-5, 'fatol': 1e-7}
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
