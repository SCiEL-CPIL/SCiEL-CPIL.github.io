# EVOLVE-BLOCK-START
import numpy as np
import time

def _hex_vertices(cx, cy, d, angle_deg):
    """Compute 6 vertices of a regular hexagon with center (cx,cy), side d, angle in degrees."""
    a = np.radians(angle_deg)
    angles = a + np.radians(60) * np.arange(6)
    return np.column_stack([cx + d * np.cos(angles), cy + d * np.sin(angles)])

def _edge_normals(verts):
    """Compute outward unit normals for edges of a convex polygon given in CCW order."""
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

def _separation_check(verts1, normals1, verts2, normals2):
    """
    Check if two convex polygons are separated.
    Returns True if separated (valid), False if overlapping.
    Uses SAT: if any axis gives positive separation, they're separated.
    """
    all_normals = np.vstack([normals1, normals2])
    
    for normal in all_normals:
        proj1 = np.dot(verts1, normal)
        proj2 = np.dot(verts2, normal)
        
        min1, max1 = np.min(proj1), np.max(proj1)
        min2, max2 = np.min(proj2), np.max(proj2)
        
        # If separated along this axis, return True (valid)
        if max1 < min2 - 1e-6 or max2 < min1 - 1e-6:
            continue  # Not separated on this axis, keep checking
        # Actually, for validity we need NO axis to show separation > 1e-6
        # Wait, re-reading: "some edge normal must give projection intervals more than 1e-6 apart"
        # This means for VALID config, there must exist at least one separating axis with gap > 1e-6
        # No wait - re-reading again: "two inner hexagons not separated: some edge normal of either one (separating axis test) must give projection intervals more than 1e-6 apart"
        # This is saying the INVALID condition is: no separating axis exists with gap > 1e-6
        # So VALID means: at least one axis has gap > 1e-6
        # Hmm, that doesn't sound right for non-overlapping convex shapes...
        
        # Let me re-read: "two inner hexagons not separated: some edge normal of either one (separating axis test) must give projection intervals more than 1e-6 apart"
        # I think this means: the config is INVALID if for ALL axes, the projections overlap (gap <= 1e-6)
        # So VALID if there EXISTS an axis with gap > 1e-6
        
        pass
    
    # Correct interpretation: For two convex polygons, they are disjoint iff there exists a separating axis
    # The SAT theorem says: two convex polygons are disjoint iff there exists an axis (from edge normals) 
    # where their projections don't overlap.
    # So "not separated" means no separating axis exists, i.e., projections overlap on ALL axes.
    # Invalid if: for all axes, gap <= 1e-6
    # Valid if: exists an axis with gap > 1e-6
    
    for normal in all_normals:
        proj1 = np.dot(verts1, normal)
        proj2 = np.dot(verts2, normal)
        
        min1, max1 = np.min(proj1), np.max(proj1)
        min2, max2 = np.min(proj2), np.max(proj2)
        
        if max1 < min2 - 1e-6:
            return True  # Separated on this axis
        elif max2 < min1 - 1e-6:
            return True  # Separated on this axis
    
    return False  # No separating axis found with gap > 1e-6

def _compute_s_and_overlap_fast(inner, outer_angle):
    """
    Fast computation of required outer hexagon side length and overlap check.
    Returns (s, is_valid) where is_valid is True if no overlaps.
    """
    centers = inner[:, :2]
    angles = inner[:, 2]
    
    # Compute all inner vertices and normals
    all_verts = np.zeros((12, 6, 2))
    all_normals = np.zeros((12, 6, 2))
    
    for i in range(12):
        verts = _hex_vertices(centers[i, 0], centers[i, 1], 1.0, angles[i])
        all_verts[i] = verts
        all_normals[i] = _edge_normals(verts)
    
    # Compute s: maximum projection onto outer hexagon outward normals
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
    
    # Check overlaps using SAT
    is_valid = True
    for i in range(12):
        for j in range(i+1, 12):
            # Quick reject: if centers are far apart, definitely no overlap
            dist = np.linalg.norm(centers[i] - centers[j])
            if dist > 2.0 * np.sqrt(3) + 0.1:
                continue
            
            if not _separation_check(all_verts[i], all_normals[i], all_verts[j], all_normals[j]):
                is_valid = False
                break
        if not is_valid:
            break
    
    return s, is_valid

def _compute_s_and_overlap(inner, outer_angle):
    """Backward-compatible wrapper."""
    s, is_valid = _compute_s_and_overlap_fast(inner, outer_angle)
    penalty = 0.0 if is_valid else 1.0
    return s, penalty

def _generate_lattice_seed(n_per_ring):
    """Generate a hexagonal lattice seed with given number of rings."""
    # Use axial coordinates for hexagonal lattice
    # Spacing for unit hexagons: 2*sqrt(3) for flat-top, sqrt(3) for pointy-top
    # For flat-top hexagons (angle=0), horizontal spacing = 2*sqrt(3), vertical = 3/2 * 2*sqrt(3)
    # Actually for flat-top: width = 2*side, height = sqrt(3)*side
    # Adjacent centers: horizontal = 2*sqrt(3)*side, diagonal = sqrt(3)*side offset
    
    L = 2 * np.sqrt(3)  # Center-to-center distance for touching flat-top hexagons
    
    points = [[0.0, 0.0, 0.0]]  # Center
    
    # Ring 1: 6 hexagons
    for k in range(6):
        angle = np.radians(60 * k)
        points.append([L * np.cos(angle), L * np.sin(angle), 0.0])
    
    # Ring 2: 12 hexagons (but we only need 5 more to reach 12 total)
    # Axial coordinates for ring 2
    ring2_axial = [(2,0), (1,1), (0,2), (-1,1), (-2,0), (-1,-1), (0,-2), (1,-1), 
                   (2,-1), (1,-2), (-1,-2), (-2,-1)]
    
    # Convert axial to pixel for flat-top hexagons
    # x = size * sqrt(3) * (q + r/2)
    # y = size * 3/2 * r
    # But our L = 2*sqrt(3) corresponds to size=1 hexagons
    # So x = sqrt(3) * (q + r/2) * 2 = 2*sqrt(3)*(q+r/2)
    # y = 3/2 * r * 2 = 3*r
    
    # Wait, let me reconsider. For flat-top hexagons with side length s:
    # Width = 2s, Height = sqrt(3)*s
    # Horizontal spacing between columns = 3/2 * 2s = 3s... no
    # Let me just use the angle-based approach
    
    for (q, r) in ring2_axial[:5]:
        # Axial to pixel: x = L * (q + r/2), y = L * r * sqrt(3)/2
        # Actually for hex grid: x = spacing * (q + r/2), y = spacing * r * sqrt(3)/2
        # where spacing is the center-to-center distance
        x = L * (q + r/2.0)
        y = L * r * np.sqrt(3)/2
        points.append([x, y, 0.0])
    
    # Center the points
    pts = np.array(points)
    center = np.mean(pts[:, :2], axis=0)
    pts[:, 0] -= center[0]
    pts[:, 1] -= center[1]
    
    return pts

def _generate_seeds():
    """Generate multiple initial configurations."""
    seeds = []
    
    # Seed 1: Hexagonal cluster (center + 6 + 5)
    L = 2 * np.sqrt(3) + 0.01  # Small gap for numerical safety
    
    points = [[0.0, 0.0, 0.0]]
    for k in range(6):
        angle = np.radians(60 * k)
        points.append([L * np.cos(angle), L * np.sin(angle), 0.0])
    
    # Add 5 more in second ring
    second_ring = [
        (2, 0), (1, 1), (0, 2), (-1, 1), (-2, 0)
    ]
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
    L = 2 * np.sqrt(3) + 0.01
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
    L = 2 * np.sqrt(3) + 0.01
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
    
    # Seed 4: Hexagonal cluster with 30-degree rotation
    L = 2 * np.sqrt(3) + 0.01
    points = [[0.0, 0.0, 30.0]]
    for k in range(6):
        angle = np.radians(60 * k + 30)
        points.append([L * np.cos(angle), L * np.sin(angle), 30.0])
    
    second_ring = [
        (2, 0), (1, 1), (0, 2), (-1, 1), (-2, 0)
    ]
    for (q, r) in second_ring:
        x = L * (q + r/2.0)
        y = L * r * np.sqrt(3)/2
        points.append([x, y, 30.0])
    
    pts = np.array(points)
    center = np.mean(pts[:, :2], axis=0)
    pts[:, 0] -= center[0]
    pts[:, 1] -= center[1]
    seeds.append(pts)
    
    return seeds

def hexagon_packing_12():
    start_time = time.time()
    deadline = start_time + 620.0  # Leave 10s safety margin
    
    seeds = _generate_seeds()
    
    best_inner = None
    best_s = float('inf')
    best_outer_angle = 0.0
    
    # Evaluate all seeds with multiple outer angles
    for seed in seeds:
        for angle_deg in [0, 30, 15, 45]:
            s, is_valid = _compute_s_and_overlap_fast(seed, angle_deg)
            if is_valid:
                if s < best_s:
                    best_s = s
                    best_inner = seed.copy()
                    best_outer_angle = angle_deg
    
    if best_inner is None:
        # Fallback: use first seed
        best_inner = seeds[0].copy()
        best_s, _ = _compute_s_and_overlap(best_inner, 0)
        best_outer_angle = 0
    
    # Phase 1: Simulated Annealing
    current_inner = best_inner.copy()
    current_s = best_s
    current_outer_angle = best_outer_angle
    
    T_initial = 0.05
    T_min = 0.0001
    n_sa_iterations = 20000
    
    np.random.seed(42)
    
    it = 0
    while it < n_sa_iterations:
        if time.time() > deadline - 120.0:
            break
        
        it += 1
        T = T_initial * (T_min / T_initial) ** (it / n_sa_iterations)
        
        new_inner = current_inner.copy()
        new_outer_angle = current_outer_angle
        
        # Perturb 1-3 hexagons
        n_perturb = np.random.choice([1, 2, 3], p=[0.5, 0.35, 0.15])
        indices = np.random.choice(12, size=n_perturb, replace=False)
        for idx in indices:
            new_inner[idx, 0] += np.random.randn() * 0.08
            new_inner[idx, 1] += np.random.randn() * 0.08
            new_inner[idx, 2] += np.random.randn() * 0.03
        
        if np.random.rand() < 0.15:
            new_outer_angle += np.random.randn() * 0.03
        
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
            # Invalid: accept with small probability to escape local minima
            if np.random.rand() < 0.03:
                current_inner = new_inner
                current_s = new_s + 0.5  # Penalize
                current_outer_angle = new_outer_angle
    
    # Phase 2: Local optimization with Nelder-Mead
    from scipy.optimize import minimize
    
    def objective(params):
        if time.time() > deadline - 30.0:
            return 1e10
        inner = params[:36].reshape(12, 3).copy()
        outer_angle = params[36]
        
        s, is_valid = _compute_s_and_overlap_fast(inner, outer_angle)
        if is_valid:
            return s
        else:
            # Penalty for invalid configurations
            # Use a smooth penalty based on how much they overlap
            # For simplicity, just add a large penalty
            return s + 10.0
    
    x0 = np.concatenate([best_inner.flatten(), [best_outer_angle]])
    
    # Multiple restarts of Nelder-Mead
    for restart in range(5):
        if time.time() > deadline - 40.0:
            break
        
        if restart == 0:
            x_start = x0.copy()
        else:
            # Small perturbation for diversity
            x_start = x0 + np.random.randn(37) * 0.005
        
        try:
            result = minimize(
                objective, x_start,
                method='Nelder-Mead',
                options={'maxiter': 800, 'xatol': 1e-8, 'fatol': 1e-10}
            )
            
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
    
    # Phase 3: Fine-tuning with very small steps
    if time.time() < deadline - 50.0:
        T_fine = 0.005
        n_fine_iterations = 10000
        np.random.seed(123)
        
        current_inner = best_inner.copy()
        current_s = best_s
        current_outer_angle = best_outer_angle
        
        for it in range(n_fine_iterations):
            if time.time() > deadline - 50.0:
                break
            
            new_inner = current_inner.copy()
            new_outer_angle = current_outer_angle
            
            # Very small perturbations
            idx = np.random.randint(0, 12)
            new_inner[idx, 0] += np.random.randn() * 0.005
            new_inner[idx, 1] += np.random.randn() * 0.005
            new_inner[idx, 2] += np.random.randn() * 0.002
            
            if np.random.rand() < 0.2:
                new_outer_angle += np.random.randn() * 0.002
            
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
                    if np.random.rand() < np.exp(-delta / max(T_fine, 1e-10)):
                        current_inner = new_inner
                        current_s = new_s
                        current_outer_angle = new_outer_angle
    
    # Final validity check
    s_final, valid_final = _compute_s_and_overlap_fast(best_inner, best_outer_angle)
    if not valid_final:
        # Revert to seed
        best_inner = seeds[0].copy()
        best_s, _ = _compute_s_and_overlap(best_inner, 0)
        best_outer_angle = 0
    
    outer_hex_data = np.array([0.0, 0.0, best_outer_angle])
    return best_inner, outer_hex_data, float(best_s)

# EVOLVE-BLOCK-END
