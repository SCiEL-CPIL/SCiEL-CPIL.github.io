# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time

def _hex_vertices(cx, cy, s, angle_deg):
    """Compute 6 vertices of a regular hexagon."""
    angles = np.radians(angle_deg + 60 * np.arange(6))
    return np.column_stack([cx + s * np.cos(angles), cy + s * np.sin(angles)])

def _hex_normals(vertices):
    """Compute 6 edge outward normals (unit) for a hexagon given its 6 CCW vertices."""
    n = 6
    normals = np.zeros((n, 2))
    for i in range(n):
        p1 = vertices[i]
        p2 = vertices[(i + 1) % n]
        edge = p2 - p1
        nx, ny = edge[1], -edge[0]
        norm = np.linalg.norm([nx, ny])
        if norm > 1e-12:
            normals[i] = [nx / norm, ny / norm]
    return normals

def _min_separation_gap(v1, n1, v2, n2):
    """Compute the minimum separation gap between two hexagons (negative means overlap)."""
    all_normals = np.vstack([n1, n2])
    min_gap = float('inf')
    # Vectorized projection
    proj1 = v1 @ all_normals.T # shape (12, 6)
    proj2 = v2 @ all_normals.T # shape (12, 6)
    
    min1 = proj1.min(axis=1)
    max1 = proj1.max(axis=1)
    min2 = proj2.min(axis=1)
    max2 = proj2.max(axis=1)
    
    # Gap for each normal direction
    # If max1 < min2, gap is min2 - max1. If max2 < min1, gap is min1 - max2.
    # We want the minimum positive gap, or the maximum negative overlap (closest to 0).
    # Actually, we just need to know if they are separated.
    # Separation exists if for ANY normal, the intervals are disjoint.
    # Disjoint if max1 < min2 - eps OR max2 < min1 - eps.
    # We want the "worst case" gap, i.e., the minimum gap over all normals.
    # Gap_i = max(0, min2 - max1, min1 - max2) ? No.
    # If they overlap, all gaps are negative? No.
    # If they overlap, for every normal, the intervals intersect.
    # The "separation" is defined by the existence of a separating axis.
    # The task says: "on at least one of their 12 edge normals the projected intervals are apart by more than 1e-6".
    # So we need to check if there exists an axis with gap > 1e-6.
    # For optimization, we want to maximize the minimum gap over all axes?
    # No, we want to ensure AT LEAST ONE axis has a gap.
    # However, usually in packing, we want to maximize the minimum clearance.
    # If we define penalty as sum of overlaps, we need a measure of "how much they overlap".
    # A common metric is the minimum gap over all 12 normals. If this min_gap > 0, they are separated.
    # If min_gap < 0, they are overlapping on all axes (which is impossible for convex sets? No, if they overlap, they project to intersecting intervals on ALL axes).
    # So min_gap < 0 implies overlap. min_gap > 0 implies separation.
    
    gaps = np.maximum(min2 - max1, min1 - max2)
    current_min_gap = np.min(gaps)
    return current_min_gap

def _all_pairs_separated(inner_hexes):
    """Check if all pairs of inner hexagons are separated."""
    n = len(inner_hexes)
    vertices_list = []
    normals_list = []
    for i in range(n):
        x, y, a = inner_hexes[i]
        v = _hex_vertices(x, y, 1.0, a)
        nm = _hex_normals(v)
        vertices_list.append(v)
        normals_list.append(nm)
    
    for i in range(n):
        for j in range(i + 1, n):
            gap = _min_separation_gap(vertices_list[i], normals_list[i], vertices_list[j], normals_list[j])
            if gap <= 1e-6:
                return False
    return True

def _outer_contains_inner(outer_center, outer_side, outer_angle, inner_hexes, tol=1e-6):
    """Check if all inner hexagon vertices are inside the outer hexagon."""
    outer_v = _hex_vertices(outer_center[0], outer_center[1], outer_side, outer_angle)
    
    for i in range(len(inner_hexes)):
        x, y, a = inner_hexes[i]
        v = _hex_vertices(x, y, 1.0, a)
        for p in v:
            for e in range(6):
                p1 = outer_v[e]
                p2 = outer_v[(e + 1) % 6]
                cross = (p2[0] - p1[0]) * (p[1] - p1[1]) - (p2[1] - p1[1]) * (p[0] - p1[0])
                if cross < -tol:
                    return False
    return True

def _compute_side_length(inner_hexes, outer_center, outer_angle):
    """Compute the minimum side length of outer hexagon that contains all inner hexagons."""
    outer_v_template = _hex_vertices(0, 0, 1.0, outer_angle)
    
    max_s = 0.0
    for e in range(6):
        p1 = outer_v_template[e]
        p2 = outer_v_template[(e + 1) % 6]
        edge = p2 - p1
        nx, ny = edge[1], -edge[0]
        norm = np.linalg.norm([nx, ny])
        if norm < 1e-12:
            continue
        n = np.array([nx / norm, ny / norm])
        
        for i in range(len(inner_hexes)):
            x, y, a = inner_hexes[i]
            v = _hex_vertices(x, y, 1.0, a)
            for p in v:
                dp = p - outer_center
                proj = dp @ n
                if proj > 0:
                    required_s = proj * 2.0 / np.sqrt(3.0)
                    max_s = max(max_s, required_s)
    
    return max_s

def _smooth_objective(params, outer_center, outer_angle, lambda_overlap=100.0):
    """
    Smooth objective: minimize side length + penalty for overlaps.
    """
    inner_hexes = params.reshape(11, 3)
    
    # Compute side length
    side = _compute_side_length(inner_hexes, outer_center, outer_angle)
    
    # Compute overlap penalty
    n = len(inner_hexes)
    vertices_list = []
    normals_list = []
    for i in range(n):
        x, y, a = inner_hexes[i]
        v = _hex_vertices(x, y, 1.0, a)
        nm = _hex_normals(v)
        vertices_list.append(v)
        normals_list.append(nm)
    
    overlap_penalty = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            gap = _min_separation_gap(vertices_list[i], normals_list[i], vertices_list[j], normals_list[j])
            # If gap < 0, they overlap. We penalize the square of the negative gap.
            # If gap > 0, they are separated.
            # We want to encourage separation, so we penalize overlap.
            if gap < 0:
                overlap_penalty += (-gap) ** 2
    
    # Check containment (hard constraint, but we can add a soft penalty if we want, 
    # but side length already forces containment if we define side correctly.
    # Wait, _compute_side_length calculates the side needed to contain them.
    # So if we minimize side, we are implicitly minimizing the container size.
    # The overlap penalty ensures they don't intersect.
    
    return side + lambda_overlap * overlap_penalty

def _objective(params, outer_center, outer_angle):
    """Objective: minimize the side length of the outer hexagon. Hard constraints."""
    inner_hexes = params.reshape(11, 3)
    
    if not _all_pairs_separated(inner_hexes):
        return 1e6
    
    side = _compute_side_length(inner_hexes, outer_center, outer_angle)
    
    if not _outer_contains_inner(outer_center, side, outer_angle, inner_hexes):
        return 1e6
    
    return side

def hexagon_packing_11():
    """
    Constructs a packing of 11 disjoint unit regular hexagons inside a larger regular hexagon.
    Uses smooth objective function with Nelder-Mead for local optimization from multiple starts.
    """
    start_time = time.time()
    deadline = start_time + 610.0
    
    outer_center = np.array([0.0, 0.0])
    best_params = None
    best_side = float('inf')
    best_angle = 0.0
    
    # Initial configurations
    # 1. Center + 6 ring + 4 ring (symmetric)
    sqrt3 = np.sqrt(3)
    inits = []
    
    # Config 1: Tight symmetric
    r1 = 2.0
    r2 = 3.5
    inits.append(np.array([
        [0, 0, 0],
        [r1, 0, 0],
        [r1/2, r1*sqrt3/2, 0],
        [-r1/2, r1*sqrt3/2, 0],
        [-r1, 0, 0],
        [-r1/2, -r1*sqrt3/2, 0],
        [r1/2, -r1*sqrt3/2, 0],
        [r2*3/2, r2*sqrt3/2, 0],
        [-r2*3/2, r2*sqrt3/2, 0],
        [r2*3/2, -r2*sqrt3/2, 0],
        [-r2*3/2, -r2*sqrt3/2, 0],
    ]))
    
    # Config 2: Slightly looser
    r1 = 2.1
    r2 = 3.8
    inits.append(np.array([
        [0, 0, 0],
        [r1, 0, 0],
        [r1/2, r1*sqrt3/2, 0],
        [-r1/2, r1*sqrt3/2, 0],
        [-r1, 0, 0],
        [-r1/2, -r1*sqrt3/2, 0],
        [r1/2, -r1*sqrt3/2, 0],
        [r2*3/2, r2*sqrt3/2, 0],
        [-r2*3/2, r2*sqrt3/2, 0],
        [r2*3/2, -r2*sqrt3/2, 0],
        [-r2*3/2, -r2*sqrt3/2, 0],
    ]))
    
    # Config 3: Randomized perturbations of Config 1
    for _ in range(5):
        base = inits[0].copy()
        base[:, :2] += np.random.normal(0, 0.1, (11, 2))
        inits.append(base)
        
    angles_to_try = [0.0, 30.0]
    
    for angle in angles_to_try:
        if time.time() > deadline - 100:
            break
            
        for init in inits:
            if time.time() > deadline - 50:
                break
            
            # Use smooth objective for Nelder-Mead
            # Nelder-Mead is robust to non-differentiable points but smooth helps.
            try:
                res = minimize(
                    _smooth_objective,
                    init.flatten(),
                    args=(outer_center, angle),
                    method='Nelder-Mead',
                    options={'maxiter': 2000, 'xatol': 1e-10, 'fatol': 1e-12, 'adaptive': True}
                )
                
                # Verify the result with hard constraints
                candidate = res.x.reshape(11, 3)
                if _all_pairs_separated(candidate):
                    side = _compute_side_length(candidate, outer_center, angle)
                    if _outer_contains_inner(outer_center, side, angle, candidate):
                        if side < best_side:
                            best_side = side
                            best_params = candidate.copy()
                            best_angle = angle
            except Exception:
                continue

    # Final polish with hard objective to ensure validity
    if best_params is not None and time.time() < deadline - 20:
        try:
            res2 = minimize(
                _objective,
                best_params.flatten(),
                args=(outer_center, best_angle),
                method='Nelder-Mead',
                options={'maxiter': 1000, 'xatol': 1e-10, 'fatol': 1e-12, 'adaptive': True}
            )
            candidate = res2.x.reshape(11, 3)
            if _all_pairs_separated(candidate):
                side = _compute_side_length(candidate, outer_center, best_angle)
                if _outer_contains_inner(outer_center, side, best_angle, candidate) and side < best_side:
                    best_side = side
                    best_params = candidate.copy()
        except Exception:
            pass

    # Fallback
    if best_params is None:
        best_params = inits[0]
        best_angle = 0.0
        best_side = _compute_side_length(best_params, outer_center, best_angle)

    inner_hex_data = best_params
    outer_hex_data = np.array([0.0, 0.0, float(best_angle)])
    outer_hex_side_length = float(best_side)
    
    return inner_hex_data, outer_hex_data, outer_hex_side_length

# EVOLVE-BLOCK-END
