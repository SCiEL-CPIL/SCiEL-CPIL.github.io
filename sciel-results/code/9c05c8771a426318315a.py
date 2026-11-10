# EVOLVE-BLOCK-START
import numpy as np
import time

def _hex_vertices(cx, cy, s, angle_deg):
    """Compute 6 vertices of a regular hexagon."""
    a = np.radians(angle_deg)
    angles = a + np.radians(60 * np.arange(6))
    return np.column_stack([cx + s * np.cos(angles), cy + s * np.sin(angles)])


def _min_outer_side(inner_hex_data):
    """
    Given inner hexagons (11,3) with (x,y,angle_deg), compute the minimum
    outer hexagon side length needed to contain all of them, centered at origin
    with angle 0.
    """
    # Outer hexagon centered at origin, angle 0, side length S.
    # Vertices at S*(cos(60i), sin(60i)) for i=0..5
    # The 6 edge normals (pointing inward) are at angles 30, 90, 150, 210, 270, 330 degrees
    # For a point p to be inside: dot(p, n_k) <= S * apothem = S * sqrt(3)/2 for each edge normal
    # Actually for a regular hexagon with side S, the apothem is S*sqrt(3)/2
    # The constraint is: for each of the 6 edges, the projection of every vertex of every inner hex
    # onto the outward normal of that edge must be <= apothem.
    
    # Edge normals (outward) for a hexagon with angle 0:
    # Edge i is between vertex i and vertex i+1. The outward normal of edge i is at angle (60*i + 30) degrees.
    edge_normal_angles = np.radians(30 + 60 * np.arange(6))
    normals = np.column_stack([np.cos(edge_normal_angles), np.sin(edge_normal_angles)])
    
    # Compute all vertices of all inner hexagons
    all_vertices = []
    for i in range(len(inner_hex_data)):
        x, y, ang = inner_hex_data[i]
        verts = _hex_vertices(x, y, 1.0, ang)
        all_vertices.append(verts)
    all_vertices = np.vstack(all_vertices)  # (66, 2)
    
    # For each edge normal, compute max projection
    # Constraint: max_projection <= S * sqrt(3)/2
    # So S >= max_projection / (sqrt(3)/2) = 2 * max_projection / sqrt(3)
    projections = all_vertices @ normals.T  # (66, 6)
    max_projs = np.max(projections, axis=0)  # (6,)
    
    S = 2.0 * np.max(max_projs) / np.sqrt(3.0)
    return S


def _check_no_overlap(inner_hex_data, tol=1e-6):
    """
    Check that no two inner hexagons overlap.
    Two convex polygons overlap if and only if there exists an edge normal
    (of either polygon) such that their projections overlap.
    If for all 12 edge normals (6 from each), the projections are separated, no overlap.
    """
    n = len(inner_hex_data)
    for i in range(n):
        for j in range(i + 1, n):
            vi = _hex_vertices(inner_hex_data[i, 0], inner_hex_data[i, 1], 1.0, inner_hex_data[i, 2])
            vj = _hex_vertices(inner_hex_data[j, 0], inner_hex_data[j, 1], 1.0, inner_hex_data[j, 2])
            
            # Get edge normals for both hexagons
            def get_normals(verts):
                normals = []
                for k in range(6):
                    e = verts[(k+1) % 6] - verts[k]
                    n = np.array([-e[1], e[0]])
                    n = n / np.linalg.norm(n)
                    normals.append(n)
                return np.array(normals)
            
            ni = get_normals(vi)
            nj = get_normals(vj)
            all_normals = np.vstack([ni, nj])
            
            overlap = False
            for normal in all_normals:
                proj_i = vi @ normal
                proj_j = vj @ normal
                # Intervals [min_i, max_i] and [min_j, max_j]
                # They are separated if max_i < min_j - tol or max_j < min_i - tol
                if proj_i.max() < proj_j.min() - tol or proj_j.max() < proj_i.min() - tol:
                    continue  # separated on this normal
                else:
                    overlap = True
                    break
            
            if overlap:
                return False
    return True


def _is_valid(inner_hex_data):
    """Full validity check."""
    # All finite
    if not np.all(np.isfinite(inner_hex_data)):
        return False
    
    # Check no overlap
    if not _check_no_overlap(inner_hex_data):
        return False
    
    # Check containment: compute outer side, then verify all vertices are inside
    S = _min_outer_side(inner_hex_data)
    if S < np.sqrt(11) - 1e-9:
        return False
    
    # Verify containment with the computed S
    outer_center = np.array([0.0, 0.0])
    outer_angle = 0.0
    outer_verts = _hex_vertices(0, 0, S, outer_angle)
    
    # For each inner vertex, check it's inside the outer hexagon
    # A point is inside a convex polygon if it's on the same side of each edge
    for i in range(len(inner_hex_data)):
        verts = _hex_vertices(inner_hex_data[i, 0], inner_hex_data[i, 1], 1.0, inner_hex_data[i, 2])
        for p in verts:
            for k in range(6):
                p1 = outer_verts[k]
                p2 = outer_verts[(k+1) % 6]
                edge = p2 - p1
                normal = np.array([-edge[1], edge[0]])
                normal = normal / np.linalg.norm(normal)
                # The normal should point inward. Check orientation.
                # For a CCW polygon, inward normal is to the left of the edge direction.
                # cross(p - p1, edge) should be >= -tol for CCW polygon
                cross_val = (p[1] - p1[1]) * edge[0] - (p[0] - p1[0]) * edge[1]
                # Wait, the task says: cross(p2 - p1, p - p1) >= -1e-6
                # cross(p2-p1, p-p1) = (p2[0]-p1[0])*(p[1]-p1[1]) - (p2[1]-p1[1])*(p[0]-p1[0])
                cross_val2 = edge[0] * (p[1] - p1[1]) - edge[1] * (p[0] - p1[0])
                if cross_val2 < -1e-6:
                    return False
    
    # Check edge lengths are within 1e-8 of 1
    for i in range(len(inner_hex_data)):
        verts = _hex_vertices(inner_hex_data[i, 0], inner_hex_data[i, 1], 1.0, inner_hex_data[i, 2])
        for k in range(6):
            e = verts[(k+1) % 6] - verts[k]
            if abs(np.linalg.norm(e) - 1.0) > 1e-8:
                return False
    
    return True


def _objective(inner_hex_data):
    """Return 1/side if valid, else -inf."""
    try:
        S = _min_outer_side(inner_hex_data)
        if S < np.sqrt(11):
            return -np.inf
        if not _check_no_overlap(inner_hex_data):
            return -np.inf
        # Quick containment check
        outer_verts = _hex_vertices(0, 0, S, 0.0)
        for i in range(len(inner_hex_data)):
            verts = _hex_vertices(inner_hex_data[i, 0], inner_hex_data[i, 1], 1.0, inner_hex_data[i, 2])
            for p in verts:
                for k in range(6):
                    p1 = outer_verts[k]
                    p2 = outer_verts[(k+1) % 6]
                    edge = p2 - p1
                    cross_val = edge[0] * (p[1] - p1[1]) - edge[1] * (p[0] - p1[0])
                    if cross_val < -1e-6:
                        return -np.inf
        return 1.0 / S
    except:
        return -np.inf


def _random_initial_config(rng):
    """Generate a random initial configuration of 11 unit hexagons."""
    configs = []
    # Strategy 1: Hexagonal grid with some randomness
    # Standard hex grid positions for 11 hexagons (center + 6 ring + 4 more)
    base_positions = [
        (0, 0),
        (2.5, 0), (-2.5, 0),
        (1.25, 2.165), (-1.25, 2.165),
        (1.25, -2.165), (-1.25, -2.165),
        (3.75, 2.165), (-3.75, 2.165),
        (3.75, -2.165), (-3.75, -2.165),
    ]
    noise_scale = rng.uniform(0.0, 0.3)
    pos = np.array(base_positions, dtype=float)
    pos += rng.normal(0, noise_scale, size=(11, 2))
    angles = rng.uniform(0, 60, size=11)
    configs.append(np.column_stack([pos, angles]))
    
    # Strategy 2: More spread out
    pos2 = np.array(base_positions, dtype=float) * rng.uniform(0.9, 1.1)
    pos2 += rng.normal(0, 0.2, size=(11, 2))
    angles2 = rng.uniform(0, 60, size=11)
    configs.append(np.column_stack([pos2, angles2]))
    
    # Strategy 3: Random perturbation of a compact arrangement
    # Try a 3-4-3-3-3 or similar arrangement
    pos3 = np.array([
        (0, 0),
        (2.5, 0), (-2.5, 0),
        (1.25, 2.165), (-1.25, 2.165), (0, 4.33),
        (1.25, -2.165), (-1.25, -2.165), (0, -4.33),
        (3.75, 2.165), (-3.75, 2.165),
    ], dtype=float)
    pos3 += rng.normal(0, 0.15, size=(11, 2))
    angles3 = rng.uniform(0, 60, size=11)
    configs.append(np.column_stack([pos3, angles3]))
    
    return rng.choice(configs)


def _coordinate_descent(inner_hex_data, rng, max_iter=500, step_init=0.1):
    """Simple coordinate descent to minimize outer side length."""
    best = inner_hex_data.copy()
    best_obj = _objective(best)
    step = step_init
    no_improve = 0
    
    for it in range(max_iter):
        improved = False
        for i in range(11):
            for dim in range(3):  # x, y, angle
                for direction in [1, -1]:
                    trial = best.copy()
                    trial[i, dim] += direction * step
                    if dim == 2:  # angle
                        trial[i, dim] = trial[i, dim] % 360
                    obj = _objective(trial)
                    if obj > best_obj:
                        best = trial
                        best_obj = obj
                        improved = True
        
        if not improved:
            no_improve += 1
            step *= 0.7
            if step < 1e-5:
                break
        else:
            no_improve = 0
    
    return best, best_obj


def _refine_with_optim(inner_hex_data, rng, time_budget=30):
    """Use scipy optimize if available, with time budget."""
    start_time = time.time()
    
    try:
        from scipy.optimize import minimize
    except ImportError:
        return inner_hex_data, _objective(inner_hex_data)
    
    best = inner_hex_data.copy()
    best_obj = _objective(best)
    
    def neg_obj(params):
        if time.time() - start_time > time_budget:
            return 1e10
        hex_data = params.reshape(11, 3)
        return -_objective(hex_data)
    
    # Try multiple random restarts
    for restart in range(5):
        if time.time() - start_time > time_budget - 5:
            break
        
        x0 = best.copy()
        if restart > 0:
            x0 += rng.normal(0, 0.1, size=(11, 3))
        
        try:
            result = minimize(
                neg_obj,
                x0.flatten(),
                method='Nelder-Mead',
                options={'maxiter': 200, 'xatol': 1e-6, 'fatol': 1e-8}
            )
        except:
            continue
        
        hex_data = result.x.reshape(11, 3)
        obj = _objective(hex_data)
        if obj > best_obj:
            best = hex_data
            best_obj = obj
    
    return best, best_obj


def hexagon_packing_11():
    """
    Constructs a packing of 11 disjoint unit regular hexagons inside a larger regular hexagon,
    maximizing 1/outer_hex_side_length.
    """
    start_time = time.time()
    deadline = 580  # seconds, leave margin
    
    rng = np.random.default_rng(42)
    
    best_config = None
    best_score = -np.inf
    
    # Generate diverse initial configurations and refine each
    n_configs = 30
    for i in range(n_configs):
        if time.time() > deadline - 60:
            break
        
        config = _random_initial_config(rng)
        
        # Coordinate descent
        refined, score = _coordinate_descent(config, rng, max_iter=200, step_init=0.1)
        
        if score > best_score:
            best_score = score
            best_config = refined.copy()
        
        # If time allows, do scipy refinement on the best so far
        if i % 5 == 0 and time.time() < deadline - 90:
            refined2, score2 = _refine_with_optim(best_config, rng, time_budget=30)
            if score2 > best_score:
                best_score = score2
                best_config = refined2.copy()
    
    # Final refinement on the best
    if best_config is not None and time.time() < deadline - 30:
        refined_final, score_final = _coordinate_descent(best_config, rng, max_iter=300, step_init=0.05)
        if score_final > best_score:
            best_score = score_final
            best_config = refined_final.copy()
    
    if best_config is None:
        # Fallback to the seed configuration
        best_config = np.array(
            [
                [0, 0, 0],
                [-2.5, 0, 0],
                [2.5, 0, 0],
                [-1.25, 2.17, 0],
                [1.25, 2.17, 0],
                [-1.25, -2.17, 0],
                [1.25, -2.17, 0],
                [-3.75, 2.17, 0],
                [3.75, 2.17, 0],
                [-3.75, -2.17, 0],
                [3.75, -2.17, 0],
            ]
        )
    
    # Compute final outer side
    S = _min_outer_side(best_config)
    
    inner_hex_data = best_config
    outer_hex_data = np.array([0.0, 0.0, 0.0])
    outer_hex_side_length = float(S)
    
    return inner_hex_data, outer_hex_data, outer_hex_side_length


# EVOLVE-BLOCK-END
