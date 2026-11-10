<<<<<<< SEARCH
def hexagon_packing_12():
    """
    Constructs a packing of 12 disjoint unit regular hexagons inside a larger regular hexagon, maximizing 1/outer_hex_side_length.
    Uses a rigid-body approach: fix the relative positions of the 12 hexagons, then optimize
    the outer hexagon's angle and center offset to minimize the containing side length.
    """
    start_time = time.time()
    deadline = start_time + 600.0  # 600 second limit with safety margin
    
    # Get the fixed rigid body positions
    base_positions = _rigid_body_positions()
    
    # Verify no overlap in the base configuration
    if not _check_separation_fast(base_positions):
        # Fallback to a known valid configuration
        base_positions = np.array([
            [0, 0, 0],
            [1.5, 0, 0], [-1.5, 0, 0],
            [0.75, 1.5, 0], [-0.75, 1.5, 0],
            [0.75, -1.5, 0], [-0.75, -1.5, 0],
            [3.0, 0, 0], [-3.0, 0, 0],
            [1.5, 3.0, 0], [-1.5, 3.0, 0],
            [1.5, -3.0, 0], [-1.5, -3.0, 0],
        ])
    
    best_s = np.inf
    best_params = (0.0, 0.0, 0.0)  # (outer_angle_deg, outer_cx, outer_cy)
    
    # Coarse grid search over outer angle and center
    # The outer hexagon's angle has 60-degree symmetry, so search [0, 60)
    # Center offset: the cluster is roughly symmetric, so center should be near origin
    # Let's search center in [-2, 2] x [-2, 2] and angle in [0, 60)
    
    angle_range = np.linspace(0, 60, 61)  # 61 points, step 1 degree
    center_range = np.linspace(-2, 2, 17)  # 17 points, step 0.25
    
    # Coarse pass
    for angle in angle_range:
        if time.time() > deadline - 60:  # Leave 60s for refinement
            break
        for cx in center_range:
            for cy in center_range:
                s = _compute_s_for_params(base_positions, [cx, cy], angle)
                if s < best_s:
                    best_s = s
                    best_params = (angle, cx, cy)
    
    # Refine with local search around the best point found
    if time.time() < deadline - 30:
        # Local refinement
        best_angle, best_cx, best_cy = best_params
        
        # Nelder-Mead style local search
        step_sizes = [0.1, 0.05, 0.01, 0.005, 0.001]
        
        for step in step_sizes:
            if time.time() > deadline - 30:
                break
            improved = True
            while improved:
                improved = False
                # Try 8 neighbors
                neighbors = [
                    (best_angle + step, best_cx, best_cy),
                    (best_angle - step, best_cx, best_cy),
                    (best_angle, best_cx + step, best_cy),
                    (best_angle, best_cx - step, best_cy),
                    (best_angle, best_cx, best_cy + step),
                    (best_angle, best_cx, best_cy - step),
                    (best_angle + step, best_cx + step, best_cy),
                    (best_angle - step, best_cx - step, best_cy),
                ]
                for ang, cx, cy in neighbors:
                    if time.time() > deadline - 30:
                        break
                    s = _compute_s_for_params(base_positions, [cx, cy], ang)
                    if s < best_s - 1e-10:
                        best_s = s
                        best_angle, best_cx, best_cy = ang, cx, cy
                        improved = True
    
    # Final refinement with very small steps
    if time.time() < deadline - 10:
        for _ in range(500):
            if time.time() > deadline - 10:
                break
            step = 0.001
            neighbors = [
                (best_angle + step, best_cx, best_cy),
                (best_angle - step, best_cx, best_cy),
                (best_angle, best_cx + step, best_cy),
                (best_angle, best_cx - step, best_cy),
                (best_angle, best_cx, best_cy + step),
                (best_angle, best_cx, best_cy - step),
            ]
            for ang, cx, cy in neighbors:
                if time.time() > deadline - 10:
                    break
                s = _compute_s_for_params(base_positions, [cx, cy], ang)
                if s < best_s - 1e-12:
                    best_s = s
                    best_angle, best_cx, best_cy = ang, cx, cy
    
    # Construct the final result
    outer_angle, outer_cx, outer_cy = best_params
    
    # Inner hexagons: shift by minus outer center so that outer center is at origin for validation
    # Actually, the inner positions are absolute. The outer center is at (outer_cx, outer_cy).
    # The task says: "after shifting all centers by minus the outer center"
    # So the inner positions in the output are the absolute positions.
    
    inner_hex_data = base_positions.copy()
    outer_hex_data = np.array([outer_cx, outer_cy, outer_angle])
    
    # Validate
    if not _validate_result(inner_hex_data, outer_hex_data, best_s):
        # Try a fallback: center the outer hexagon at the centroid of inner hexagons
        centroid = np.mean(base_positions[:, :2], axis=0)
        outer_hex_data = np.array([centroid[0], centroid[1], 0.0])
        s = _compute_s_for_params(base_positions, centroid, 0.0)
        best_s = s
        if not _validate_result(inner_hex_data, outer_hex_data, best_s):
            # Last resort: use the seed program's approach
            inner_hex_data = np.array([
                [0, 0, 0],
                [-2.5, 0, 0], [2.5, 0, 0],
                [-1.25, 2.17, 0], [1.25, 2.17, 0],
                [-1.25, -2.17, 0], [1.25, -2.17, 0],
                [-3.75, 2.17, 0], [3.75, 2.17, 0],
                [-3.75, -2.17, 0], [3.75, -2.17, 0],
                [0, -4, 0],
            ])
            outer_hex_data = np.array([0, 0, 0])
            best_s = 8.0
    
    return inner_hex_data, outer_hex_data, best_s
=======
def _optimize_for_config(inner_positions, deadline):
    """
    Optimize outer hexagon parameters for a given inner configuration.
    Returns (best_s, best_params) where best_params = (angle, cx, cy).
    """
    if not _check_separation_fast(inner_positions):
        return np.inf, None
    
    best_s = np.inf
    best_params = None
    
    # Coarse grid search
    angle_range = np.linspace(0, 60, 31)
    center_range = np.linspace(-3, 3, 13)
    
    for angle in angle_range:
        if time.time() > deadline - 30:
            break
        for cx in center_range:
            if time.time() > deadline - 30:
                break
            for cy in center_range:
                if time.time() > deadline - 30:
                    break
                s = _min_containing_hex_side(
                    _all_inner_vertices(inner_positions), [cx, cy], angle
                )
                if s < best_s:
                    best_s = s
                    best_params = (angle, cx, cy)
    
    if best_params is None:
        return np.inf, None
    
    # Local optimization with scipy Nelder-Mead
    if time.time() < deadline - 20:
        def objective(params):
            angle, cx, cy = params
            verts = _all_inner_vertices(inner_positions)
            return _min_containing_hex_side(verts, [cx, cy], angle)
        
        x0 = np.array(best_params)
        result = minimize(
            objective, x0,
            method='Nelder-Mead',
            options={'maxiter': 500, 'xatol': 1e-8, 'fatol': 1e-10}
        )
        
        if result.fun < best_s:
            best_s = result.fun
            best_params = tuple(result.x)
    
    return best_s, best_params


def hexagon_packing_12():
    """
    Constructs a packing of 12 disjoint unit regular hexagons inside a larger regular hexagon.
    Tries multiple initial configurations and optimizes the outer hexagon for each.
    """
    start_time = time.time()
    deadline = start_time + 600.0
    
    # Generate multiple initial configurations
    configs = [
        _make_hex_lattice(12),
        _make_circular(12),
        _make_compact(12),
    ]
    
    best_overall_s = np.inf
    best_overall_inner = None
    best_overall_outer = None
    
    for config in configs:
        if time.time() > deadline - 60:
            break
        
        if not _check_separation_fast(config):
            continue
        
        s, params = _optimize_for_config(config, deadline)
        
        if params is not None and s < best_overall_s:
            best_overall_s = s
            best_overall_inner = config.copy()
            best_overall_outer = np.array([params[1], params[2], params[0]])
    
    # Fallback if no valid result
    if best_overall_inner is None:
        sqrt3 = np.sqrt(3.0)
        inner = np.array([
            [0, 0, 0],
            [sqrt3, 0, 0], [-sqrt3, 0, 0],
            [sqrt3/2, 1.5, 0], [-sqrt3/2, 1.5, 0],
            [sqrt3/2, -1.5, 0], [-sqrt3/2, -1.5, 0],
            [2*sqrt3, 0, 0], [-2*sqrt3, 0, 0],
            [sqrt3, 3.0, 0], [-sqrt3, 3.0, 0],
            [sqrt3, -3.0, 0], [-sqrt3, -3.0, 0],
        ])
        outer = np.array([0, 0, 0])
        s = _min_containing_hex_side(_all_inner_vertices(inner), [0, 0], 0)
        best_overall_s = s
        best_overall_inner = inner
        best_overall_outer = outer
    
    # Validate
    if not _validate_result(best_overall_inner, best_overall_outer, best_overall_s):
        centroid = np.mean(best_overall_inner[:, :2], axis=0)
        best_overall_outer = np.array([centroid[0], centroid[1], 0.0])
        shifted = best_overall_inner.copy()
        shifted[:, :2] -= centroid
        best_overall_s = _min_containing_hex_side(_all_inner_vertices(shifted), [0, 0], 0)
        
        if not _validate_result(best_overall_inner, best_overall_outer, best_overall_s):
            best_overall_s = max(best_overall_s, np.sqrt(12.0) + 1.0)
    
    return best_overall_inner, best_overall_outer, best_overall_s
>>>>>>> REPLACE

Now I need to add the `_all_inner_vertices` helper function:

