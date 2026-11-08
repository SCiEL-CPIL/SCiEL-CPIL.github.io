<<<<<<< SEARCH
def _min_enclosing_hexagon(inner_positions, inner_angles, inner_sides):
    """
    Find minimal regular hexagon (center, side, angle) containing all inner hexagons.
    """
    # Collect all vertices of all inner hexagons
    all_verts = []
    for (cx, cy, ang), s in zip(inner_positions, inner_sides):
        verts = _hex_vertices(cx, cy, s, ang)
        all_verts.append(verts)
    all_verts = np.vstack(all_verts)
    
    best_side = float('inf')
    best_center = None
    best_angle = 0.0
    
    # The outer hexagon angle can be restricted to [0, 60) due to symmetry
    # Try many angles
    for phi_deg in np.linspace(0, 60, 1200):
        phi = np.radians(phi_deg)
        # Face normals of outer hexagon
        normals = []
        for i in range(6):
            theta = phi + i * np.pi / 3 + np.pi / 6
            normals.append([np.cos(theta), np.sin(theta)])
        normals = np.array(normals)  # shape (6, 2)
        
        # Compute projections
        projs = all_verts @ normals.T  # shape (num_verts, 6)
        max_projs = projs.max(axis=0)  # shape (6,)
        min_projs = projs.min(axis=0)  # shape (6,)
        widths = max_projs - min_projs  # shape (6,)
        
        # Required apothem: a >= max(widths) / 2
        required_apothem = np.max(widths) / 2
        s_candidate = required_apothem * 2 / np.sqrt(3)
        
        # Check feasibility and find center
        lo = max_projs - required_apothem
        hi = min_projs + required_apothem
        
        if np.any(lo > hi + 1e-12):
            continue
        
        # Find center c: we want dot(c, n_i) to be in [lo[i], hi[i]] for all i.
        # Use the midpoint of the intervals as the target projection.
        mid = (lo + hi) / 2
        
        # Solve for c from dot(c, n_i) = mid[i] for i=0..5
        # This is an overdetermined system. Use least squares.
        c, residuals, rank, sv = np.linalg.lstsq(normals, mid, rcond=None)
        
        # Verify the solution satisfies all constraints
        c_projs = normals @ c
        if np.any(c_projs < lo - 1e-9) or np.any(c_projs > hi + 1e-9):
            continue
        
        if s_candidate < best_side:
            best_side = s_candidate
            best_center = c
            best_angle = phi_deg
    
    if best_center is None:
        # Fallback: use centroid and a large side
        centroid = all_verts.mean(axis=0)
        return centroid[0], centroid[1], 100.0, 0.0
    
    return best_center[0], best_center[1], best_side, best_angle
=======
def _min_enclosing_hexagon(inner_positions, inner_angles, inner_sides):
    """
    Find minimal regular hexagon (center, side, angle) containing all inner hexagons.
    Uses support function: for a fixed outer hexagon orientation, the minimal
    enclosing hexagon has apothem = max over face normals of (max_proj - min_proj)/2,
    and the center is determined by the midpoint of the tightest constraint.
    """
    # Collect all vertices of all inner hexagons
    all_verts = []
    for (cx, cy, ang), s in zip(inner_positions, inner_sides):
        verts = _hex_vertices(cx, cy, s, ang)
        all_verts.append(verts)
    all_verts = np.vstack(all_verts)
    n_verts = len(all_verts)
    
    best_side = float('inf')
    best_center = None
    best_angle = 0.0
    
    sqrt3 = np.sqrt(3)
    
    def evaluate_orientation(phi_deg):
        """Evaluate the minimal enclosing hexagon for a given orientation."""
        phi = np.radians(phi_deg)
        # Face normals of outer hexagon (6 directions, 60 deg apart)
        normals = np.array([[np.cos(phi + i * np.pi / 3 + np.pi / 6),
                             np.sin(phi + i * np.pi / 3 + np.pi / 6)] for i in range(6)])
        
        # Compute projections of all vertices onto each normal
        projs = all_verts @ normals.T  # shape (n_verts, 6)
        max_projs = projs.max(axis=0)  # shape (6,)
        min_projs = projs.min(axis=0)  # shape (6,)
        widths = max_projs - min_projs  # shape (6,)
        
        # Required apothem: a >= max(widths) / 2
        required_apothem = np.max(widths) / 2.0
        s_candidate = required_apothem * 2.0 / sqrt3
        
        # Find center: dot(c, n_i) must be in [max_projs[i] - a, min_projs[i] + a]
        lo = max_projs - required_apothem
        hi = min_projs + required_apothem
        
        # Check feasibility
        if np.any(lo > hi + 1e-10):
            return None, None, s_candidate
        
        # Use least squares to find center
        mid = (lo + hi) / 2.0
        c, residuals, rank, sv = np.linalg.lstsq(normals, mid, rcond=None)
        
        # Verify
        c_projs = normals @ c
        if np.any(c_projs < lo - 1e-8) or np.any(c_projs > hi + 1e-8):
            return None, None, s_candidate
        
        return c, s_candidate, s_candidate
    
    # Strategy: The optimal orientation is one where at least two of the six
    # width constraints are simultaneously tight (binding). This happens when
    # the max and min projections along some normal pair are achieved by specific
    # vertices. We can find candidate angles by checking when a vertex's projection
    # onto a normal equals the max/min.
    
    # Phase 1: Coarse scan to find good region
    coarse_angles = np.linspace(0, 60, 360)
    coarse_best = float('inf')
    coarse_best_angle = 0.0
    
    for phi_deg in coarse_angles:
        c, s_cand, _ = evaluate_orientation(phi_deg)
        if c is not None and s_cand < coarse_best:
            coarse_best = s_cand
            coarse_best_angle = phi_deg
    
    # Phase 2: Fine scan around the best coarse angle
    fine_angles = np.linspace(max(0, coarse_best_angle - 3), min(60, coarse_best_angle + 3), 360)
    for phi_deg in fine_angles:
        c, s_cand, _ = evaluate_orientation(phi_deg)
        if c is not None and s_cand < best_side:
            best_side = s_cand
            best_center = c
            best_angle = phi_deg
    
    # Phase 3: Very fine polish
    if best_center is not None:
        ultra_fine = np.linspace(max(0, best_angle - 0.5), min(60, best_angle + 0.5), 200)
        for phi_deg in ultra_fine:
            c, s_cand, _ = evaluate_orientation(phi_deg)
            if c is not None and s_cand < best_side:
                best_side = s_cand
                best_center = c
                best_angle = phi_deg
    
    if best_center is None:
        # Fallback
        centroid = all_verts.mean(axis=0)
        max_r = np.max(np.linalg.norm(all_verts - centroid, axis=1))
        return centroid[0], centroid[1], max_r * 2 / sqrt3 + 1, 0.0
    
    return best_center[0], best_center[1], best_side, best_angle
>>>>>>> REPLACE

Now let me also improve the search over honeycomb patterns and add some optimization of the inner hexagon positions:

