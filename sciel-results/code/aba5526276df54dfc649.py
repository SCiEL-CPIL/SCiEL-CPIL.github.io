import numpy as np
import time

N = 12

def _hex_vertices(cx, cy, s, angle_deg):
    """Get vertices of a regular hexagon with side s, center (cx,cy), rotation angle_deg."""
    a = np.radians(angle_deg)
    verts = []
    for i in range(6):
        theta = a + i * np.pi / 3
        x = cx + s * np.cos(theta)
        y = cy + s * np.sin(theta)
        verts.append([x, y])
    return np.array(verts)

def _hex_face_normals(angle_deg):
    """Get the 6 outward face normals of a regular hexagon with rotation angle_deg."""
    a = np.radians(angle_deg)
    normals = []
    for i in range(6):
        theta = a + i * np.pi / 3 + np.pi / 6  # face normals are offset by 30 deg
        normals.append([np.cos(theta), np.sin(theta)])
    return np.array(normals)

def _min_enclosing_hexagon(inner_positions, inner_angles, inner_sides, start_time=None, deadline=None):
    """
    Find minimal regular hexagon (center, side, angle) containing all inner hexagons.
    Uses support function: for a given outer angle, the required apothem is max projection width / 2.
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
    # First pass: coarse search
    n_coarse = 360
    for phi_deg in np.linspace(0, 60, n_coarse):
        if start_time is not None and time.time() > deadline:
            break
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
        
        if s_candidate >= best_side:
            continue
        
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
        
        best_side = s_candidate
        best_center = c
        best_angle = phi_deg
    
    if best_center is None:
        # Fallback: use centroid and a large side
        centroid = all_verts.mean(axis=0)
        return centroid[0], centroid[1], 100.0, 0.0
    
    # Second pass: fine search around best_angle
    if start_time is None or time.time() < deadline - 1.0:
        search_range = 1.5  # degrees
        n_fine = 100
        for phi_deg in np.linspace(max(0, best_angle - search_range), min(60, best_angle + search_range), n_fine):
            if start_time is not None and time.time() > deadline:
                break
            phi = np.radians(phi_deg)
            normals = []
            for i in range(6):
                theta = phi + i * np.pi / 3 + np.pi / 6
                normals.append([np.cos(theta), np.sin(theta)])
            normals = np.array(normals)
            
            projs = all_verts @ normals.T
            max_projs = projs.max(axis=0)
            min_projs = projs.min(axis=0)
            widths = max_projs - min_projs
            
            required_apothem = np.max(widths) / 2
            s_candidate = required_apothem * 2 / np.sqrt(3)
            
            if s_candidate >= best_side:
                continue
            
            lo = max_projs - required_apothem
            hi = min_projs + required_apothem
            
            if np.any(lo > hi + 1e-12):
                continue
            
            mid = (lo + hi) / 2
            c, residuals, rank, sv = np.linalg.lstsq(normals, mid, rcond=None)
            
            c_projs = normals @ c
            if np.any(c_projs < lo - 1e-9) or np.any(c_projs > hi + 1e-9):
                continue
            
            best_side = s_candidate
            best_center = c
            best_angle = phi_deg
    
    return best_center[0], best_center[1], best_side, best_angle

def construct():
    start_time = time.time()
    deadline = start_time + 125.0  # safety margin
    
    s = 1.0  # unit hexagons
    sqrt3 = np.sqrt(3)
    
    best_result = None
    best_score = float('inf')
    
    # Generate honeycomb positions with a small gap to ensure disjointness
    scale = 1.001
    
    def honeycomb_positions(count, pattern, rot_offset=0.0):
        """Generate positions for 'count' hexagons in honeycomb pattern."""
        positions = []
        for j, row_len in enumerate(pattern):
            for i in range(row_len):
                x = (i + 0.5 * (j % 2)) * sqrt3 * scale
                y = j * 1.5 * scale
                positions.append((x, y, rot_offset))
        return positions[:count]
    
    # Try several row patterns that sum to 12, prioritizing compact ones
    patterns = [
        [4, 4, 4],
        [3, 4, 3, 2],
        [4, 3, 3, 2],
        [2, 3, 4, 3],
        [3, 3, 4, 2],
        [4, 4, 2, 2],
        [2, 2, 4, 4],
        [5, 3, 2, 2],
        [2, 2, 3, 5],
        [6, 3, 3],
        [3, 3, 6],
        [2, 4, 4, 2],
        [4, 2, 4, 2],
        [2, 4, 2, 4],
        [3, 5, 4],
        [4, 5, 3],
        [5, 4, 3],
        [3, 4, 5],
        [3, 3, 3, 3],
        [4, 3, 2, 3],
        [3, 2, 3, 4],
        [2, 3, 3, 4],
    ]
    
    # Try different inner hexagon orientations
    inner_rotations = [30.0, 0.0, 15.0, 45.0]
    
    for pattern in patterns:
        if sum(pattern) != 12:
            continue
        if len(pattern) > 6:
            continue
        
        for rot in inner_rotations:
            if time.time() > deadline:
                break
            
            positions = honeycomb_positions(12, pattern, rot)
            if len(positions) != 12:
                continue
            
            inner = [(x, y, r) for (x, y, r) in positions]
            inner_sides = [1.0] * 12
            
            cx, cy, side, angle = _min_enclosing_hexagon(inner, [r[2] for r in inner], inner_sides, start_time, deadline)
            
            # Add small safety margin for numerical precision
            side_safe = side * 1.0001 + 1e-6
            
            if side_safe < best_score:
                best_score = side_safe
                best_result = (inner, (cx, cy), side_safe, angle)
            
            if time.time() > deadline:
                break
    
    if best_result is None:
        # Fallback: linear arrangement
        inner = [(4.0 * k, 0.0, 0.0) for k in range(-6, 6)]
        return inner, (0.0, 0.0), 40.0, 0.0
    
    return best_result
