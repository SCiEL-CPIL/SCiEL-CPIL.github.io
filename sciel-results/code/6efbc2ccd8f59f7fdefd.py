"""Optimized hexagon packing: 11 unit hexagons in a regular hexagon."""
import numpy as np
import time
import itertools

N = 11
DEADLINE = 140.0  # safety margin before 150s kill

def _hex_vertices(cx, cy, side, angle_deg):
    """Return 6 vertices of a regular hexagon centered at (cx,cy)."""
    angle = np.radians(angle_deg)
    verts = np.zeros((6, 2))
    for i in range(6):
        theta = angle + np.pi * i / 3.0
        verts[i, 0] = cx + side * np.cos(theta)
        verts[i, 1] = cy + side * np.sin(theta)
    return verts

def _all_inner_vertices(centers, angles):
    """Get all 6N vertices of inner hexagons."""
    all_verts = []
    for i in range(N):
        verts = _hex_vertices(centers[i, 0], centers[i, 1], 1.0, angles[i])
        all_verts.append(verts)
    return np.vstack(all_verts)  # shape (6N, 2)

def _min_enclosing_hexagon(params, angle_deg):
    """Given a set of 2D points and hexagon orientation, find min enclosing regular hexagon.
    Returns (center, side_length) or None if impossible."""
    # params: (6N, 2) array of points
    angle = np.radians(angle_deg)
    # Edge normals for regular hexagon with this orientation
    normals = []
    for i in range(6):
        theta = angle + np.pi / 6 + np.pi * i / 3.0
        normals.append((np.cos(theta), np.sin(theta)))
    normals = np.array(normals)  # (6, 2)
    
    # Project all points onto each normal
    # projections shape: (6, 6N)
    projections = params @ normals.T  # (6N, 6) -> transpose to (6, 6N)
    
    # For each pair of opposite edges (i, i+3), compute required inradius
    # The inradius must be >= max(proj_i) - min(proj_i+3) / 2 ... 
    # Actually for a hexagon centered at c with inradius r:
    # For normal n_i: n_i . p <= n_i . c + r for all p
    # For normal n_{i+3} = -n_i: -n_i . p <= -n_i . c + r => n_i . p >= n_i . c - r
    # So: n_i . c - r <= min_p(n_i . p) and max_p(n_i . c) <= n_i . c + r
    # Thus: r >= (max_p(n_i . p) - min_p(n_i . p)) / 2
    # And: n_i . c must be in [min_p(n_i . p) + r, max_p(n_i . p) - r]
    # For feasibility: min_p(n_i . p) + r <= max_p(n_i . p) - r => r >= span/2
    
    # For each of 3 pairs of opposite normals, compute required r
    r_required = 0.0
    c_bounds = []  # list of (low, high) for c in each normal direction
    for i in range(3):
        n_i = normals[i]
        n_opposite = normals[i + 3]  # = -n_i
        
        proj_i = params @ n_i
        proj_opposite = params @ n_opposite
        
        span = proj_i.max() - proj_i.min()
        r_req = span / 2.0
        r_required = max(r_required, r_req)
        
        # c must satisfy: n_i . c in [proj_i.min() + r, proj_i.max() - r]
        # But r is the same for all, so we need to find c that satisfies all 3 pairs
        # Let's collect constraints
        c_bounds.append((proj_i.min(), proj_i.max(), n_i))
    
    # Now we need to find center c such that for each pair i:
    # n_i . c in [proj_i.min() + r, proj_i.max() - r]
    # where r = r_required (or larger)
    # This is a linear feasibility problem in 2D
    
    # Use the fact that we can express c = c0 + alpha * d1 + beta * d2
    # and solve for alpha, beta
    
    # Simpler: try to find c by intersecting half-planes
    # For each of 3 normal pairs, we get 2 half-plane constraints on c
    # Total 6 half-planes in 2D -> find feasible point
    
    # Let's use a simple approach: the center should be at the "center" of the feasible region
    # For each normal pair, the midpoint of the projection range gives a candidate
    
    # Actually, let's just compute the center as the average of midpoints
    # and verify it works
    
    # Better: solve using the fact that optimal c minimizes the max over i of (n_i . c - proj_i.min())
    # subject to n_i . c <= proj_i.max() - r for all i
    
    # Simplest: try center at centroid of all points, then adjust
    # Or: for each pair, the optimal c in that direction is at the midpoint
    
    # Let's compute the feasible region intersection
    # Constraints: for i in 0,1,2:
    #   proj_i.min() + r <= n_i . c <= proj_i.max() - r
    
    # This defines a hexagonal region (or empty) in the plane
    # Find a point in this region
    
    # Use linear programming or simple bisection
    # For 2D, we can check if the intersection of 6 half-planes is non-empty
    
    # Let's try the centroid first
    c_candidate = np.mean(params, axis=0)
    
    # Check if c_candidate satisfies all constraints
    valid = True
    for i in range(3):
        n_i = normals[i]
        proj_i = params @ n_i
        c_proj = c_candidate @ n_i
        if c_proj < proj_i.min() + r_required - 1e-9 or c_proj > proj_i.max() - r_required + 1e-9:
            valid = False
            break
    
    if not valid:
        # Try to find a valid center by solving the constraints
        # Use scipy or manual approach
        # For now, try a few candidate centers
        # The center should be at the intersection of the "middle" lines
        # For each pair, the middle line is n_i . c = (proj_i.min() + proj_i.max()) / 2
        # Solve these 3 equations (take 2 independent ones)
        
        # Solve n_0 . c = m_0 and n_1 . c = m_1
        m = []
        for i in range(3):
            proj_i = params @ normals[i]
            m.append((proj_i.min() + proj_i.max()) / 2.0)
        
        A = np.array([normals[0], normals[1]])
        b = np.array([m[0], m[1]])
        
        try:
            c_candidate = np.linalg.solve(A, b)
        except:
            # If singular, try other pairs
            A = np.array([normals[0], normals[2]])
            b = np.array([m[0], m[2]])
            try:
                c_candidate = np.linalg.solve(A, b)
            except:
                return None
        
        # Verify
        for i in range(3):
            n_i = normals[i]
            proj_i = params @ n_i
            c_proj = c_candidate @ n_i
            if c_proj < proj_i.min() + r_required - 1e-6 or c_proj > proj_i.max() - r_required + 1e-6:
                return None
    
    side = 2.0 * r_required / np.sqrt(3)
    return (c_candidate, side)

def _check_validity(centers, angles, outer_center, outer_side, outer_angle):
    """Check if arrangement is valid: no overlaps, all inside outer."""
    # Check pairwise overlaps using SAT
    for i in range(N):
        for j in range(i + 1, N):
            if _hex_overlap(centers[i], angles[i], centers[j], angles[j]) > 1e-7:
                return False
    # Check all inner vertices inside outer
    all_verts = _all_inner_vertices(centers, angles)
    for vx, vy in all_verts:
        if _point_in_hexagon(vx, vy, outer_center, outer_side, outer_angle) < -1e-7:
            return False
    return True

def _hex_overlap(c1, a1, c2, a2, side=1.0):
    """Return overlap depth if two hexagons overlap, 0 if separated."""
    v1 = _hex_vertices(c1[0], c1[1], side, a1)
    v2 = _hex_vertices(c2[0], c2[1], side, a2)
    
    def get_axes(verts):
        axes = []
        for i in range(6):
            x1, y1 = verts[i]
            x2, y2 = verts[(i + 1) % 6]
            ex, ey = x2 - x1, y2 - y1
            norm = np.sqrt(ex**2 + ey**2)
            if norm > 1e-12:
                axes.append((-ey / norm, ex / norm))
        return axes
    
    axes = get_axes(v1) + get_axes(v2)
    min_overlap = float('inf')
    
    for ax, ay in axes:
        proj1 = v1 @ np.array([ax, ay])
        proj2 = v2 @ np.array([ax, ay])
        if proj1.max() < proj2.min() - 1e-9 or proj2.max() < proj1.min() - 1e-9:
            return 0.0
        overlap = min(proj1.max(), proj2.max()) - max(proj1.min(), proj2.min())
        min_overlap = min(min_overlap, overlap)
    
    return max(0.0, -min_overlap)

def _point_in_hexagon(px, py, center, side, angle_deg):
    """Signed distance from point to hexagon boundary (positive inside)."""
    cx, cy = center
    angle = np.radians(angle_deg)
    cos_a, sin_a = np.cos(angle), np.sin(angle)
    lx = (px - cx) * cos_a + (py - cy) * sin_a
    ly = -(px - cx) * sin_a + (py - cy) * cos_a
    
    r = side * np.sqrt(3) / 2.0
    min_dist = float('inf')
    for i in range(6):
        theta = np.pi / 6 + np.pi * i / 3.0
        nx, ny = np.cos(theta), np.sin(theta)
        dist = lx * nx + ly * ny - r
        min_dist = min(min_dist, dist)
    return min_dist

def _hex_lattice_positions():
    """Generate all hex-lattice positions for 11 hexagons.
    Returns list of (centers, angles) tuples."""
    # Hex lattice: basis vectors
    # For flat-top hexagons, neighbors are at angles 0, 60, 120, 180, 240, 300
    # Distance between centers = sqrt(3)
    
    # Generate positions in axial coordinates (q, r)
    # Convert to pixel: x = sqrt(3) * (q + r/2), y = 3/2 * r  (for pointy-top)
    # Or for flat-top: x = 3/2 * q, y = sqrt(3) * (r + q/2)
    
    # Let's use a simple approach: generate all positions within radius 2
    # in the hex lattice
    
    positions = []
    # Axial coordinate to cartesian (flat-top hexagons)
    def axial_to_cart(q, r):
        x = 1.5 * q
        y = np.sqrt(3) * (r + q / 2.0)
        return (x, y)
    
    # Generate all axial coords with hex distance <= 2
    for q in range(-2, 3):
        for r in range(-2, 3):
            # Hex distance in axial coords
            dist = (abs(q) + abs(r) + abs(q + r)) / 2
            if dist <= 2:
                positions.append((q, r))
    
    # We need to choose 11 positions from these
    # The natural choice is the center + first ring (6) + part of second ring (4)
    # But we can also try other combinations
    
    return positions

def _try_arrangement(centers, angles, start_time):
    """Try to find optimal outer hexagon for given inner arrangement."""
    all_verts = _all_inner_vertices(centers, angles)
    
    best_side = float('inf')
    best_outer = None
    
    # Try different outer hexagon angles
    for angle_deg in np.arange(0, 60, 1.0):
        if time.time() > DEADLINE:
            break
        
        result = _min_enclosing_hexagon(all_verts, angle_deg)
        if result is not None:
            center, side = result
            if side < best_side:
                best_side = side
                best_outer = (center, side, angle_deg)
    
    # Refine with finer angle search around best
    if best_outer is not None:
        _, _, best_angle = best_outer
        for angle_deg in np.arange(best_angle - 5, best_angle + 5, 0.1):
            if time.time() > DEADLINE:
                break
            result = _min_enclosing_hexagon(all_verts, angle_deg)
            if result is not None:
                center, side = result
                if side < best_side:
                    best_side = side
                    best_outer = (center, side, angle_deg)
    
    return best_side, best_outer

def construct():
    start_time = time.time()
    
    best_score = float('inf')
    best_result = None
    
    def make_base_config():
        """Create hexagonal close-packed arrangement."""
        centers = np.zeros((N, 2))
        angles = np.zeros(N)
        
        # Center
        centers[0] = [0.0, 0.0]
        
        # First ring: 6 hexagons at distance sqrt(3)
        r1 = np.sqrt(3)
        for i in range(6):
            angle = np.pi / 3 * i
            centers[1 + i] = [r1 * np.cos(angle), r1 * np.sin(angle)]
        
        # Second ring: 12 possible positions, we need 4 more
        # Positions at distance 2*sqrt(3) from center
        # Or at hex lattice positions with distance 2
        
        # Hex lattice positions at distance 2 from center (axial coords)
        # In flat-top: (q,r) with hex_dist=2
        # (2,0), (1,1), (0,2), (-1,1), (-2,0), (-1,-1), (0,-2), (1,-1) -> 8 at dist 2? No
        # Hex dist = (|q| + |r| + |q+r|)/2
        # (2,0): (2+0+2)/2 = 2
        # (1,1): (1+1+2)/2 = 2
        # (0,2): (0+2+2)/2 = 2
        # (-1,1): (1+1+0)/2 = 1 -> no, that's dist 1
        # Let me recalculate:
        # (q,r) hex dist: max(|q|, |r|, |q+r|) ... no
        # Standard: (|q| + |r| + |q+r|) / 2
        # (2,0): (2+0+2)/2 = 2
        # (1,1): (1+1+2)/2 = 2
        # (0,2): (0+2+2)/2 = 2
        # (-1,1): (1+1+0)/2 = 1
        # (-2,0): (2+0+2)/2 = 2
        # (-1,-1): (1+1+2)/2 = 2
        # (0,-2): (0+2+2)/2 = 2
        # (1,-1): (1+1+0)/2 = 1
        
        # So distance 2 positions: (2,0), (1,1), (0,2), (-2,0), (-1,-1), (0,-2) -> 6 positions
        # Wait, that's only 6. Let me list all with dist=2:
        # (2,0), (1,1), (0,2), (-1,3)? No...
        
        # Actually for hex lattice, the second ring has 12 positions.
        # Let me just generate them geometrically.
        
        # Second ring positions: at angles 0, 30, 60, ..., 330 degrees
        # But not all at the same distance. In hex lattice:
        # From center, the 12 neighbors at distance 2 are:
        # 6 at distance 2*sqrt(3) in directions 0, 60, 120, 180, 240, 300
        # 6 at distance sqrt(3)*sqrt(7) ... no
        
        # Simpler: just place 4 hexagons in the second ring at specific positions
        # The standard 11-hexagon packing uses center + 6 + 4
        # The 4 in the second ring are at alternating positions
        
        # Let's use the hex lattice properly
        # Flat-top hexagon lattice, neighbor vectors:
        # (sqrt(3), 0), (sqrt(3)/2, 3/2), (-sqrt(3)/2, 3/2), (-sqrt(3), 0), (-sqrt(3)/2, -3/2), (sqrt(3)/2, -3/2)
        
        n1 = np.array([np.sqrt(3), 0.0])
        n2 = np.array([np.sqrt(3)/2, 1.5])
        
        # Generate all positions as a*n1 + b*n2 for a,b in -2..2
        lattice_points = []
        for a in range(-2, 3):
            for b in range(-2, 3):
                pt = a * n1 + b * n2
                dist_from_center = np.linalg.norm(pt)
                if dist_from_center < 1e-9:
                    lattice_points.append((pt, 0, 0))  # center
                else:
                    # Determine ring
                    ring = int(round(dist_from_center / np.sqrt(3)))
                    lattice_points.append((pt, a, b))
        
        # Get unique positions
        pos_dict = {}
        for pt, a, b in lattice_points:
            key = (round(pt[0], 6), round(pt[1], 6))
            if key not in pos_dict:
                pos_dict[key] = pt
        
        all_positions = list(pos_dict.values())
        
        # Sort by distance from center
        all_positions.sort(key=lambda p: np.linalg.norm(p))
        
        # Take center + first 6 + next 4
        # Center should be first
        centers = np.array(all_positions[:N])
        
        return centers, angles
    
    # Generate base configuration
    centers_base, angles_base = make_base_config()
    
    # Try the base configuration
    side, outer = _try_arrangement(centers_base, angles_base, start_time)
    if outer is not None and side < best_score:
        best_score = side
        best_result = (centers_base, angles_base, outer)
    
    # Try different second-ring selections
    # The 12 positions in the second ring: choose 4 out of 12
    # But we need to be careful about which positions are "second ring"
    
    # Let's enumerate different arrangements by permuting which 4 of the 12 second-ring positions we use
    # First, identify all lattice positions
    
    n1 = np.array([np.sqrt(3), 0.0])
    n2 = np.array([np.sqrt(3)/2, 1.5])
    
    # Generate all lattice positions
    lattice_positions = {}
    for a in range(-2, 3):
        for b in range(-2, 3):
            pt = a * n1 + b * n2
            key = (round(pt[0], 6), round(pt[1], 6))
            if key not in lattice_positions:
                lattice_positions[key] = (pt, a, b)
    
    # Classify by ring
    ring0 = []
    ring1 = []
    ring2 = []
    for key, (pt, a, b) in lattice_positions.items():
        dist = np.linalg.norm(pt)
        if dist < 1e-9:
            ring0.append(pt)
        elif dist < 1.5 * np.sqrt(3):
            ring1.append(pt)
        elif dist < 2.5 * np.sqrt(3):
            ring2.append(pt)
    
    # ring0 should have 1, ring1 should have 6, ring2 should have 12
    # Try different selections of 4 from ring2
    
    # To limit computation, try a subset of combinations
    # There are C(12,4) = 495 combinations
    
    tried = 0
    for combo in itertools.combinations(range(len(ring2)), 4):
        if time.time() > DEADLINE - 10:
            break
        
        # Build centers: ring0 + ring1 + selected ring2
        centers = np.array(ring0 + ring1 + [ring2[i] for i in combo])
        angles = np.zeros(N)
        
        side, outer = _try_arrangement(centers, angles, start_time)
        if outer is not None and side < best_score:
            best_score = side
            best_result = (centers, angles, outer)
        
        tried += 1
        if tried >= 200 and time.time() > start_time + 100:
            break
    
    if best_result is None:
        # Fallback
        inner = [[4.0 * k, 0.0, 0.0] for k in range(-(N // 2), N - N // 2)]
        return inner, (0.0, 0.0), 40.0, 0.0
    
    centers, angles, (outer_center, outer_side, outer_angle) = best_result
    inner = [[float(centers[i][0]), float(centers[i][1]), float(angles[i])] for i in range(N)]
    return inner, (float(outer_center[0]), float(outer_center[1])), float(outer_side), float(outer_angle)
