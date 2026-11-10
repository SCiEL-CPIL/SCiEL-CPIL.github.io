# EVOLVE-BLOCK-START
import numpy as np
import time


def _hex_vertices(cx, cy, s, a_deg):
    """Return 6 vertices of a regular hexagon."""
    a = np.radians(a_deg)
    angles = a + np.arange(6) * np.pi / 3
    return np.column_stack([cx + s * np.cos(angles), cy + s * np.sin(angles)])


def _edge_normals(verts):
    """Return unit edge normals (6) for a hexagon's vertices in CCW order."""
    edges = np.roll(verts, -1, axis=0) - verts
    # For CCW polygon, outward normal is perpendicular to edge direction
    # If edge is (dx, dy), outward normal for CCW is (dy, -dx) normalized
    nx = edges[:, 1]
    ny = -edges[:, 0]
    norms = np.sqrt(nx ** 2 + ny ** 2)
    norms[norms < 1e-12] = 1.0
    return np.column_stack([nx / norms, ny / norms])


def _project_interval(verts, normal):
    """Project hexagon vertices onto a normal direction, return (min, max)."""
    dots = verts @ normal
    return dots.min(), dots.max()


def _hexagons_separated(v1, v2):
    """Check if two hexagons (each with 6 vertices) are separated by > 1e-6 gap."""
    n1 = _edge_normals(v1)
    n2 = _edge_normals(v2)
    all_normals = np.vstack([n1, n2])
    for normal in all_normals:
        min1, max1 = _project_interval(v1, normal)
        min2, max2 = _project_interval(v2, normal)
        # Check if intervals are apart (with gap > 1e-6)
        if max1 < min2 - 1e-6 or max2 < min1 - 1e-6:
            return True
    return False


def _all_inner_separated(inner_hexes):
    """Check all pairs of inner hexagons are separated."""
    n = len(inner_hexes)
    for i in range(n):
        for j in range(i + 1, n):
            if not _hexagons_separated(inner_hexes[i], inner_hexes[j]):
                return False
    return True


def _inner_in_outer(inner_hexes, outer_verts):
    """Check all inner hex vertices are inside or on the outer hexagon."""
    n_outer = len(outer_verts)
    for hex_verts in inner_hexes:
        for p in hex_verts:
            for i in range(n_outer):
                p1 = outer_verts[i]
                p2 = outer_verts[(i + 1) % n_outer]
                cross = (p2[0] - p1[0]) * (p[1] - p1[1]) - (p2[1] - p1[1]) * (p[0] - p1[0])
                if cross < -1e-6:
                    return False
    return True


def _valid_configuration(inner_data, outer_data, side):
    """Full validity check."""
    # Check finiteness
    if not np.all(np.isfinite(inner_data)) or not np.all(np.isfinite(outer_data)):
        return False
    if side < np.sqrt(11) - 1e-9:
        return False
    
    ox, oy, oa = outer_data
    outer_verts = _hex_vertices(ox, oy, side, oa)
    
    inner_hexes = []
    for row in inner_data:
        cx, cy, ca = row
        verts = _hex_vertices(cx, cy, 1.0, ca)
        inner_hexes.append(verts)
    
    # Check all inner pairs separated
    if not _all_inner_separated(inner_hexes):
        return False
    
    # Check all inner vertices inside outer
    if not _inner_in_outer(inner_hexes, outer_verts):
        return False
    
    # Check edge lengths within 1e-8 of 1
    for hex_verts in inner_hexes:
        for i in range(6):
            d = np.linalg.norm(hex_verts[(i + 1) % 6] - hex_verts[i])
            if abs(d - 1.0) > 1e-8:
                return False
    
    return True


def _separation_margin(inner_hexes):
    """Return minimum gap over all pairs and all normals (larger = more separated)."""
    min_gap = np.inf
    n = len(inner_hexes)
    for i in range(n):
        for j in range(i + 1, n):
            n1 = _edge_normals(inner_hexes[i])
            n2 = _edge_normals(inner_hexes[j])
            all_normals = np.vstack([n1, n2])
            for normal in all_normals:
                min1, max1 = _project_interval(inner_hexes[i], normal)
                min2, max2 = _project_interval(inner_hexes[j], normal)
                gap = max(min2 - max1, min1 - max2)
                if gap < min_gap:
                    min_gap = gap
    return min_gap


def _outer_constraint_margin(inner_hexes, outer_verts):
    """Return minimum cross product margin (larger = more inside)."""
    min_margin = np.inf
    n_outer = len(outer_verts)
    for hex_verts in inner_hexes:
        for p in hex_verts:
            for i in range(n_outer):
                p1 = outer_verts[i]
                p2 = outer_verts[(i + 1) % n_outer]
                cross = (p2[0] - p1[0]) * (p[1] - p1[1]) - (p2[1] - p1[1]) * (p[0] - p1[0])
                if cross < min_margin:
                    min_margin = cross
    return min_margin


def hexagon_packing_11():
    """
    Constructs a packing of 11 disjoint unit regular hexagons inside a larger regular hexagon, maximizing 1/outer_hex_side_length.
    """
    start_time = time.time()
    deadline = start_time + 580  # Safety margin before 630s limit
    
    # Start with a dense hexagonal arrangement of 11 unit hexagons
    # Center + 6 in first ring + 4 more in second ring (partial)
    # Using hexagonal lattice spacing: adjacent centers are sqrt(3) apart for touching hexagons
    # Actually for unit hexagons, the distance between centers of touching hexagons is sqrt(3) * 1 = sqrt(3)
    # Wait - for regular hexagons with side 1, the distance from center to vertex is 1, and the
    # distance between parallel sides is sqrt(3). Two hexagons touch when their centers are sqrt(3) apart.
    
    s3 = np.sqrt(3)
    
    # Hexagonal lattice positions for 11 hexagons
    # Ring 0: center (0,0)
    # Ring 1: 6 hexagons at distance s3 in 6 directions
    # Ring 2: 12 hexagons at distance 2*s3, but we only need 4 more
    
    positions = np.array([[0.0, 0.0]])
    
    # Ring 1: 6 positions
    for i in range(6):
        angle = i * np.pi / 3
        positions = np.vstack([positions, [s3 * np.cos(angle), s3 * np.sin(angle)]])
    
    # Ring 2: 12 positions, take 4
    for i in range(12):
        angle = i * np.pi / 6 + np.pi / 6
        positions = np.vstack([positions, [2 * s3 * np.cos(angle), 2 * s3 * np.sin(angle)]])
    
    # We have 1 + 6 + 12 = 19 positions, take first 11
    positions = positions[:11]
    
    # Initial outer hexagon: large enough
    outer_data = np.array([0.0, 0.0, 0.0])
    side = 10.0
    
    best_side = side
    best_inner = positions.copy()
    
    # Verify initial configuration
    inner_hexes = [_hex_vertices(p[0], p[1], 1.0, 0.0) for p in positions]
    outer_verts = _hex_vertices(outer_data[0], outer_data[1], side, outer_data[2])
    
    if _all_inner_separated(inner_hexes) and _inner_in_outer(inner_hexes, outer_verts):
        best_side = side
        best_inner = positions.copy()
    
    # Local optimization: try to shrink outer hexagon by adjusting positions
    # Use a simple gradient-free approach: try small perturbations and shrink
    
    # First, compute the minimum required side for current positions
    def compute_min_side(inner_data, outer_angle=0.0):
        """Find minimum side length for given inner positions."""
        # The outer hexagon must contain all inner hex vertices
        # For a regular hexagon centered at origin with side s and angle a,
        # the constraint is that all vertices must satisfy the half-plane inequalities
        
        min_s = np.sqrt(11)
        
        # For each inner hex, compute the maximum extent in each of the 6 normal directions
        # of the outer hexagon
        
        for trial_angle in [0.0, np.pi/6, np.pi/3, np.pi/2, 2*np.pi/3, 5*np.pi/6]:
            a = np.radians(trial_angle)
            # Outer hexagon edge normals (outward)
            # Vertices at angles a, a+60, ..., a+300
            # Edge from vertex i to i+1 has outward normal at angle a + 60*(i+1) - 30
            normals = []
            for i in range(6):
                norm_angle = a + np.pi/3 * (i + 1) - np.pi/6
                normals.append(np.array([np.cos(norm_angle), np.sin(norm_angle)]))
            normals = np.array(normals)
            
            # For each normal, the max projection of any inner vertex gives required extent
            max_proj = 0
            for row in inner_data:
                cx, cy, ca = row
                verts = _hex_vertices(cx, cy, 1.0, ca)
                for normal in normals:
                    proj = np.max(verts @ normal)
                    if proj > max_proj:
                        max_proj = proj
            
            # For a regular hexagon with side s, the distance from center to each edge is s * sqrt(3)/2
            # So we need s * sqrt(3)/2 >= max_proj
            required_s = max_proj / (np.sqrt(3) / 2)
            if required_s > min_s:
                min_s = required_s
        
        return min_s
    
    # Optimize positions to minimize the required side
    current_positions = positions.copy()
    current_side = compute_min_side(current_positions)
    
    if current_side < best_side:
        best_side = current_side
        best_inner = current_positions.copy()
    
    # Simple local search: random perturbations with acceptance if side decreases
    rng = np.random.RandomState(42)
    step_size = 0.1
    n_iterations = 5000
    
    for it in range(n_iterations):
        if time.time() > deadline - 30:
            break
        
        # Try random perturbation
        idx = rng.randint(0, 11)
        delta = rng.normal(0, step_size, size=2)
        
        new_positions = current_positions.copy()
        new_positions[idx] += delta
        
        # Check if inner hexagons are still separated
        new_inner_hexes = [_hex_vertices(p[0], p[1], 1.0, 0.0) for p in new_positions]
        
        if not _all_inner_separated(new_inner_hexes):
            step_size *= 0.95
            continue
        
        new_side = compute_min_side(new_positions)
        
        if new_side < current_side:
            current_positions = new_positions
            current_side = new_side
            step_size = min(step_size * 1.05, 0.3)
            
            if current_side < best_side:
                best_side = current_side
                best_inner = current_positions.copy()
        else:
            step_size *= 0.98
    
    # Final: use the best configuration found
    final_inner = best_inner.copy()
    final_side = best_side
    
    # Center the configuration: shift so that the bounding is symmetric
    # Compute centroid of all inner hex centers
    centroid = np.mean(final_inner[:, :2], axis=0)
    final_inner[:, 0] -= centroid[0]
    final_inner[:, 1] -= centroid[1]
    
    # Recompute side after centering
    final_side = compute_min_side(final_inner)
    
    # Ensure side is at least sqrt(11)
    final_side = max(final_side, np.sqrt(11))
    
    outer_data = np.array([0.0, 0.0, 0.0])
    
    # Final validity check
    if not _valid_configuration(final_inner, outer_data, final_side):
        # Fallback to known valid configuration
        final_inner = np.array([
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
        ])
        outer_data = np.array([0, 0, 0])
        final_side = 8.0
    
    return final_inner, outer_data, final_side


# EVOLVE-BLOCK-END
