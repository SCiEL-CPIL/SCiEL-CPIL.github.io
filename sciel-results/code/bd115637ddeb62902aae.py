# EVOLVE-BLOCK-START
import numpy as np
import time


def _hex_vertices(cx, cy, s, a_deg):
    """Return 6 vertices of a regular hexagon with center (cx,cy), side s, angle a_deg."""
    a = np.radians(a_deg)
    angles = a + np.deg2rad(np.arange(6) * 60)
    return np.column_stack([cx + s * np.cos(angles), cy + s * np.sin(angles)])


def _pair_separated(v1, v2, eps=1e-6):
    """Check if two convex hexagons (arrays of 6 vertices) are separated by > eps on at least one edge normal."""
    # Edge normals for hexagon 1
    for i in range(6):
        p1 = v1[i]
        p2 = v1[(i + 1) % 6]
        edge = p2 - p1
        normal = np.array([-edge[1], edge[0]])
        norm = np.linalg.norm(normal)
        if norm < 1e-12:
            continue
        normal /= norm
        # Project both hexagons onto this normal
        proj1 = v1 @ normal
        proj2 = v2 @ normal
        min1, max1 = proj1.min(), proj1.max()
        min2, max2 = proj2.min(), proj2.max()
        # Check if intervals are apart by more than eps
        if max1 < min2 - eps or max2 < min1 - eps:
            return True
    # Edge normals for hexagon 2
    for i in range(6):
        p1 = v2[i]
        p2 = v2[(i + 1) % 6]
        edge = p2 - p1
        normal = np.array([-edge[1], edge[0]])
        norm = np.linalg.norm(normal)
        if norm < 1e-12:
            continue
        normal /= norm
        proj1 = v1 @ normal
        proj2 = v2 @ normal
        min1, max1 = proj1.min(), proj1.max()
        min2, max2 = proj2.min(), proj2.max()
        if max1 < min2 - eps or max2 < min1 - eps:
            return True
    return False


def _all_separated(inner_hexes, eps=1e-6):
    """Check all pairs of inner hexagons are separated."""
    n = len(inner_hexes)
    for i in range(n):
        for j in range(i + 1, n):
            if not _pair_separated(inner_hexes[i], inner_hexes[j], eps):
                return False
    return True


def _inside_outer(inner_hexes, outer_verts, tol=-1e-6):
    """Check all inner vertices are inside or on the outer hexagon."""
    n_outer = len(outer_verts)
    for hex_v in inner_hexes:
        for p in hex_v:
            for i in range(n_outer):
                p1 = outer_verts[i]
                p2 = outer_verts[(i + 1) % n_outer]
                cross = (p2[0] - p1[0]) * (p[1] - p1[1]) - (p2[1] - p1[1]) * (p[0] - p1[0])
                if cross < tol:
                    return False
    return True


def _compute_min_side(inner_centers, inner_angles, max_iter=1000):
    """Find minimum side length of outer hexagon (centered at origin, angle 0) containing all inner hexagons."""
    # For a regular hexagon centered at origin with angle 0, the constraint is:
    # For each vertex of each inner hex, it must satisfy the 6 half-plane constraints.
    # The 6 edge normals for outer hex with side s, center (0,0), angle 0:
    # Vertices at angles 0, 60, 120, 180, 240, 300 degrees from center at distance s
    # Edge i goes from vertex i to vertex i+1
    # The inward normal for edge i points toward center
    
    # For a regular hexagon with circumradius s (side length = circumradius for regular hex),
    # vertices are at distance s from center.
    # Actually for a regular hexagon, side length = circumradius.
    
    # The 6 constraints: for each edge, the dot product of (p - edge_midpoint) with outward normal <= 0
    # Or equivalently, for each of the 6 directions at angles 30, 90, 150, 210, 270, 330 degrees:
    # |p · u_k| <= s * cos(30°) = s * sqrt(3)/2
    # Wait, let me think more carefully.
    
    # For a regular hexagon with side s centered at origin, angle 0:
    # Vertices at (s*cos(60k), s*sin(60k)) for k=0..5
    # Edge 0: from vertex 0 to vertex 1
    # The support function in direction u is s * cos(angle between u and nearest vertex direction)
    # Actually, the width in direction u for a regular hexagon with circumradius s is:
    # h(u) = s * max_k cos(angle(u, 60k°))
    # But we need the constraint that all points satisfy p·u <= h(u) for all u, and p·(-u) <= h(-u)
    
    # Simpler: the 6 half-plane constraints.
    # Edge k connects vertex k and vertex (k+1)%6.
    # Vertex k is at angle 60k degrees, distance s.
    # Edge k midpoint is at angle 30k+30 degrees... no.
    # Vertex 0: (s, 0), Vertex 1: (s/2, s*sqrt(3)/2)
    # Edge 0 midpoint: (3s/4, s*sqrt(3)/4), direction of edge: (-s/2, s*sqrt(3)/2)
    # Outward normal: perpendicular to edge, pointing away from center.
    # Edge direction: (-1/2, sqrt(3)/2), normal: (sqrt(3)/2, 1/2) normalized = (sqrt(3)/2, 1/2)
    # The constraint: p · (sqrt(3)/2, 1/2) <= s * sqrt(3)/2
    
    # In general, for edge k, the outward normal direction is at angle (30 + 60k) degrees.
    # The constraint is: p · n_k <= s * cos(30°) = s * sqrt(3)/2
    # where n_k is the unit normal at angle (30 + 60k) degrees.
    
    n_angles = [30 + 60 * k for k in range(6)]
    normals = np.array([[np.cos(np.radians(a)), np.sin(np.radians(a))] for a in n_angles])
    # Constraint: p · n_k <= s * sqrt(3)/2 for all k
    
    # For each inner hex vertex, compute the max over all k of (p · n_k) / (sqrt(3)/2)
    # The required s is the max over all vertices and all k of (p · n_k) / (sqrt(3)/2)
    
    all_vertices = []
    for center, angle in zip(inner_centers, inner_angles):
        verts = _hex_vertices(center[0], center[1], 1.0, angle)
        all_vertices.append(verts)
    all_vertices = np.vstack(all_vertices)
    
    # For each vertex p and each normal n_k, compute p · n_k
    # Required s = max(p · n_k) / (sqrt(3)/2)
    # But also need p · (-n_k) <= s * sqrt(3)/2, i.e., -p · n_k <= s * sqrt(3)/2
    # So |p · n_k| <= s * sqrt(3)/2
    # Required s = max(|p · n_k|) / (sqrt(3)/2)
    
    # Compute all dot products
    # all_vertices: (N, 2), normals: (6, 2)
    dots = all_vertices @ normals.T  # (N, 6)
    abs_dots = np.abs(dots)
    max_dot = abs_dots.max()
    s = max_dot / (np.sqrt(3) / 2)
    
    return s


def hexagon_packing_11():
    """
    Constructs a packing of 11 disjoint unit regular hexagons inside a larger regular hexagon, maximizing 1/outer_hex_side_length.
    Returns
        inner_hex_data: np.ndarray of shape (11,3), where each row is of the form (x, y, angle_degrees) containing the (x,y) coordinates and angle_degree of the respective inner hexagon.
        outer_hex_data: np.ndarray of shape (3,) of form (x,y,angle_degree) containing the (x,y) coordinates and angle_degree of the outer hexagon.
        outer_hex_side_length: float representing the side length of the outer hexagon.
    """
    start_time = time.time()
    deadline = start_time + 600
    
    # Start with a good initial packing: hexagonal arrangement
    # Unit hexagons in a honeycomb pattern
    # Spacing between adjacent hex centers: 2 * (sqrt(3)/2) = sqrt(3) for horizontal neighbors
    # Actually, for unit hexagons (side 1), the distance between centers of adjacent hexagons in a honeycomb is 2 * (sqrt(3)/2) = sqrt(3) ≈ 1.732
    # Wait, let me reconsider. A regular hexagon with side 1 has inradius sqrt(3)/2 ≈ 0.866.
    # Two adjacent hexagons sharing an edge have centers separated by 2 * inradius = sqrt(3).
    
    # Let's use a compact arrangement. The optimal packing of 11 hexagons in a hexagon...
    # Try a 3-4-3 or similar arrangement.
    
    # Hexagonal lattice positions for 11 hexagons:
    # Center at (0,0)
    # 6 neighbors at distance sqrt(3) in directions 0, 60, 120, 180, 240, 300
    # Plus 4 more to make 11
    
    # Let me try: a hexagon of hexagons with 3 on each side would be 19 total (1+6+12).
    # For 11, maybe: center + 6 neighbors + 4 more.
    
    # Actually, let's just use a simple grid and optimize.
    
    # Initial guess: hexagonal lattice
    d = np.sqrt(3)  # distance between adjacent hex centers
    positions = [
        (0, 0),           # center
        (d, 0),           # right
        (-d, 0),          # left
        (d/2, d*3/2),     # top-right
        (-d/2, d*3/2),    # top-left
        (d/2, -d*3/2),    # bottom-right
        (-d/2, -d*3/2),   # bottom-left
        (2*d, 0),         # far right
        (-2*d, 0),        # far left
        (0, d*3),         # top
        (0, -d*3),        # bottom
    ]
    
    inner_hex_data = np.array([(x, y, 0.0) for x, y in positions])
    
    # Compute initial side length
    centers = inner_hex_data[:, :2]
    angles = inner_hex_data[:, 2]
    side = _compute_min_side(centers, angles)
    
    best_side = side
    best_inner = inner_hex_data.copy()
    
    # Try to optimize by adjusting positions
    # Use a simple gradient-free optimization: random perturbations
    
    np.random.seed(42)
    
    max_iterations = 50000
    for it in range(max_iterations):
        if time.time() > deadline - 30:
            break
        
        # Random perturbation
        perturbation = np.random.randn(11, 2) * 0.01
        new_centers = centers + perturbation
        new_inner = np.column_stack([new_centers, angles])
        
        # Check separation
        hex_verts = [_hex_vertices(cx, cy, 1.0, a) for cx, cy, a in new_inner]
        if _all_separated(hex_verts):
            new_side = _compute_min_side(new_centers, angles)
            if new_side < best_side:
                best_side = new_side
                best_inner = new_inner.copy()
                centers = new_centers.copy()
    
    # Final result
    outer_hex_data = np.array([0.0, 0.0, 0.0])
    
    return best_inner, outer_hex_data, best_side


# EVOLVE-BLOCK-END
