# EVOLVE-BLOCK-START
import numpy as np


def _hex_vertices(cx, cy, s, a_deg):
    """Return 6 vertices of a regular hexagon centered at (cx,cy), side s, angle a_deg."""
    a = np.deg2rad(a_deg)
    angles = a + np.deg2rad(60.0 * np.arange(6))
    return np.column_stack([cx + s * np.cos(angles), cy + s * np.sin(angles)])


def _min_outer_side_for_inner(inner_data, outer_center, outer_angle_deg):
    """
    Compute the minimum outer hex side length such that all inner hex vertices
    are inside the outer hexagon centered at outer_center with given angle.
    Uses the half-plane (cross product) containment test.
    """
    ox, oy, oa = outer_center
    oa_rad = np.deg2rad(oa)
    
    # Outer hex vertices
    outer_verts = _hex_vertices(ox, oy, 1.0, oa_deg=oa)  # unit side for direction
    
    # Edge normals (inward-pointing)
    # For edge from v_i to v_{i+1}, the inward normal direction
    # cross(p2-p1, p-p1) >= 0 means p is on the left of directed edge v1->v2
    # For a CCW polygon, interior is to the left.
    # We need all inner vertices to satisfy: for each outer edge (v_i, v_{i+1}),
    # cross(v_{i+1}-v_i, p - v_i) >= 0 (with tolerance)
    
    # The constraint: for outer edge from v1 to v2 (CCW), 
    # cross(v2-v1, p-v1) >= 0
    # This is equivalent to: dot(normal, p - v1) >= 0 where normal is the inward normal
    
    # Let's compute the required side length.
    # For a regular hexagon with center (ox,oy), side s, angle a:
    # Vertices at (ox + s*cos(a+60i), oy + s*sin(a+60i))
    # The distance from center to each edge (apothem) = s * sqrt(3)/2
    # The direction of each edge normal (inward) is at angle a + 60i + 30 (perpendicular to edge)
    
    # Actually, for a regular hexagon, the 6 inward edge normals point in directions:
    # a + 30 + 60*i for i=0..5 (in degrees), scaled by apothem = s*sqrt(3)/2
    
    # For each inner vertex p, the constraint for edge i is:
    # dot(n_i, p - center) <= apothem
    # where n_i is the unit inward normal for edge i
    
    # So: s * sqrt(3)/2 >= max over all inner vertices p and all edges i of dot(n_i, p - center)
    # Therefore: s >= (2/sqrt(3)) * max_{p, i} dot(n_i, p - center)
    
    # Compute all inner vertices
    all_inner_verts = []
    for row in inner_data:
        x, y, ang = row
        verts = _hex_vertices(x, y, 1.0, ang)
        all_inner_verts.append(verts)
    all_inner_verts = np.vstack(all_inner_verts)
    
    # Shift by outer center
    shifted = all_inner_verts - np.array([ox, oy])
    
    # Inward normal directions for outer hex with angle a
    # Edge i goes from vertex i to vertex i+1
    # Vertex i is at angle a + 60i
    # Edge direction: v_{i+1} - v_i
    # Inward normal is perpendicular to edge, pointing toward center
    
    # For a regular hexagon, the inward normal for edge i points in direction:
    # a + 60i + 30 degrees? Let me think...
    # Vertex i at angle theta_i = a + 60i
    # Vertex i+1 at angle theta_{i+1} = a + 60(i+1)
    # Edge vector: (cos(theta_{i+1}) - cos(theta_i), sin(theta_{i+1}) - sin(theta_i))
    # Inward normal: rotate edge vector by -90 degrees (for CCW polygon, interior is left)
    # Actually for CCW polygon, inward normal is to the right of edge direction
    # Right of (dx, dy) is (dy, -dx)
    
    # Let me just compute directly:
    # For unit hex, vertices at angles a + 60i
    # Edge from v_i to v_{i+1}
    # The inward normal direction (unit) can be computed
    
    normal_dirs = []
    for i in range(6):
        theta_i = oa_rad + np.deg2rad(60.0 * i)
        theta_next = oa_rad + np.deg2rad(60.0 * (i + 1))
        v_i = np.array([np.cos(theta_i), np.sin(theta_i)])
        v_next = np.array([np.cos(theta_next), np.sin(theta_next)])
        edge = v_next - v_i
        # Inward normal: for CCW polygon, interior is to the left of edge direction
        # Left of (dx,dy) is (-dy, dx)
        n = np.array([-edge[1], edge[0]])
        n = n / np.linalg.norm(n)
        normal_dirs.append(n)
    normal_dirs = np.array(normal_dirs)  # shape (6, 2)
    
    # For each shifted inner vertex, compute max dot product with each normal
    # dot(n_i, p) for all p, all i
    # We need: for each edge i, max over p of dot(n_i, p) <= apothem = s * sqrt(3)/2
    # So s >= (2/sqrt(3)) * max_i (max_p dot(n_i, p))
    
    # Compute max dot for each normal direction
    # dot(n_i, p) for all points p
    # = p @ n_i for each i
    max_dots = np.max(shifted @ normal_dirs.T, axis=0)  # shape (6,)
    max_dot = np.max(max_dots)
    
    if max_dot <= 0:
        return 0.0
    
    s_min = (2.0 / np.sqrt(3.0)) * max_dot
    return s_min


def _check_validity(inner_data, outer_data, side):
    """Check if the packing is valid."""
    if not np.all(np.isfinite(inner_data)) or not np.all(np.isfinite(outer_data)):
        return False
    if side < np.sqrt(11.0) - 1e-9:
        return False
    
    ox, oy, oa = outer_data
    
    # Check all inner vertices are inside outer hex
    all_inner_verts = []
    for row in inner_data:
        x, y, ang = row
        verts = _hex_vertices(x, y, 1.0, ang)
        all_inner_verts.append(verts)
    all_inner_verts = np.vstack(all_inner_verts)
    
    # Outer hex vertices
    outer_verts = _hex_vertices(ox, oy, side, oa)
    
    # For each outer edge, check cross product constraint
    for i in range(6):
        v1 = outer_verts[i]
        v2 = outer_verts[(i + 1) % 6]
        edge = v2 - v1
        # cross(edge, p - v1) >= -1e-6 for all p
        diffs = all_inner_verts - v1
        crosses = diffs[:, 0] * edge[1] - diffs[:, 1] * edge[0]
        if np.min(crosses) < -1e-6:
            return False
    
    # Check pairwise separation of inner hexagons
    # Two hexagons are separated if on at least one of their 12 edge normals,
    # the projected intervals are apart by more than 1e-6
    # This is complex; for axis-aligned hexagons with same orientation,
    # we can check if any pair overlaps using SAT-like approach
    
    # For simplicity, check if centers are far enough apart
    # Two unit hexagons with same orientation: minimum distance to not overlap
    # is related to the hexagon geometry
    
    # Actually, let's do a proper check:
    # For two convex polygons, they are separated if there exists a separating axis
    # The 12 edge normals (6 from each hex) are the candidate axes
    
    for i in range(11):
        for j in range(i + 1, 11):
            vi = _hex_vertices(*inner_data[i])
            vj = _hex_vertices(*inner_data[j])
            
            # Get all 12 normal directions
            normals = []
            for v in [vi, vj]:
                for k in range(6):
                    e = v[(k + 1) % 6] - v[k]
                    n = np.array([-e[1], e[0]])
                    norm = np.linalg.norm(n)
                    if norm > 1e-12:
                        normals.append(n / norm)
            
            separated = False
            for n in normals:
                proj_i = vi @ n
                proj_j = vj @ n
                min_i, max_i = proj_i.min(), proj_i.max()
                min_j, max_j = proj_j.min(), proj_j.max()
                # Intervals are apart if max_i < min_j or max_j < min_i
                if max_i < min_j - 1e-6 or max_j < min_i - 1e-6:
                    separated = True
                    break
            
            if not separated:
                return False
    
    # Check edge lengths of shifted inner hexagons
    for row in inner_data:
        x, y, ang = row
        verts = _hex_vertices(x, y, 1.0, ang)
        shifted = verts - np.array([ox, oy])
        for k in range(6):
            e = shifted[(k + 1) % 6] - shifted[k]
            length = np.linalg.norm(e)
            if abs(length - 1.0) > 1e-8:
                return False
    
    return True


def hexagon_packing_11():
    """
    Constructs a packing of 11 disjoint unit regular hexagons inside a larger regular hexagon, maximizing 1/outer_hex_side_length.
    """
    # Hexagonal cluster: 1 center + 6 surrounding + 4 more
    # For unit hexagons with flat-top orientation (angle=0 means vertex at right):
    # Actually angle=0 means first vertex at angle 0 (pointing right)
    # 
    # For hexagons with "pointy-top" arrangement, adjacent hexagons in a honeycomb:
    # Horizontal spacing: 1.5 (for pointy-top) or sqrt(3) (for flat-top)
    # 
    # Let's use angle=0 (vertex pointing right, so flat-top is not quite right)
    # With angle=0, vertices are at 0, 60, 120, 180, 240, 300 degrees
    # The "width" (leftmost to rightmost) = 2 (from -1 to 1)
    # The "height" (top to bottom) = sqrt(3)
    
    # For two adjacent hexagons sharing an edge:
    # If hex A is at origin with angle 0, its right vertex is at (1,0)
    # Hex B sharing the right edge: the right edge of A goes from (1,0) to (0.5, sqrt(3)/2)
    # Wait, let me recompute. Vertices at angles 0,60,120,180,240,300:
    # v0=(1,0), v1=(0.5, sqrt(3)/2), v2=(-0.5, sqrt(3)/2), v3=(-1,0), v4=(-0.5,-sqrt(3)/2), v5=(0.5,-sqrt(3)/2)
    # 
    # Edge from v0 to v1: direction (−0.5, sqrt(3)/2), this is the upper-right edge
    # For hex B to share this edge, B's center would be at...
    # The midpoint of edge v0-v1 is (0.75, sqrt(3)/4)
    # The normal to this edge pointing outward from A is in direction (sqrt(3)/2, 0.5) (perpendicular, outward)
    # B's center = A's center + normal * distance
    # Distance between centers of adjacent hexagons = 2 * apothem = 2 * sqrt(3)/2 = sqrt(3)
    # So B's center = (0,0) + sqrt(3) * (sqrt(3)/2, 1/2) = (1.5, sqrt(3)/2)
    
    # So for a honeycomb with angle=0:
    # Horizontal neighbors: offset (1.5, ±sqrt(3)/2) and (-1.5, ±sqrt(3)/2)
    # Let's place:
    # Center at (0,0)
    # 6 surrounding at: (1.5, sqrt(3)/2), (-1.5, sqrt(3)/2), (0, sqrt(3)), (1.5, -sqrt(3)/2), (-1.5, -sqrt(3)/2), (0, -sqrt(3))
    
    sq3 = np.sqrt(3.0)
    
    inner_hex_data = np.array([
        [0, 0, 0],                          # center
        [1.5, sq3/2, 0],                     # right-upper
        [-1.5, sq3/2, 0],                    # left-upper
        [0, sq3, 0],                         # top
        [1.5, -sq3/2, 0],                    # right-lower
        [-1.5, -sq3/2, 0],                   # left-lower
        [0, -sq3, 0],                        # bottom
        # 4 more hexagons - extend the cluster
        [3.0, sq3/2, 0],                     # far right-upper
        [-3.0, sq3/2, 0],                    # far left-upper
        [3.0, -sq3/2, 0],                    # far right-lower
        [-3.0, -sq3/2, 0],                   # far left-lower
    ])
    
    # Outer hexagon centered at origin
    outer_center = np.array([0.0, 0.0, 0.0])
    
    # Compute minimum side length
    side = _min_outer_side_for_inner(inner_hex_data, outer_center, 0.0)
    # Add small margin for numerical safety
    side += 1e-7
    
    outer_hex_data = outer_center
    
    # Verify validity
    if not _check_validity(inner_hex_data, outer_hex_data, side):
        # Fallback to a larger side
        side = 10.0
    
    return inner_hex_data, outer_hex_data, side


# EVOLVE-BLOCK-END
