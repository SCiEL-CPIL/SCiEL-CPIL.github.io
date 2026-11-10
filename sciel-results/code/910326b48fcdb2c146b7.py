# EVOLVE-BLOCK-START
import numpy as np
import time

def _hex_vertices(cx, cy, s, a_deg):
    """Return 6 vertices of a regular hexagon centered at (cx,cy), side s, angle a_deg."""
    a = np.deg2rad(a_deg)
    angles = a + np.deg2rad(60.0 * np.arange(6))
    return np.column_stack([cx + s * np.cos(angles), cy + s * np.sin(angles)])

def _min_outer_side_for_inner(inner_data, outer_center, outer_angle_deg):
    """
    Compute the minimum outer hex side length such that all inner hex vertices
    are inside the outer hexagon centered at outer_center with given angle.
    """
    ox, oy, oa = outer_center
    oa_rad = np.deg2rad(oa)
    
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
    # We need: for each edge i, max over p of dot(n_i, p) <= apothem = s * sqrt(3)/2
    # So s >= (2/sqrt(3)) * max_i (max_p dot(n_i, p))
    
    # Compute max dot for each normal direction
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
    start_time = time.time()
    deadline = start_time + 600.0  # Safety margin
    
    sq3 = np.sqrt(3.0)
    
    # Scaling factor to ensure gap > 1e-6
    # Touching distance is sqrt(3).
    # We want distance > sqrt(3) + 1e-6.
    # Distance scales with k. k*sqrt(3) > sqrt(3) + 1e-6 => k > 1 + 1e-6/sqrt(3).
    # Use a slightly larger epsilon for safety.
    k = 1.0 + 1e-5
    
    # Base positions for a compact cluster (7 hexagons + 4 extensions)
    # Center (0,0)
    # 6 neighbors
    # 4 second-ring neighbors
    
    base_positions = [
        [0.0, 0.0],
        [0.0, sq3],
        [1.5, sq3/2],
        [1.5, -sq3/2],
        [0.0, -sq3],
        [-1.5, -sq3/2],
        [-1.5, sq3/2],
        # Add 4 more to make 11. 
        # To keep it compact, we add to the 'top', 'bottom', 'left', 'right' extremes?
        # The 7-hex flower is symmetric. 
        # Adding to (0, 2*sq3), (0, -2*sq3), (3*1.5, 2*sq3/2 + sq3/2)? No.
        # Let's add to the 4 cardinal-ish directions of the lattice.
        # Up: (0, 2*sq3)
        # Down: (0, -2*sq3)
        # Right-Upper: (1.5+1.5, sq3/2+sq3/2) = (3.0, sq3)
        # Right-Lower: (1.5+1.5, -sq3/2-sq3/2) = (3.0, -sq3)
        # This makes the shape wider on the right.
        # Alternatively, add to Up, Down, Left-Upper, Left-Lower?
        # Let's try a more balanced addition.
        # Add to Up, Down, Left-Upper, Left-Lower?
        # Or just pick any 4. The outer hexagon will adapt.
        # Let's pick: Up, Down, Right-Upper, Right-Lower.
        [0.0, 2.0 * sq3],
        [0.0, -2.0 * sq3],
        [3.0, sq3],
        [3.0, -sq3],
    ]
    
    inner_hex_data = np.array([
        [x * k, y * k, 0.0] for x, y in base_positions
    ])
    
    # Outer hexagon centered at origin
    outer_center = np.array([0.0, 0.0, 0.0])
    
    # Try a few angles for the outer hexagon to find the minimal side
    # The inner cluster has some symmetry, but breaking it might mean a specific angle is better.
    # However, angle 0 is usually a good start for lattice-aligned shapes.
    # We can also try angle 30.
    
    best_side = np.inf
    best_angle = 0.0
    
    for angle in [0.0, 30.0, 60.0, 15.0, 45.0]:
        s = _min_outer_side_for_inner(inner_hex_data, outer_center, angle)
        if s < best_side:
            best_side = s
            best_angle = angle
            
    # Add small margin for numerical safety
    best_side += 1e-6
    
    outer_hex_data = np.array([0.0, 0.0, best_angle])
    
    # Verify validity
    if _check_validity(inner_hex_data, outer_hex_data, best_side):
        return inner_hex_data, outer_hex_data, best_side
    
    # Fallback: increase side further
    best_side += 1.0
    return inner_hex_data, outer_hex_data, best_side


# EVOLVE-BLOCK-END
