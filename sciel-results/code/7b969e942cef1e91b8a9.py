# EVOLVE-BLOCK-START
import numpy as np
import time


def _hex_vertices(cx, cy, side, angle_deg):
    """Compute vertices of a regular hexagon."""
    a = np.radians(angle_deg)
    angles = a + np.arange(6) * np.pi / 3
    return np.column_stack([
        cx + side * np.cos(angles),
        cy + side * np.sin(angles)
    ])


def _min_outer_side(vertices, outer_angle_deg=0.0):
    """
    Compute minimum side length s of a regular hexagon (centered at origin, given angle)
    that contains all given vertices.
    """
    a = np.radians(outer_angle_deg)
    # Edge normals for a regular hexagon with angle a
    normal_angles = a + np.pi/6 + np.arange(6) * np.pi / 3
    normals = np.column_stack([np.cos(normal_angles), np.sin(normal_angles)])
    
    dots = vertices @ normals.T
    max_dots = np.max(dots, axis=0)
    s = np.max(max_dots) / (np.sqrt(3) / 2)
    
    return max(s, 0.0)


def _check_separation(inner_hex_data, tol=1e-6):
    """Check that no two inner hexagons overlap using SAT."""
    n = len(inner_hex_data)
    vertices_list = []
    for i in range(n):
        cx, cy, ang = inner_hex_data[i]
        verts = _hex_vertices(cx, cy, 1.0, ang)
        vertices_list.append(verts)
    
    for i in range(n):
        for j in range(i+1, n):
            v1 = vertices_list[i]
            v2 = vertices_list[j]
            separated = False
            for k in range(6):
                p1 = v1[k]
                p2 = v1[(k+1) % 6]
                edge = p2 - p1
                normal = np.array([-edge[1], edge[0]])
                norm_len = np.linalg.norm(normal)
                if norm_len < 1e-12:
                    continue
                normal = normal / norm_len
                proj1 = v1 @ normal
                proj2 = v2 @ normal
                if np.max(proj1) < np.min(proj2) - tol or np.max(proj2) < np.min(proj1) - tol:
                    separated = True
                    break
            if separated:
                continue
            for k in range(6):
                p1 = v2[k]
                p2 = v2[(k+1) % 6]
                edge = p2 - p1
                normal = np.array([-edge[1], edge[0]])
                norm_len = np.linalg.norm(normal)
                if norm_len < 1e-12:
                    continue
                normal = normal / norm_len
                proj1 = v1 @ normal
                proj2 = v2 @ normal
                if np.max(proj1) < np.min(proj2) - tol or np.max(proj2) < np.min(proj1) - tol:
                    separated = True
                    break
            if not separated:
                return False
    return True


def hexagon_packing_12():
    """
    Constructs a packing of 12 disjoint unit regular hexagons inside a larger regular hexagon,
    minimizing the outer hexagon side length.
    """
    start_time = time.time()
    deadline = start_time + 600
    
    sqrt3 = np.sqrt(3)
    
    # For unit hexagons with angle 0 (flat-top), adjacent centers in a hexagonal
    # lattice are at distance sqrt(3). Use basis vectors:
    # v1 = (sqrt(3), 0) - horizontal neighbor
    # v2 = (sqrt(3)/2, 1.5) - diagonal neighbor
    # This gives proper tiling with no overlaps.
    
    v1 = np.array([sqrt3, 0.0])
    v2 = np.array([sqrt3/2, 1.5])
    
    # Define 12 positions using axial coordinates (i, j) -> i*v1 + j*v2
    # A compact cluster: center + 6 surrounding + 5 more
    axial_coords = [
        (0, 0),    # center
        (1, 0), (-1, 0),   # horizontal neighbors
        (0, 1), (0, -1),   # vertical-ish neighbors
        (1, -1), (-1, 1),  # diagonal neighbors
        # 5 more to make 12
        (2, 0),    # further right
        (-2, 0),   # further left
        (0, 2),    # further up
        (0, -2),   # further down
        (2, -1),   # diagonal
    ]
    
    positions = np.array([i * v1[0] + j * v2[0], i * v1[1] + j * v2[1]]).T
    positions = positions[:12]
    
    # Center the arrangement
    center = np.mean(positions, axis=0)
    positions = positions - center
    
    # Create inner hex data with angle 0
    inner_hex_data = np.zeros((12, 3))
    for i in range(12):
        inner_hex_data[i, 0] = positions[i, 0]
        inner_hex_data[i, 1] = positions[i, 1]
        inner_hex_data[i, 2] = 0
    
    # Verify separation
    if not _check_separation(inner_hex_data):
        # Fallback: use a simpler valid configuration
        # 7 hexagons in a flower pattern + 5 more
        inner_hex_data = np.array([
            [0, 0, 0],
            [sqrt3, 0, 0],
            [-sqrt3, 0, 0],
            [sqrt3/2, 1.5, 0],
            [-sqrt3/2, 1.5, 0],
            [sqrt3/2, -1.5, 0],
            [-sqrt3/2, -1.5, 0],
            [2*sqrt3, 0, 0],
            [-2*sqrt3, 0, 0],
            [0, 3, 0],
            [0, -3, 0],
            [3*sqrt3/2, 1.5, 0],
        ])
        center = np.mean(inner_hex_data[:, :2], axis=0)
        inner_hex_data[:, 0] -= center[0]
        inner_hex_data[:, 1] -= center[1]
    
    # Compute all vertices
    all_vertices = []
    for i in range(12):
        cx, cy, ang = inner_hex_data[i]
        verts = _hex_vertices(cx, cy, 1.0, ang)
        all_vertices.append(verts)
    all_vertices = np.vstack(all_vertices)
    
    # Optimize outer hexagon angle to minimize s
    best_s = np.inf
    best_angle = 0.0
    for angle_deg in np.arange(0, 30, 0.5):
        s = _min_outer_side(all_vertices, angle_deg)
        if s < best_s:
            best_s = s
            best_angle = angle_deg
    
    # Refine
    for angle_deg in np.linspace(best_angle - 1, best_angle + 1, 21):
        s = _min_outer_side(all_vertices, angle_deg)
        if s < best_s:
            best_s = s
            best_angle = angle_deg
    
    outer_hex_data = np.array([0.0, 0.0, best_angle])
    
    elapsed = time.time() - start_time
    if elapsed > deadline - 5:
        return inner_hex_data, outer_hex_data, best_s
    
    return inner_hex_data, outer_hex_data, best_s


# EVOLVE-BLOCK-END
