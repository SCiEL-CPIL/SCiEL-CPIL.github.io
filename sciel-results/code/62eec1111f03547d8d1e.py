# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time


def _hex_vertices(cx, cy, s, a_deg):
    """Return 6 vertices of a regular hexagon centered at (cx,cy) with side s and angle a_deg."""
    a = np.deg2rad(a_deg)
    angles = a + np.deg2rad(60 * np.arange(6))
    return np.column_stack([cx + s * np.cos(angles), cy + s * np.sin(angles)])


def _compute_outer_side_fast(inner_hex_data, outer_center):
    """
    Compute the minimum side length of a regular hexagon centered at outer_center
    (with angle 0) that contains all vertices of all inner hexagons.
    A regular hexagon with side s and angle 0 has vertices at angles 0,60,...,300.
    The containment condition: for each edge of the outer hexagon, all inner vertices
    must be on the interior side.
    
    For a regular hexagon centered at origin with side s, the support function in direction
    theta is s * cos(theta - nearest_edge_normal). The 6 edge normals are at angles
    30, 90, 150, 210, 270, 330 degrees (perpendicular to edges).
    
    Actually, for a regular hexagon with vertices at angles 0,60,...,300 (pointy-top),
    the edges are between consecutive vertices. The outward normal of edge i (between vertex i and i+1)
    is at angle (60*i + 30) degrees.
    
    The distance from center to each edge (apothem) is s * cos(30 deg) = s * sqrt(3)/2.
    
    So for a point p relative to center, the constraint is:
    dot(p, n_i) <= s * sqrt(3)/2 for all 6 normals n_i.
    
    Thus s >= max_i dot(p_i, n_i) / (sqrt(3)/2) for all points p_i.
    """
    # Get all vertices of all inner hexagons
    all_verts = []
    for i in range(len(inner_hex_data)):
        x, y, ang = inner_hex_data[i]
        verts = _hex_vertices(x, y, 1.0, ang)
        all_verts.append(verts)
    all_verts = np.vstack(all_verts)
    
    # Shift by outer center
    all_verts = all_verts - outer_center
    
    # 6 outward normals of the outer hexagon (angle 0, pointy-top)
    # Vertices at angles 0, 60, 120, 180, 240, 300
    # Edge between vertex i and i+1 has outward normal at angle 60*i + 30
    normal_angles = np.deg2rad(30 + 60 * np.arange(6))
    normals = np.column_stack([np.cos(normal_angles), np.sin(normal_angles)])
    
    # For each normal, compute max dot product over all vertices
    dots = all_verts @ normals.T  # shape (n_verts, 6)
    max_dots = dots.max(axis=0)  # shape (6,)
    
    # s * sqrt(3)/2 >= max_dots[i] for each i
    # s >= 2 * max_dots[i] / sqrt(3)
    s = np.max(max_dots) * 2.0 / np.sqrt(3.0)
    
    # Ensure s is positive
    s = max(s, 0.0)
    
    return s


def _check_validity(inner_hex_data, outer_hex_data, side):
    """Check if the packing is valid."""
    n = len(inner_hex_data)
    if n != 11:
        return False
    
    # Check all values finite
    if not np.all(np.isfinite(inner_hex_data)):
        return False
    if not np.all(np.isfinite(outer_hex_data)):
        return False
    if not np.isfinite(side):
        return False
    
    if side < np.sqrt(11.0) - 1e-9:
        return False
    
    # Check pairwise separation
    for i in range(n):
        for j in range(i + 1, n):
            vi = _hex_vertices(inner_hex_data[i, 0], inner_hex_data[i, 1], 1.0, inner_hex_data[i, 2])
            vj = _hex_vertices(inner_hex_data[j, 0], inner_hex_data[j, 1], 1.0, inner_hex_data[j, 2])
            
            # Check on all 12 edge normals (6 from each hexagon)
            # For hexagon i, edge normals are at angles ang_i + 30 + 60*k
            # For hexagon j, edge normals are at angles ang_j + 30 + 60*k
            separated = False
            for hex_idx, verts in [(i, vi), (j, vj)]:
                ang = inner_hex_data[hex_idx, 2]
                for k in range(6):
                    normal_angle = np.deg2rad(ang + 30 + 60 * k)
                    nx, ny = np.cos(normal_angle), np.sin(normal_angle)
                    
                    # Project both hexagons onto this normal
                    proj_i = verts @ np.array([nx, ny])
                    proj_j = vj @ np.array([nx, ny])
                    
                    # Intervals
                    min_i, max_i = proj_i.min(), proj_i.max()
                    min_j, max_j = proj_j.min(), proj_j.max()
                    
                    # Check if intervals are apart by more than 1e-6
                    if max_i < min_j - 1e-6 or max_j < min_i - 1e-6:
                        separated = True
                        break
                if separated:
                    break
            if not separated:
                return False
    
    # Check all inner vertices are inside outer hexagon
    outer_x, outer_y, outer_ang = outer_hex_data
    outer_verts = _hex_vertices(outer_x, outer_y, side, outer_ang)
    
    for i in range(n):
        vi = _hex_vertices(inner_hex_data[i, 0], inner_hex_data[i, 1], 1.0, inner_hex_data[i, 2])
        for p in vi:
            # Check cross product condition for each outer edge
            inside = True
            for e in range(6):
                p1 = outer_verts[e]
                p2 = outer_verts[(e + 1) % 6]
                cross = (p2[0] - p1[0]) * (p[1] - p1[1]) - (p2[1] - p1[1]) * (p[0] - p1[0])
                if cross < -1e-6:
                    inside = False
                    break
            if not inside:
                return False
    
    # Check edge lengths of shifted inner hexagons
    for i in range(n):
        vi = _hex_vertices(inner_hex_data[i, 0], outer_hex_data[0], 1.0, inner_hex_data[i, 2])
        vi = vi - outer_hex_data[:2]
        for e in range(6):
            d = np.linalg.norm(vi[(e + 1) % 6] - vi[e])
            if abs(d - 1.0) > 1e-8:
                return False
    
    return True


def hexagon_packing_11():
    """
    Constructs a packing of 11 disjoint unit regular hexagons inside a larger regular hexagon,
    minimizing the outer hexagon side length.
    """
    start_time = time.time()
    deadline = 620.0  # safety margin before 630s limit
    
    # Initial configuration: flower pattern
    # Center hexagon at origin
    # 6 hexagons in a ring at distance 2 (touching)
    # 4 more hexagons in the gaps
    
    # Ring 1: 6 hexagons at distance 2 from center, angles 0, 60, 120, 180, 240, 300
    ring1_angles = np.arange(6) * 60.0
    ring1_dist = 2.0
    ring1 = np.column_stack([
        ring1_dist * np.cos(np.deg2rad(ring1_angles)),
        ring1_dist * np.sin(np.deg2rad(ring1_angles)),
        np.zeros(6)
    ])
    
    # Ring 2: 4 hexagons in the gaps between ring1 hexagons
    # The gaps are at angles 30, 90, 150, 210, 270, 330 (6 gaps)
    # We need 4 more, so pick 4 of the 6 gap positions
    # For symmetry, pick every other gap: 30, 150, 270, 30? No, that's only 3 unique with symmetry.
    # Let's try: 30, 90, 150, 210 (4 consecutive gaps)
    # Or for better symmetry: 30, 150, 270, and... we need 4.
    # Let me try 4 hexagons at angles 30, 90, 150, 210 with appropriate distance.
    
    # Actually, let me think more carefully. The 11 hexagons in the smallest hexagon:
    # A known optimal packing uses a specific arrangement.
    # Let me try: 1 center, 6 in first ring (dist 2), 4 in second ring.
    # The 4 in the second ring should be at the "corner" positions.
    
    # For a hexagonal lattice, the second ring positions at distance 2*sqrt(3) ≈ 3.464
    # at angles 30, 90, 150, 210, 270, 330. Pick 4 of these.
    
    # Let me try a different approach: use optimization from a reasonable starting point.
    
    # Starting positions:
    # Hex 0: center (0, 0)
    # Hex 1-6: ring at dist 2, angles 0, 60, 120, 180, 240, 300
    # Hex 7-10: 4 more at the "outer" positions
    
    # Let me try placing hex 7-10 at the midpoints of the outer edges of the first ring.
    # The first ring forms a hexagon with "vertices" at the 6 hex centers.
    # The outer boundary of the 7 hexagons (center + 6) is a larger hexagon.
    # The 4 additional hexagons go in 4 of the 6 "indentations" or "corners".
    
    # Actually, for 11 hexagons, a good arrangement might be:
    # 1 center
    # 6 in first ring (dist 2)
    # 4 in second ring at specific positions
    
    # Let me just set up initial positions and optimize.
    
    initial_positions = np.array([
        [0.0, 0.0, 0.0],          # 0: center
        [2.0, 0.0, 0.0],          # 1
        [1.0, np.sqrt(3), 0.0],   # 2
        [-1.0, np.sqrt(3), 0.0],  # 3
        [-2.0, 0.0, 0.0],         # 4
        [-1.0, -np.sqrt(3), 0.0], # 5
        [1.0, -np.sqrt(3), 0.0],  # 6
        [3.0, np.sqrt(3), 0.0],   # 7
        [-3.0, np.sqrt(3), 0.0],  # 8
        [-3.0, -np.sqrt(3), 0.0], # 9
        [3.0, -np.sqrt(3), 0.0],  # 10
    ])
    
    best_side = _compute_outer_side_fast(initial_positions, np.array([0.0, 0.0]))
    best_config = initial_positions.copy()
    best_outer = np.array([0.0, 0.0, 0.0])
    
    # Optimize positions to minimize outer side length
    # Variables: x, y for each of 11 hexagons (angles fixed at 0 for now)
    
    def objective(params):
        if time.time() > deadline:
            return best_side * 1000  # penalty
        positions = params.reshape(11, 2)
        inner = np.column_stack([positions, np.zeros(11)])
        # Quick check for overlaps (approximate: center distance > 2)
        for i in range(11):
            for j in range(i + 1, 11):
                d = np.linalg.norm(positions[i] - positions[j])
                if d < 1.9:  # too close
                    return 1000.0
        side = _compute_outer_side_fast(inner, np.array([0.0, 0.0]))
        return side
    
    # Use Nelder-Mead optimization
    x0 = initial_positions[:, :2].flatten()
    
    result = minimize(objective, x0, method='Nelder-Mead',
                      options={'maxiter': 500, 'xatol': 1e-8, 'fatol': 1e-10, 'adaptive': True})
    
    if result.fun < best_side:
        positions = result.x.reshape(11, 2)
        inner = np.column_stack([positions, np.zeros(11)])
        best_side = result.fun
        best_config = inner.copy()
    
    # Now optimize angles too
    def objective_full(params):
        if time.time() > deadline:
            return best_side * 1000
        positions = params[:22].reshape(11, 2)
        angles = params[22:]
        inner = np.column_stack([positions, angles])
        # Quick overlap check
        for i in range(11):
            for j in range(i + 1, 11):
                d = np.linalg.norm(positions[i] - positions[j])
                if d < 1.5:
                    return 1000.0
        side = _compute_outer_side_fast(inner, np.array([0.0, 0.0]))
        return side
    
    x0_full = np.concatenate([best_config[:, :2].flatten(), best_config[:, 2]])
    
    result2 = minimize(objective_full, x0_full, method='Nelder-Mead',
                       options={'maxiter': 300, 'xatol': 1e-8, 'fatol': 1e-10, 'adaptive': True})
    
    if result2.fun < best_side:
        positions = result2.x[:22].reshape(11, 2)
        angles = result2.x[22:]
        inner = np.column_stack([positions, angles])
        best_side = result2.fun
        best_config = inner.copy()
    
    # Also try optimizing outer center
    def objective_center(params):
        if time.time() > deadline:
            return best_side * 1000
        ocx, ocy = params
        side = _compute_outer_side_fast(best_config, np.array([ocx, ocy]))
        return side
    
    x0_center = np.array([0.0, 0.0])
    result3 = minimize(objective_center, x0_center, method='Nelder-Mead',
                       options={'maxiter': 200, 'xatol': 1e-8, 'fatol': 1e-10, 'adaptive': True})
    
    if result3.fun < best_side:
        best_side = result3.fun
        best_outer = np.array([result3.x[0], result3.x[1], 0.0])
    
    # Final fine-tuning
    def objective_all(params):
        if time.time() > deadline:
            return best_side * 1000
        positions = params[:22].reshape(11, 2)
        angles = params[22:33]
        ocx, ocy = params[33:35]
        inner = np.column_stack([positions, angles])
        # Quick overlap check
        for i in range(11):
            for j in range(i + 1, 11):
                d = np.linalg.norm(positions[i] - positions[j])
                if d < 1.5:
                    return 1000.0
        side = _compute_outer_side_fast(inner, np.array([ocx, ocy]))
        return side
    
    x0_all = np.concatenate([
        best_config[:, :2].flatten(),
        best_config[:, 2],
        best_outer[:2]
    ])
    
    result4 = minimize(objective_all, x0_all, method='Nelder-Mead',
                       options={'maxiter': 200, 'xatol': 1e-8, 'fatol': 1e-10, 'adaptive': True})
    
    if result4.fun < best_side:
        positions = result4.x[:22].reshape(11, 2)
        angles = result4.x[22:33]
        ocx, ocy = result4.x[33:35]
        inner = np.column_stack([positions, angles])
        best_side = result4.fun
        best_config = inner.copy()
        best_outer = np.array([ocx, ocy, 0.0])
    
    # Verify validity
    if not _check_validity(best_config, best_outer, best_side):
        # Fall back to a known valid configuration
        inner_hex_data = np.array(
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
        outer_hex_data = np.array([0, 0, 0])
        outer_hex_side_length = 8.0
    else:
        inner_hex_data = best_config
        outer_hex_data = best_outer
        outer_hex_side_length = best_side
    
    return inner_hex_data, outer_hex_data, outer_hex_side_length


# EVOLVE-BLOCK-END
