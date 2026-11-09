# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time


def _hex_vertices(cx, cy, s, a_deg):
    """Return 6 vertices of a regular hexagon (center, side, angle in degrees)."""
    a = np.radians(a_deg)
    angles = a + np.arange(6) * np.pi / 3.0
    return np.column_stack([cx + s * np.cos(angles), cy + s * np.sin(angles)])


def _hex_edge_normals(cx, cy, s, a_deg):
    """Return 12 edge normal directions (unit vectors) for a hexagon."""
    verts = _hex_vertices(cx, cy, s, a_deg)
    normals = np.zeros((12, 2))
    for i in range(6):
        v1 = verts[i]
        v2 = verts[(i + 1) % 6]
        edge = v2 - v1
        # outward normal (perpendicular, rotated -90 deg from edge direction)
        n = np.array([-edge[1], edge[0]])
        n /= np.linalg.norm(n)
        # ensure outward: check against center
        mid = (v1 + v2) / 2.0
        if np.dot(n, mid - np.array([cx, cy])) < 0:
            n = -n
        normals[i] = n
        normals[i + 6] = -n
    return normals


def _hexagons_separated(hex1, hex2, eps=1e-6):
    """Check if two unit hexagons are separated (gap > eps on at least one normal)."""
    cx1, cy1, a1 = hex1
    cx2, cy2, a2 = hex2
    normals1 = _hex_edge_normals(cx1, cy1, 1.0, a1)
    normals2 = _hex_edge_normals(cx2, cy2, 1.0, a2)
    all_normals = np.vstack([normals1, normals2])
    
    v1 = _hex_vertices(cx1, cy1, 1.0, a1)
    v2 = _hex_vertices(cx2, cy2, 1.0, a2)
    
    for n in all_normals:
        proj1 = np.dot(v1, n)
        proj2 = np.dot(v2, n)
        min1, max1 = proj1.min(), proj1.max()
        min2, max2 = proj2.min(), proj2.max()
        # Check if intervals are apart by more than eps
        if max1 + eps < min2 or max2 + eps < min1:
            return True
    return False


def _hexagon_in_outer(inner_hex, outer_center, outer_side, outer_angle, eps=1e-6):
    """Check if inner hexagon is inside outer hexagon."""
    cx, cy, a = inner_hex
    verts = _hex_vertices(cx, cy, 1.0, a)
    outer_verts = _hex_vertices(outer_center[0], outer_center[1], outer_side, outer_angle)
    
    for i in range(6):
        p1 = outer_verts[i]
        p2 = outer_verts[(i + 1) % 6]
        edge = p2 - p1
        # cross product for each vertex of inner hex
        for p in verts:
            cross = np.cross(p2 - p1, p - p1)
            if cross < -eps:
                return False
    return True


def _min_enclosing_hex_side(inner_positions, inner_angles, global_rotation):
    """Compute minimal enclosing hexagon side length for given hexagon positions.
    
    inner_positions: (11, 2) array of (x, y) centers (before global rotation)
    inner_angles: (11,) array of angles in degrees (before global rotation)
    global_rotation: rotation angle in degrees applied to all positions
    """
    rot = np.radians(global_rotation)
    cos_r, sin_r = np.cos(rot), np.sin(rot)
    
    rotated_positions = np.column_stack([
        inner_positions[:, 0] * cos_r - inner_positions[:, 1] * sin_r,
        inner_positions[:, 0] * sin_r + inner_positions[:, 1] * cos_r
    ])
    
    # For a regular hexagon centered at origin with side s and angle a,
    # the constraint for a point p to be inside is:
    # For each of 6 edge normals n_i, dot(p, n_i) <= s * cos(30 deg) = s * sqrt(3)/2
    # Actually, for a regular hexagon with side s, the apothem is s*sqrt(3)/2
    # The edge normals are at angles a + 30 + 60i degrees
    # The constraint is dot(p, n_i) <= apothem = s * sqrt(3)/2
    
    # To find minimal s, we need s >= max_i (dot(p, n_i)) / (sqrt(3)/2) for all p and all i
    # But we also need to consider the outer hexagon angle. Let's fix outer angle to 0 for simplicity.
    
    # For outer hexagon at origin with angle 0, edge normals are at 30, 90, 150, 210, 270, 330 degrees
    normal_angles = np.radians(30 + np.arange(6) * 60)
    normals = np.column_stack([np.cos(normal_angles), np.sin(normal_angles)])
    
    apothem = np.sqrt(3) / 2.0
    
    # For each rotated inner hexagon, compute the maximum dot product with each normal
    # considering all 6 vertices of the inner hexagon
    max_proj = 0.0
    for i in range(len(rotated_positions)):
        cx, cy = rotated_positions[i]
        angle = inner_angles[i] + global_rotation
        verts = _hex_vertices(cx, cy, 1.0, angle)
        for n in normals:
            proj = np.dot(verts, n)
            max_proj = max(max_proj, proj.max())
    
    # s = max_proj / apothem
    return max_proj / apothem


def _check_separations(inner_positions, inner_angles, global_rotation, eps=1e-6):
    """Check all pairs of inner hexagons are separated."""
    rot = np.radians(global_rotation)
    cos_r, sin_r = np.cos(rot), np.sin(rot)
    
    rotated_positions = np.column_stack([
        inner_positions[:, 0] * cos_r - inner_positions[:, 1] * sin_r,
        inner_positions[:, 0] * sin_r + inner_positions[:, 1] * cos_r
    ])
    
    n = len(rotated_positions)
    for i in range(n):
        for j in range(i + 1, n):
            hex_i = [rotated_positions[i, 0], rotated_positions[i, 1], inner_angles[i] + global_rotation]
            hex_j = [rotated_positions[j, 0], rotated_positions[j, 1], inner_angles[j] + global_rotation]
            if not _hexagons_separated(hex_i, hex_j, eps):
                return False
    return True


def hexagon_packing_11():
    """
    Constructs a packing of 11 disjoint unit regular hexagons inside a larger regular hexagon,
    using 6-fold rotational symmetry with 1 center + 6 inner ring + 4 outer ring hexagons.
    
    Returns
        inner_hex_data: np.ndarray of shape (11,3), where each row is (x, y, angle_degrees)
        outer_hex_data: np.ndarray of shape (3,) of form (x, y, angle_degrees)
        outer_hex_side_length: float
    """
    start_time = time.time()
    deadline = start_time + 600.0
    safety_margin = 30.0  # seconds before deadline
    
    # 6-fold symmetric configuration:
    # Hex 0: center at origin
    # Hexes 1-6: inner ring at radius r1, angles 0, 60, 120, 180, 240, 300
    # Hexes 7-10: outer ring at radius r2, angles theta, theta+120, theta+240, theta+360 (but only 4, so theta, theta+120, theta+240, and one more)
    # Actually for 6-fold symmetry with 11 hexagons: 1 + 6 + 4 doesn't work for 6-fold.
    # Let me reconsider: 1 center + 6 inner + 4 outer. The 4 outer break 6-fold symmetry.
    # Alternative: 1 center + 6 inner + 4 outer where outer are at 0, 120, 240 and one at 60? No.
    # Let's try: 1 center + 6 inner ring + 4 outer ring at angles 30, 150, 270, and... 
    # Actually, let me just use 1 + 6 + 4 with the 4 outer at 0, 120, 240, 60 (not symmetric)
    # Or better: just optimize free positions with symmetry constraints.
    
    # Let me use a simpler approach: 1 center + 6 inner + 4 outer
    # Inner ring: 6 hexagons at radius r1, angles 0, 60, 120, 180, 240, 300
    # Outer ring: 4 hexagons at radius r2, angles 30, 150, 270, and 90? 
    # For 6-fold symmetry we'd need 6 outer. Let's do 1 + 6 + 6 = 13, too many.
    # 1 + 6 + 4: let's place outer at 0, 120, 240, and 60 (breaking symmetry slightly)
    # Actually, let me just do free optimization with initial guess from symmetry.
    
    # Initial configuration: 1 center + 6 inner ring + 4 outer ring
    r1_init = 2.0
    r2_init = 4.0
    theta_outer = 30.0  # offset for outer ring
    
    inner_positions = np.zeros((11, 2))
    inner_angles = np.zeros(11)
    
    # Center
    inner_positions[0] = [0, 0]
    inner_angles[0] = 0
    
    # Inner ring (6 hexagons)
    for i in range(6):
        angle = np.radians(i * 60)
        inner_positions[i + 1] = [r1_init * np.cos(angle), r1_init * np.sin(angle)]
        inner_angles[i + 1] = 0
    
    # Outer ring (4 hexagons) at angles 30, 150, 270, 90 (not 6-fold symmetric, but close)
    outer_angles_deg = [30, 150, 270, 90]
    for i, ang in enumerate(outer_angles_deg):
        angle = np.radians(ang)
        inner_positions[i + 7] = [r2_init * np.cos(angle), r2_init * np.sin(angle)]
        inner_angles[i + 7] = 0
    
    # Optimize: r1, r2, global_rotation
    def objective(params):
        r1, r2, global_rot = params
        
        # Check time
        if time.time() > deadline - safety_margin:
            return 1e10
        
        positions = np.zeros((11, 2))
        angles = np.zeros(11)
        
        positions[0] = [0, 0]
        angles[0] = 0
        
        for i in range(6):
            angle = np.radians(i * 60)
            positions[i + 1] = [r1 * np.cos(angle), r1 * np.sin(angle)]
            angles[i + 1] = 0
        
        for i, ang in enumerate(outer_angles_deg):
            angle = np.radians(ang)
            positions[i + 7] = [r2 * np.cos(angle), r2 * np.sin(angle)]
            angles[i + 7] = 0
        
        # Check separations
        if not _check_separations(positions, angles, global_rot, eps=1e-6):
            return 1e10
        
        # Compute enclosing hexagon side
        side = _min_enclosing_hex_side(positions, angles, global_rot)
        return side
    
    # Try multiple initial guesses
    best_result = None
    best_side = np.inf
    
    initial_guesses = [
        [2.0, 4.0, 0.0],
        [2.1, 4.2, 15.0],
        [2.2, 4.4, 30.0],
        [1.9, 3.8, 0.0],
        [2.0, 4.0, 60.0],
        [2.5, 5.0, 0.0],
        [2.0, 4.5, 45.0],
        [2.3, 4.6, 20.0],
    ]
    
    for guess in initial_guesses:
        if time.time() > deadline - safety_margin:
            break
        
        try:
            result = minimize(objective, guess, method='Nelder-Mead',
                             options={'maxiter': 500, 'xatol': 1e-6, 'fatol': 1e-8})
            
            if result.fun < best_side and result.fun < 1e9:
                best_side = result.fun
                best_result = result.x
        except Exception:
            continue
    
    if best_result is None:
        # Fallback: use initial configuration
        best_result = np.array([2.0, 4.0, 0.0])
        best_side = _min_enclosing_hex_side(inner_positions, inner_angles, 0.0)
    
    r1, r2, global_rot = best_result
    
    # Build final configuration
    final_positions = np.zeros((11, 2))
    final_angles = np.zeros(11)
    
    final_positions[0] = [0, 0]
    final_angles[0] = 0
    
    for i in range(6):
        angle = np.radians(i * 60)
        final_positions[i + 1] = [r1 * np.cos(angle), r1 * np.sin(angle)]
        final_angles[i + 1] = 0
    
    for i, ang in enumerate(outer_angles_deg):
        angle = np.radians(ang)
        final_positions[i + 7] = [r2 * np.cos(angle), r2 * np.sin(angle)]
        final_angles[i + 7] = 0
    
    # Apply global rotation
    rot = np.radians(global_rot)
    cos_r, sin_r = np.cos(rot), np.sin(rot)
    rotated_positions = np.column_stack([
        final_positions[:, 0] * cos_r - final_positions[:, 1] * sin_r,
        final_positions[:, 0] * sin_r + final_positions[:, 1] * cos_r
    ])
    rotated_angles = final_angles + global_rot
    
    # Compute final enclosing hexagon
    outer_side = _min_enclosing_hex_side(final_positions, final_angles, global_rot)
    
    # Verify validity
    if not _check_separations(final_positions, final_angles, global_rot, eps=1e-6):
        # Try to fix by scaling up
        for scale in [1.01, 1.02, 1.05, 1.1]:
            scaled_positions = final_positions * scale
            if _check_separations(scaled_positions, final_angles, global_rot, eps=1e-6):
                final_positions = scaled_positions
                outer_side = _min_enclosing_hex_side(scaled_positions, final_angles, global_rot)
                break
    
    # Verify all hexagons are inside outer hexagon
    outer_center = np.array([0.0, 0.0])
    outer_angle = 0.0
    for i in range(11):
        hex_i = [rotated_positions[i, 0], rotated_positions[i, 1], rotated_angles[i]]
        if not _hexagon_in_outer(hex_i, outer_center, outer_side, outer_angle, eps=1e-6):
            # Scale up outer side slightly
            outer_side *= 1.01
    
    # Build return values
    inner_hex_data = np.column_stack([rotated_positions, rotated_angles])
    outer_hex_data = np.array([0.0, 0.0, 0.0])
    
    return inner_hex_data, outer_hex_data, float(outer_side)


# EVOLVE-BLOCK-END
