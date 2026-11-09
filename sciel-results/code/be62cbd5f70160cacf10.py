# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time
import itertools


def _hex_vertices(cx, cy, s, a_deg):
    """Vertices of a regular hexagon centered at (cx,cy) with side s and angle a_deg."""
    a = np.deg2rad(a_deg)
    angles = a + np.deg2rad(60.0 * np.arange(6))
    return np.column_stack([cx + s * np.cos(angles), cy + s * np.sin(angles)])


def _edge_normals(hex_verts):
    """Return 6 outward unit edge normals for a hexagon given its 6 vertices in order."""
    normals = []
    centroid = hex_verts.mean(axis=0)
    for i in range(6):
        p1 = hex_verts[i]
        p2 = hex_verts[(i + 1) % 6]
        edge = p2 - p1
        n = np.array([-edge[1], edge[0]])
        mid = (p1 + p2) / 2.0
        if np.dot(n, mid - centroid) < 0:
            n = -n
        n = n / np.linalg.norm(n)
        normals.append(n)
    return np.array(normals)


def _separated(v1, n1, v2, n2):
    """
    Check if two hexagons with vertices v1 (6x2) and v2 (6x2) and normals n1, n2 (6x2)
    are separated. Returns True if separated by > 1e-6 on at least one of 12 normals.
    """
    all_normals = np.vstack([n1, n2])  # 12 x 2
    for k in range(12):
        n = all_normals[k]
        proj1 = v1 @ n
        proj2 = v2 @ n
        min1, max1 = proj1.min(), proj1.max()
        min2, max2 = proj2.min(), proj2.max()
        if (max1 < min2 - 1e-6) or (max2 < min1 - 1e-6):
            return True
    return False


def _all_separated(inner_hexes):
    """Check all pairs of inner hexagons are separated."""
    n = len(inner_hexes)
    for i in range(n):
        for j in range(i + 1, n):
            if not _separated(inner_hexes[i][0], inner_hexes[i][1],
                              inner_hexes[j][0], inner_hexes[j][1]):
                return False
    return True


def _min_outer_side(inner_hexes):
    """
    Find the minimum side length S such that all inner hexagons fit inside
    a regular hexagon centered at origin with side S and some angle.
    Returns (S, outer_angle) or (inf, 0) if infeasible.
    """
    def _fits(S, angle_deg):
        outer_verts = _hex_vertices(0, 0, S, angle_deg)
        for hex_verts, normals in inner_hexes:
            for p in hex_verts:
                for i in range(6):
                    p1 = outer_verts[i]
                    p2 = outer_verts[(i + 1) % 6]
                    cross = (p2[0] - p1[0]) * (p[1] - p1[1]) - (p2[1] - p1[1]) * (p[0] - p1[0])
                    if cross < -1e-6:
                        return False
        return True

    # Compute bounding radius
    max_r = 0
    for hex_verts, normals in inner_hexes:
        r = np.max(np.linalg.norm(hex_verts, axis=1))
        max_r = max(max_r, r)

    # Binary search for minimum S with angle=0 first
    S_low = max_r
    S_high = S_low * 2.0

    for _ in range(50):
        S_mid = (S_low + S_high) / 2.0
        if _fits(S_mid, 0.0):
            S_high = S_mid
        else:
            S_low = S_mid

    best_S = S_high
    best_angle = 0.0

    # Try different angles to see if we can do better
    for angle_deg in [0, 10, 20, 30, 40, 50, 60]:
        sl = max_r * 0.95
        sh = max_r * 2.0
        for _ in range(50):
            sm = (sl + sh) / 2.0
            if _fits(sm, angle_deg):
                sh = sm
            else:
                sl = sm
        if sh < best_S:
            best_S = sh
            best_angle = angle_deg

    return best_S, best_angle


def _build_inner_hexes(positions, angles):
    """Build list of (vertices, normals) for each inner hexagon."""
    hexes = []
    for i in range(11):
        verts = _hex_vertices(positions[i, 0], positions[i, 1], 1.0, angles[i])
        norms = _edge_normals(verts)
        hexes.append((verts, norms))
    return hexes


def _hex_lattice_positions(rings):
    """
    Generate positions for hexagons on a hexagonal lattice with given number of rings.
    Each ring k has 6k hexagons at distance k*2.5 from center (for unit hexagons with side 1).
    The hexagonal lattice spacing for non-overlapping unit hexagons is 2.5 (distance between centers).
    
    Returns list of (x, y) positions.
    """
    positions = [(0.0, 0.0)]  # center
    
    for ring in range(1, rings + 1):
        # 6*ring hexagons in this ring
        # Distance from center: ring * 2.5
        dist = ring * 2.5
        for i in range(6 * ring):
            # Angle for this hexagon
            angle = np.deg2rad(60.0 * i / ring)
            x = dist * np.cos(angle)
            y = dist * np.sin(angle)
            positions.append((x, y))
    
    return positions


def _generate_symmetric_arrangements():
    """
    Generate various symmetric arrangements of 11 hexagons.
    Returns list of (positions, angles) tuples.
    """
    arrangements = []
    
    # Arrangement 1: 1 center + 6 in ring 1 + 4 in ring 2 (partial)
    # Ring 1: 6 hexagons at distance 2.5
    # Ring 2: pick 4 out of 12 positions
    
    ring1_dist = 2.5
    ring2_dist = 5.0
    
    # Ring 1 positions
    ring1 = []
    for i in range(6):
        angle = np.deg2rad(60.0 * i)
        x = ring1_dist * np.cos(angle)
        y = ring1_dist * np.sin(angle)
        ring1.append((x, y))
    
    # Ring 2 positions (12 positions, pick 4)
    ring2_all = []
    for i in range(12):
        angle = np.deg2rad(30.0 * i)
        x = ring2_dist * np.cos(angle)
        y = ring2_dist * np.sin(angle)
        ring2_all.append((x, y))
    
    # Try different subsets of 4 from ring2
    # Use symmetric selections
    # Option A: every 3rd (angles 0, 90, 180, 270) - but that's not on hex lattice
    # Better: pick positions that maintain 6-fold or 3-fold symmetry
    
    # 3-fold symmetric: pick 3 pairs or 4 with some symmetry
    # Let's try: indices 0, 3, 6, 9 (every 3rd, 4 positions with 90-degree spacing)
    # But hex lattice has 60-degree symmetry, so let's use:
    # Indices 0, 2, 4, 6, 8, 10 (every other, 6 positions) - too many
    # Indices 0, 3, 6, 9 (4 positions, 90-degree) - breaks 6-fold
    
    # Let's try different ring spacings and arrangements
    
    # Arrangement with 1 center + 6 ring1 + 4 ring2
    # Try ring2 at 60-degree intervals but only 4: not symmetric
    # Instead, let's use a different structure
    
    # Alternative: 1 center + 6 ring1 + 4 in a different pattern
    # Use 4 hexagons at 90-degree intervals in ring2
    for offset in [0, 15, 30, 45, 60]:
        ring2_sel = []
        for i in [0, 3, 6, 9]:
            angle = np.deg2rad(30.0 * i + offset)
            x = ring2_dist * np.cos(angle)
            y = ring2_dist * np.sin(angle)
            ring2_sel.append((x, y))
        
        positions = [(0.0, 0.0)] + ring1 + ring2_sel
        angles = [0.0] * 11
        arrangements.append((np.array(positions), np.array(angles)))
    
    # Arrangement 2: Different ring distances
    # Try compressed ring1
    for r1_scale in [1.0, 0.95, 0.9, 1.05, 1.1]:
        for r2_scale in [1.0, 0.95, 0.9, 1.05, 1.1]:
            r1d = ring1_dist * r1_scale
            r2d = ring2_dist * r2_scale
            
            ring1_s = []
            for i in range(6):
                angle = np.deg2rad(60.0 * i)
                x = r1d * np.cos(angle)
                y = r1d * np.sin(angle)
                ring1_s.append((x, y))
            
            ring2_s = []
            for i in [0, 3, 6, 9]:
                angle = np.deg2rad(30.0 * i)
                x = r2d * np.cos(angle)
                y = r2d * np.sin(angle)
                ring2_s.append((x, y))
            
            positions = [(0.0, 0.0)] + ring1_s + ring2_s
            angles = [0.0] * 11
            arrangements.append((np.array(positions), np.array(angles)))
    
    # Arrangement 3: 1 center + 5 ring1 + 5 ring2 (asymmetric but compact)
    for r1_scale in [1.0, 0.95, 1.05]:
        for r2_scale in [1.0, 0.95, 1.05]:
            r1d = ring1_dist * r1_scale
            r2d = ring2_dist * r2_scale
            
            ring1_s = []
            for i in range(5):
                angle = np.deg2rad(60.0 * i + 30)
                x = r1d * np.cos(angle)
                y = r1d * np.sin(angle)
                ring1_s.append((x, y))
            
            ring2_s = []
            for i in range(5):
                angle = np.deg2rad(60.0 * i)
                x = r2d * np.cos(angle)
                y = r2d * np.sin(angle)
                ring2_s.append((x, y))
            
            positions = [(0.0, 0.0)] + ring1_s + ring2_s
            angles = [0.0] * 11
            arrangements.append((np.array(positions), np.array(angles)))
    
    # Arrangement 4: 2 center cluster + 9 outer
    for dx in [0, 0.5, 1.0, 1.5, 2.0]:
        for dy in [0, 0.5, 1.0, 1.5, 2.0]:
            if dx == 0 and dy == 0:
                continue
            # Two centers at (dx, dy) and (-dx, -dy)
            c1 = (dx, dy)
            c2 = (-dx, -dy)
            
            # 9 hexagons in outer ring
            outer_dist = 4.5
            outer = []
            for i in range(9):
                angle = np.deg2rad(40.0 * i + 20)
                x = outer_dist * np.cos(angle)
                y = outer_dist * np.sin(angle)
                outer.append((x, y))
            
            positions = [c1, c2] + outer
            angles = [0.0] * 11
            arrangements.append((np.array(positions), np.array(angles)))
    
    # Arrangement 5: Original seed arrangement
    seed_positions = np.array(
        [
            [0, 0],
            [-2.5, 0],
            [2.5, 0],
            [-1.25, 2.165],
            [1.25, 2.165],
            [-1.25, -2.165],
            [1.25, -2.165],
            [-3.75, 2.165],
            [3.75, 2.165],
            [-3.75, -2.165],
            [3.75, -2.165],
        ]
    )
    arrangements.append((seed_positions, np.zeros(11)))
    
    # Arrangement 6: Hexagonal lattice with 1+6+4 but different angles
    # Try rotating some hexagons
    for rot in [0, 15, 30, 45, 60]:
        ring1_s = []
        for i in range(6):
            angle = np.deg2rad(60.0 * i)
            x = ring1_dist * np.cos(angle)
            y = ring1_dist * np.sin(angle)
            ring1_s.append((x, y))
        
        ring2_s = []
        for i in [0, 3, 6, 9]:
            angle = np.deg2rad(30.0 * i)
            x = ring2_dist * np.cos(angle)
            y = ring2_dist * np.sin(angle)
            ring2_s.append((x, y))
        
        positions = [(0.0, 0.0)] + ring1_s + ring2_s
        angles = [0.0] + [rot] * 6 + [rot] * 4
        arrangements.append((np.array(positions), np.array(angles)))
    
    return arrangements


def hexagon_packing_11():
    """
    Constructs a packing of 11 disjoint unit regular hexagons inside a larger regular hexagon, maximizing 1/outer_hex_side_length.
    Returns
        inner_hex_data: np.ndarray of shape (11,3), where each row is of the form (x, y, angle_degrees) containing the (x,y) coordinates and angle_degree of the respective inner hexagon.
        outer_hex_data: np.ndarray of shape (3,) of form (x,y,angle_degree) containing the (x,y) coordinates and angle_degree of the outer hexagon.
        outer_hex_side_length: float representing the side length of the outer hexagon.
    """
    t_start = time.time()
    deadline = 600.0
    
    # Generate candidate arrangements
    arrangements = _generate_symmetric_arrangements()
    
    best_S = float('inf')
    best_positions = None
    best_angles = None
    best_angle = 0.0
    
    # Evaluate each arrangement
    for positions, angles in arrangements:
        if time.time() > deadline - 60:
            break
        
        # Check separation
        inner_hexes = _build_inner_hexes(positions, angles)
        if not _all_separated(inner_hexes):
            continue
        
        # Find min outer side
        S, angle = _min_outer_side(inner_hexes)
        if np.isfinite(S) and S < best_S:
            best_S = S
            best_positions = positions.copy()
            best_angles = angles.copy()
            best_angle = angle
    
    # If no valid arrangement found, use the seed
    if best_positions is None:
        seed_positions = np.array(
            [
                [0, 0],
                [-2.5, 0],
                [2.5, 0],
                [-1.25, 2.165],
                [1.25, 2.165],
                [-1.25, -2.165],
                [1.25, -2.165],
                [-3.75, 2.165],
                [3.75, 2.165],
                [-3.75, -2.165],
                [3.75, -2.165],
            ]
        )
        best_positions = seed_positions
        best_angles = np.zeros(11)
        inner_hexes = _build_inner_hexes(best_positions, best_angles)
        best_S, best_angle = _min_outer_side(inner_hexes)
    
    # Local optimization from best arrangement
    if time.time() < deadline - 120:
        x0 = np.zeros(33)
        x0[:22] = best_positions.flatten()
        x0[22:] = best_angles
        
        def _objective(params):
            if time.time() > deadline - 30:
                return 1e10
            
            positions = params[:22].reshape(11, 2)
            angles = params[22:]
            
            inner_hexes = _build_inner_hexes(positions, angles)
            if not _all_separated(inner_hexes):
                return 1e10
            
            S, angle = _min_outer_side(inner_hexes)
            if not np.isfinite(S):
                return 1e10
            return S
        
        try:
            res = minimize(
                _objective,
                x0,
                method='Nelder-Mead',
                options={'maxiter': 1000, 'xatol': 1e-6, 'fatol': 1e-6, 'adaptive': True}
            )
            if res.fun < best_S:
                best_S = res.fun
                best_positions = res.x[:22].reshape(11, 2)
                best_angles = res.x[22:]
                inner_hexes = _build_inner_hexes(best_positions, best_angles)
                best_S, best_angle = _min_outer_side(inner_hexes)
        except Exception:
            pass
    
    # Final result
    inner_hex_data = np.column_stack([best_positions, best_angles])
    outer_hex_data = np.array([0.0, 0.0, best_angle])
    outer_hex_side_length = float(best_S)
    
    return inner_hex_data, outer_hex_data, outer_hex_side_length


# EVOLVE-BLOCK-END
