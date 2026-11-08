"""Optimized hexagon packing: 11 unit hexagons in a regular hexagon.
Uses hex-lattice enumeration with continuous outer-hexagon optimization."""
import numpy as np
import time
from itertools import combinations

N = 11
DEADLINE = 130.0  # safety margin before 150s kill

def _hex_vertices(cx, cy, side, angle_deg):
    """Return 6 vertices of a regular hexagon centered at (cx,cy)."""
    angle = np.radians(angle_deg)
    verts = []
    for i in range(6):
        theta = angle + np.pi * i / 3.0
        x = cx + side * np.cos(theta)
        y = cy + side * np.sin(theta)
        verts.append([x, y])
    return np.array(verts)

def _hex_edge_normals(angle_deg):
    """Return the 6 inward unit normals for a hexagon with given rotation."""
    angle = np.radians(angle_deg)
    normals = []
    for i in range(6):
        # Edge between vertex i and vertex i+1; inward normal
        theta1 = angle + np.pi * i / 3.0
        theta2 = angle + np.pi * (i + 1) / 3.0
        # Edge direction
        ex = np.cos(theta2) - np.cos(theta1)
        ey = np.sin(theta2) - np.sin(theta1)
        # Inward normal (rotate edge dir by -90 deg, then normalize)
        nx, ny = -ey, ex
        # For a regular hexagon, the inward normal points toward center
        # The midpoint of edge i is at angle (theta1+theta2)/2 = angle + (i+0.5)*pi/3
        # Inward normal should point opposite to that direction
        mid_angle = angle + (i + 0.5) * np.pi / 3.0
        # Inward normal is -cos(mid_angle), -sin(mid_angle)
        nx2 = -np.cos(mid_angle)
        ny2 = -np.sin(mid_angle)
        normals.append([nx2, ny2])
    return np.array(normals)

def _min_enclosing_side(centers, angles, outer_center, outer_angle):
    """Compute minimum side length of outer hexagon (at given center/angle)
    that encloses all inner hexagons. Returns (side, valid)."""
    # For each inner hexagon, for each of its 6 edges (defined by inward normal),
    # the constraint is: dot(n, v - oc) <= s * r_in  for all vertices v
    # where r_in = sqrt(3)/2 * side_inner = sqrt(3)/2
    # Actually, for a regular hexagon with circumradius R, the inradius is R*sqrt(3)/2.
    # The distance from center to each edge is the inradius.
    # So the constraint for an inner hex vertex v to be inside outer hex:
    #   For each of the 6 inward normals n_k of the outer hex:
    #     dot(n_k, v - oc) <= outer_side * sqrt(3)/2
    # We need outer_side >= max over all (v, k) of dot(n_k, v - oc) / (sqrt(3)/2)
    
    r_in = np.sqrt(3) / 2.0
    outer_normals = _hex_edge_normals(outer_angle)
    
    max_val = 0.0
    for i in range(N):
        cx, cy = centers[i]
        verts = _hex_vertices(cx, cy, 1.0, angles[i])
        for v in verts:
            d = v - np.array(outer_center)
            for k in range(6):
                val = np.dot(outer_normals[k], d)
                if val > max_val:
                    max_val = val
    
    side = max_val / r_in
    if side < 0:
        side = 0.0
    return side, True

def _check_validity(centers, angles, outer_center, outer_side, outer_angle):
    """Check if configuration is valid: all inner hexes inside outer, no overlaps."""
    # Check inner hexes are inside outer
    outer_verts = _hex_vertices(outer_center[0], outer_center[1], outer_side, outer_angle)
    # For each inner vertex, check it's inside outer hex
    for i in range(N):
        verts = _hex_vertices(centers[i][0], centers[i][1], 1.0, angles[i])
        for v in verts:
            if not _point_in_convex(v, outer_verts):
                return False
    # Check no overlaps between inner hexagons
    all_verts = [_hex_vertices(centers[i][0], centers[i][1], 1.0, angles[i]) for i in range(N)]
    for i in range(N):
        for j in range(i + 1, N):
            if _hex_overlap(all_verts[i], all_verts[j]):
                return False
    return True

def _point_in_convex(p, poly):
    """Check if point p is inside convex polygon poly (CCW or CW)."""
    n = len(poly)
    sign = 0
    for i in range(n):
        x1, y1 = poly[i]
        x2, y2 = poly[(i + 1) % n]
        cross = (x2 - x1) * (p[1] - y1) - (y2 - y1) * (p[0] - x1)
        if cross > 1e-9:
            if sign < 0:
                return False
            sign = 1
        elif cross < -1e-9:
            if sign > 0:
                return False
            sign = -1
    return True

def _hex_overlap(v1, v2):
    """Separating axis theorem for two convex hexagons."""
    def get_axes(verts):
        axes = []
        n = len(verts)
        for i in range(n):
            x1, y1 = verts[i]
            x2, y2 = verts[(i + 1) % n]
            ex, ey = x2 - x1, y2 - y1
            norm = np.sqrt(ex*ex + ey*ey)
            if norm > 1e-12:
                axes.append((-ey/norm, ex/norm))
        return axes
    
    axes = get_axes(v1) + get_axes(v2)
    for ax, ay in axes:
        proj1 = np.array([vx*ax + vy*ay for vx, vy in v1])
        proj2 = np.array([vx*ax + vy*ay for vx, vy in v2])
        if proj1.max() < proj2.min() - 1e-9 or proj2.max() < proj1.min() - 1e-9:
            return False
    return True

def _hex_lattice_positions():
    """Generate hex-lattice positions for hexagons of side 1.
    In a hexagonal tiling, adjacent hexagons have centers at distance 2*sqrt(3)/2 = sqrt(3).
    Lattice vectors: a1 = (sqrt(3), 0), a2 = (sqrt(3)/2, 3/2)
    """
    a1 = np.array([np.sqrt(3), 0.0])
    a2 = np.array([np.sqrt(3)/2, 1.5])
    positions = []
    for i in range(-3, 4):
        for j in range(-3, 4):
            pos = i * a1 + j * a2
            positions.append(tuple(pos))
    return positions

def _optimize_outer(centers, angles):
    """Optimize outer hexagon center and angle to minimize enclosing side.
    Uses iterative projection balancing."""
    best_side = float('inf')
    best_oc = np.array([0.0, 0.0])
    best_oa = 0.0
    
    # Try multiple initial angles
    for oa_init in range(0, 360, 15):
        oc = np.array([0.0, 0.0])
        oa = float(oa_init)
        
        for it in range(500):
            if time.time() > DEADLINE:
                break
            
            side, _ = _min_enclosing_side(centers, angles, oc, oa)
            if side < best_side:
                best_side = side
                best_oc = oc.copy()
                best_oa = oa
            
            # Try to improve by adjusting center and angle
            # Gradient-free: try small perturbations
            improved = False
            for d_ox in [-0.05, 0.05, 0]:
                for d_oy in [-0.05, 0.05, 0]:
                    for d_oa in [-0.05, 0.05, 0]:
                        new_oc = oc + np.array([d_ox, d_oy])
                        new_oa = oa + d_oa
                        s2, _ = _min_enclosing_side(centers, angles, new_oc, new_oa)
                        if s2 < side - 1e-10:
                            oc = new_oc
                            oa = new_oa
                            side = s2
                            improved = True
                            break
                    if improved:
                        break
                if improved:
                    break
            if not improved:
                break
    
    return best_side, best_oc, best_oa

def _try_arrangement(centers, angles):
    """Try an arrangement: optimize outer hexagon, check validity."""
    side, oc, oa = _optimize_outer(centers, angles)
    if time.time() > DEADLINE:
        return float('inf'), None
    
    # Fine-tune: verify validity
    if not _check_validity(centers, angles, tuple(oc), side, oa):
        # Try slightly larger side
        side2 = side + 0.01
        if _check_validity(centers, angles, tuple(oc), side2, oa):
            return side2, (centers, angles, oc, side2, oa)
        return float('inf'), None
    
    return side, (centers, angles, oc, side, oa)

def construct():
    start_time = time.time()
    
    best_side = float('inf')
    best_result = None
    
    # Generate hex-lattice positions
    lattice = _hex_lattice_positions()
    
    # We need to choose N=11 positions from the lattice.
    # Strategy: enumerate subsets that include the origin and nearby points.
    # The lattice has positions indexed by (i,j) where pos = i*a1 + j*a2.
    # Let's enumerate all subsets of size 11 from a reasonable subset of lattice points.
    
    # First, let's try the "compact" arrangements: center + ring + partial ring
    # Hexagonal number: 1 + 6 + 12 + ...
    # For 11 hexagons: 1 center + 6 first ring + 4 from second ring
    
    a1 = np.array([np.sqrt(3), 0.0])
    a2 = np.array([np.sqrt(3)/2, 1.5])
    
    # Generate lattice points with their (i,j) indices
    lattice_dict = {}
    for i in range(-3, 4):
        for j in range(-3, 4):
            pos = tuple(i * a1 + j * a2)
            lattice_dict[(i, j)] = pos
    
    # First ring: 6 points at distance 1 in lattice coords
    first_ring = [(1,0), (-1,0), (0,1), (0,-1), (1,-1), (-1,1)]
    # Second ring: 12 points at distance 2
    second_ring = [(2,0), (-2,0), (0,2), (0,-2), (2,-2), (-2,2),
                   (1,1), (-1,-1), (2,-1), (-2,1), (1,-2), (-1,2)]
    
    # Strategy 1: Center + all 6 first ring + 4 from second ring
    # Enumerate all C(12,4) = 495 choices of 4 from second ring
    center_pos = (0, 0)
    all_positions = [center_pos] + first_ring
    base_indices = [center_pos] + first_ring
    
    # Enumerate subsets of second ring of size 4
    for subset in combinations(range(12), 4):
        if time.time() > DEADLINE:
            break
        
        chosen_second = [second_ring[k] for k in subset]
        indices = base_indices + chosen_second
        
        centers = [lattice_dict[idx] for idx in indices]
        angles = [0.0] * N
        
        side, result = _try_arrangement(centers, angles)
        if side < best_side:
            best_side = side
            best_result = result
            # Don't break; keep searching for better
    
    # Strategy 2: Try other compact arrangements
    # e.g., 3-4-3-1 pattern or other shapes
    # Let's also try arrangements without the center
    
    # Generate more lattice-based arrangements
    # Try all subsets of 11 from a 5x5 lattice region
    lattice_5x5 = [(i, j) for i in range(-2, 3) for j in range(-2, 3)]
    
    # To keep it manageable, try subsets that are "connected" or compact
    # For now, try a few more specific patterns
    
    # Pattern: 2-3-2-2-2 (staggered rows)
    patterns = [
        # Row-based patterns
        [(0,0), (1,0), (2,0), (0,1), (1,1), (2,1), (0,2), (1,2), (2,2), (0,3), (1,3)],
        [(0,0), (1,0), (2,0), (3,0), (1,1), (2,1), (3,1), (0,1), (1,2), (2,2), (3,2)],
        [(0,0), (1,0), (2,0), (3,0), (4,0), (1,1), (2,1), (3,1), (0,1), (1,2), (2,2)],
        # Diamond
        [(0,0), (1,0), (-1,0), (0,1), (0,-1), (1,1), (-1,-1), (2,0), (-2,0), (1,-1), (-1,1)],
        # Cross
        [(0,0), (1,0), (-1,0), (0,1), (0,-1), (1,1), (1,-1), (-1,1), (-1,-1), (2,0), (-2,0)],
    ]
    
    for pattern in patterns:
        if time.time() > DEADLINE:
            break
        if len(pattern) != N:
            continue
        # Check all indices are valid
        valid = all(idx in lattice_dict for idx in pattern)
        if not valid:
            continue
        centers = [lattice_dict[idx] for idx in pattern]
        angles = [0.0] * N
        side, result = _try_arrangement(centers, angles)
        if side < best_side:
            best_side = side
            best_result = result
    
    # Strategy 3: Try rotating individual hexagons
    # For the best arrangement found, try different angles
    if best_result is not None and time.time() < DEADLINE - 20:
        centers, angles, oc, side, oa = best_result
        for i in range(N):
            for trial_angle in [0, 30, 60, 90, 120, 150]:
                if time.time() > DEADLINE:
                    break
                new_angles = angles.copy()
                new_angles[i] = float(trial_angle)
                side2, result2 = _try_arrangement(centers, new_angles)
                if side2 < best_side:
                    best_side = side2
                    best_result = result2
    
    # Fallback: linear arrangement
    if best_result is None:
        centers = [(2 * np.sqrt(3) * k, 0.0) for k in range(N)]
        angles = [0.0] * N
        best_side, best_result = _try_arrangement(centers, angles)
    
    if best_result is None:
        # Ultimate fallback
        inner = [[4.0 * k, 0.0, 0.0] for k in range(-(N // 2), N - N // 2)]
        return inner, (0.0, 0.0), 40.0, 0.0
    
    centers, angles, oc, side, oa = best_result
    inner = [[float(centers[i][0]), float(centers[i][1]), float(angles[i])] for i in range(N)]
    return inner, (float(oc[0]), float(oc[1])), float(side), float(oa)
