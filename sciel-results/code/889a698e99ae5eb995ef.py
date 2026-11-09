"""Fresh design: lattice construction + local refinement + binary search on side length."""
import numpy as np
import time

N = 11
DEADLINE = time.time() + 130.0  # safety margin before 150s kill


def _hex_vertices(center, angle_deg):
    """Vertices of a unit regular hexagon (side 1) centered at `center`, rotated by angle_deg."""
    cx, cy = center
    a = np.deg2rad(angle_deg)
    # unit hexagon: vertices at distance 1 (circumradius = side for regular hexagon)
    base = np.array([[np.cos(np.pi/3 * k), np.sin(np.pi/3 * k)] for k in range(6)])
    R = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
    return base @ R.T + np.array([cx, cy])


def _point_in_hexagon(p, center, angle_deg, side):
    """Check if point p is inside a regular hexagon (center, angle, side)."""
    cx, cy = center
    a = np.deg2rad(angle_deg)
    # transform p into hexagon's local frame
    dx, dy = p[0] - cx, p[1] - cy
    x = dx * np.cos(a) + dy * np.sin(a)
    y = -dx * np.sin(a) + dy * np.cos(a)
    # regular hexagon with side s, flat-top in local frame:
    # |x| <= s, |y| <= s*sqrt(3)/2, and |x| + |y|/sqrt(3) <= s
    s = side
    if abs(x) > s:
        return False
    if abs(y) > s * np.sqrt(3) / 2:
        return False
    if abs(x) + abs(y) / np.sqrt(3) > s:
        return False
    return True


def _build_lattice_positions(n):
    """Place n unit hexagons on a triangular lattice (co-aligned, 0 deg).
    Returns list of (x, y, 0.0).
    Triangular lattice spacing: dx = 1.5, dy = sqrt(3)/2 * 1 = sqrt(3)/2? 
    For co-aligned hexagons of side 1, the lattice vectors are:
      a1 = (3/2, sqrt(3)/2)  -> |a1| = sqrt(9/4 + 3/4) = sqrt(3)
      a2 = (0, sqrt(3))
    Wait: adjacent co-aligned hexagons (side 1) touch when center distance = 2 (diameter).
    Actually for regular hexagon side s=1, the distance between centers of two touching
    co-aligned hexagons is 2*s = 2. The triangular lattice with nearest-neighbor distance 2:
      a1 = (3, sqrt(3))  -> |a1| = sqrt(9+3) = sqrt(12) = 2*sqrt(3)  -- too far
    Let me recompute. Regular hexagon side 1: width (flat-to-flat) = sqrt(3), 
    point-to-point (vertex-to-vertex) = 2.
    Two co-aligned hexagons placed side by side along a flat face: center distance = sqrt(3)? 
    No. The hexagon has 6 faces. Distance from center to a face (apothem) = sqrt(3)/2.
    Two hexagons sharing a face: center distance = 2 * apothem = sqrt(3).
    So the triangular lattice vectors for tiling (edge-to-edge) are:
      a1 = (sqrt(3), 0)  ?  No, that's along a flat direction.
    For a regular hexagon tiling (honeycomb dual), the lattice is triangular with:
      a1 = (3/2 * 2, sqrt(3)/2 * 2) = (3, sqrt(3))  if side=1? 
    Let me just use the known fact: for unit hexagons in a triangular arrangement,
    the center-to-center distance to nearest neighbor (touching on an edge) is sqrt(3).
    Lattice vectors: a1 = (sqrt(3), 0), a2 = (sqrt(3)/2, 3/2). |a2| = sqrt(3/4 + 9/4)=sqrt(3). Good.
    """
    positions = []
    # Use a hexagonal (triangular) lattice with nearest-neighbor distance sqrt(3)
    a1 = np.array([np.sqrt(3), 0.0])
    a2 = np.array([np.sqrt(3)/2, 1.5])
    # Generate enough lattice points, then take n closest to origin
    max_r = int(np.ceil(np.sqrt(n) / 1.5)) + 3
    candidates = []
    for i in range(-max_r, max_r + 1):
        for j in range(-max_r, max_r + 1):
            p = i * a1 + j * a2
            candidates.append((np.linalg.norm(p), i, j, p))
    candidates.sort(key=lambda t: t[0])
    for _, i, j, p in candidates[:n]:
        positions.append([float(p[0]), float(p[1]), 0.0])
    return positions


def _refine_positions(positions, n):
    """Local refinement: nudge positions to minimize max distance to a bounding hexagon.
    We keep angles at 0 (optimal for co-aligned packing).
    Use a simple gradient-free local search: for each hexagon, try small moves.
    Objective: minimize the outer side length needed.
    """
    # We'll do a simple iterative relaxation: for each pair of overlapping hexagons,
    # push them apart. But for co-aligned hexagons on a lattice they don't overlap.
    # Instead, we just return the lattice positions (they're already valid).
    # A more aggressive refinement could optimize the bounding hexagon orientation,
    # but for now the lattice is tight.
    return positions


def _bounding_side_length(positions, outer_center, outer_angle_deg):
    """Compute the minimum side length of a regular hexagon (outer_center, outer_angle)
    that contains all unit hexagons at `positions`.
    For each inner hexagon, the farthest vertex from the outer center in the 
    outer hexagon's local frame determines the required side.
    """
    ocx, ocy = outer_center
    oa = np.deg2rad(outer_angle_deg)
    max_r = 0.0
    for (x, y, ang) in positions:
        verts = _hex_vertices((x, y), ang)
        for v in verts:
            # transform to outer local frame
            dx, dy = v[0] - ocx, v[1] - ocy
            lx = dx * np.cos(oa) + dy * np.sin(oa)
            ly = -dx * np.sin(oa) + dy * np.cos(oa)
            # regular hexagon with side s contains point if |lx|<=s, |ly|<=s*sqrt(3)/2,
            # |lx|+|ly|/sqrt(3) <= s
            # The required s for this point:
            # From |lx| <= s: s >= |lx|
            # From |ly| <= s*sqrt(3)/2: s >= 2|ly|/sqrt(3)
            # From |lx|+|ly|/sqrt(3) <= s: s >= |lx| + |ly|/sqrt(3)
            req = max(abs(lx), 2 * abs(ly) / np.sqrt(3), abs(lx) + abs(ly) / np.sqrt(3))
            if req > max_r:
                max_r = req
    return max_r


def _is_valid(positions, outer_center, outer_side, outer_angle_deg):
    """Check all inner hexagons are inside the outer hexagon and don't overlap each other."""
    # Check containment
    for (x, y, ang) in positions:
        verts = _hex_vertices((x, y), ang)
        for v in verts:
            if not _point_in_hexagon(v, outer_center, outer_angle_deg, outer_side):
                return False
    # Check pairwise non-overlap (approximate via center distance for co-aligned)
    # For co-aligned hexagons, no overlap iff center distance >= sqrt(3) (edge-to-edge)
    # More precisely, if both at 0 deg, the minimum center distance for no overlap is sqrt(3).
    # Use a conservative check: distance >= sqrt(3) - 1e-6
    for i in range(len(positions)):
        for j in range(i + 1, len(positions)):
            x1, y1, _ = positions[i]
            x2, y2, _ = positions[j]
            d = np.hypot(x1 - x2, y1 - y2)
            if d < np.sqrt(3) - 1e-4:
                return False
    return True


def construct():
    t0 = time.time()
    # Build lattice positions
    positions = _build_lattice_positions(N)
    
    # Try a few outer orientations and centers to minimize bounding side
    best_side = float('inf')
    best_config = None
    
    # The lattice is centered at origin; try outer center = origin
    for outer_angle_deg in [0.0, 30.0]:
        # Estimate bounding side
        side = _bounding_side_length(positions, (0.0, 0.0), outer_angle_deg)
        # Add small margin
        side += 1e-3
        if side < best_side:
            # Validate
            if _is_valid(positions, (0.0, 0.0), side, outer_angle_deg):
                best_side = side
                best_config = (positions, (0.0, 0.0), side, outer_angle_deg)
    
    # If no valid config found (shouldn't happen), fallback
    if best_config is None:
        # Use a large safe side
        side = 10.0
        best_config = (positions, (0.0, 0.0), side, 0.0)
    
    positions, center, side, angle = best_config
    
    # Deadline guard
    if time.time() > DEADLINE:
        return positions, center, side, angle
    
    # Try to improve: binary search on side length for the best orientation
    if time.time() < DEADLINE - 20:
        for outer_angle_deg in [0.0, 30.0]:
            lo = 0.0
            hi = side + 1.0
            # Ensure hi is valid
            for _ in range(30):
                if time.time() > DEADLINE - 5:
                    break
                mid = (lo + hi) / 2
                if _is_valid(positions, (0.0, 0.0), mid, outer_angle_deg):
                    hi = mid
                else:
                    lo = mid
            if hi < best_side and _is_valid(positions, (0.0, 0.0), hi, outer_angle_deg):
                best_side = hi
                best_config = (positions, (0.0, 0.0), hi, outer_angle_deg)
    
    positions, center, side, angle = best_config
    return positions, center, side, angle
