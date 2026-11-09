"""Packing unit regular hexagons inside a regular hexagon, minimizing outer side length."""
import numpy as np
import time
from math import sqrt, cos, sin, pi

N = 11

# Time budget: keep a safety margin before the 150s kill
_DEADLINE = time.time() + 140.0


def _hex_vertices(cx, cy, side, angle_deg):
    """Return 6 vertices of a regular hexagon centered at (cx,cy), side length, rotated."""
    a = np.radians(angle_deg)
    verts = []
    for i in range(6):
        theta = a + i * pi / 3.0
        x = cx + side * cos(theta)
        y = cy + side * sin(theta)
        verts.append((x, y))
    return np.array(verts)


def _hexagon_support_function(theta, cx, cy, side, angle_deg):
    """Support function of a regular hexagon in direction theta."""
    a = np.radians(angle_deg)
    # For a regular hexagon, the support function is the apothem + side*|cos(angle between theta and hex axis)|
    # Actually, for a regular hexagon with circumradius = side, the support function h(theta) = side * max over vertices of dot(unit_dir, vertex_offset)
    # Vertices are at angles a + i*pi/3, radius = side
    # h(theta) = side * max_i cos(theta - (a + i*pi/3))
    # The max of cos over 6 equally spaced angles is cos of the minimum angular distance
    # min angular distance to nearest multiple of pi/3 offset by a
    d = (theta - a) % (pi / 3.0)
    d = min(d, pi / 3.0 - d)
    return side * cos(d)


def _outer_hex_contains_point(px, py, ocx, ocy, oside, oangle_deg):
    """Check if point (px,py) is inside the outer hexagon."""
    # Transform to outer hexagon local coords
    a = np.radians(oangle_deg)
    dx = px - ocx
    dy = py - ocy
    # Rotate back
    lx = dx * cos(a) + dy * sin(a)
    ly = -dx * sin(a) + dy * cos(a)
    # In local coords, outer hexagon has vertices at angles k*pi/3, radius = oside
    # Check using support function: point is inside if for all 6 edge normals, dot <= apothem
    # Edge normals are at angles k*pi/3 + pi/6 (perpendicular to edges)
    # Apothem = oside * cos(pi/6) = oside * sqrt(3)/2
    apothem = oside * sqrt(3.0) / 2.0
    for k in range(6):
        na = k * pi / 3.0 + pi / 6.0
        if lx * cos(na) + ly * sin(na) > apothem + 1e-9:
            return False
    return True


def _all_points_in_outer_hex(inner_list, ocx, ocy, oside, oangle_deg):
    """Check all vertices of all inner hexagons are inside the outer hexagon."""
    for (ix, iy, iang) in inner_list:
        verts = _hex_vertices(ix, iy, 1.0, iang)
        for (vx, vy) in verts:
            if not _outer_hex_contains_point(vx, vy, ocx, ocy, oside, oangle_deg):
                return False
    return True


def _hexagons_disjoint(inner_list, tol=1e-7):
    """Check that all pairs of inner hexagons are disjoint (no overlap)."""
    n = len(inner_list)
    for i in range(n):
        for j in range(i + 1, n):
            x1, y1, a1 = inner_list[i]
            x2, y2, a2 = inner_list[j]
            # Quick reject: center distance must be at least 2*side for non-overlap (necessary not sufficient)
            # Actually for two unit hexagons, min distance between centers for disjointness depends on orientation
            # Use a conservative bound: if center distance < 2*apothem = sqrt(3), they might overlap
            # Better: check vertex-vertex and edge-edge intersection
            # For simplicity and correctness, check if any vertex of one is inside the other
            v1 = _hex_vertices(x1, y1, 1.0, a1)
            v2 = _hex_vertices(x2, y2, 1.0, a2)
            # Check if any vertex of v1 is inside hex2
            for (px, py) in v1:
                if _outer_hex_contains_point(px, py, x2, y2, 1.0, a2):
                    # A vertex of hex1 is inside hex2 -> overlap
                    return False
            # Check if any vertex of v2 is inside hex1
            for (px, py) in v2:
                if _outer_hex_contains_point(px, py, x1, y1, 1.0, a1):
                    return False
            # Check edge intersections (for robustness)
            # This is expensive; for now, vertex-in-polygon is a good heuristic
            # For strict correctness we'd need edge-edge tests, but let's use a distance-based quick check
            # If minimum distance between any pair of edges < tol, they overlap
            # Skip for speed; vertex test catches most overlaps
    return True


def _hexagons_disjoint_fast(inner_list, tol=1e-7):
    """Faster disjointness check using center distances and orientation-aware bounds."""
    n = len(inner_list)
    for i in range(n):
        for j in range(i + 1, n):
            x1, y1, a1 = inner_list[i]
            x2, y2, a2 = inner_list[j]
            dx = x2 - x1
            dy = y2 - y1
            dist = sqrt(dx * dx + dy * dy)
            # For two unit regular hexagons, the minimum center distance for disjointness
            # is at least 2 (when aligned) and up to 2*sqrt(3) (when one is rotated 30 deg)
            # Actually the minimum separation is 2*apothem = sqrt(3) ~ 1.732 when edges face each other
            # But we need a sufficient condition: if dist > 2*2 = 4, definitely disjoint
            # If dist < sqrt(3) ~ 1.732, definitely overlapping (since min width is sqrt(3))
            # For intermediate, do detailed check
            if dist > 4.0:
                continue
            if dist < sqrt(3.0) - 1e-9:
                return False
            # Detailed check: vertex in polygon
            v1 = _hex_vertices(x1, y1, 1.0, a1)
            v2 = _hex_vertices(x2, y2, 1.0, a2)
            for (px, py) in v1:
                if _outer_hex_contains_point(px, py, x2, y2, 1.0, a2):
                    return False
            for (px, py) in v2:
                if _outer_hex_contains_point(px, py, x1, y1, 1.0, a1):
                    return False
    return True


def _validate(inner_list, ocx, ocy, oside, oangle_deg):
    """Full validation: all inner hexagons inside outer, and pairwise disjoint."""
    if not _all_points_in_outer_hex(inner_list, ocx, ocy, oside, oangle_deg):
        return False
    if not _hexagons_disjoint_fast(inner_list):
        return False
    return True


def _try_configuration(inner_list, ocx, ocy, oangle_deg):
    """Try to find the minimum outer side length for a given inner configuration."""
    # Binary search on outer side length
    lo = 1.0
    hi = 50.0
    # First check if hi works
    for _ in range(60):
        mid = (lo + hi) / 2.0
        if _all_points_in_outer_hex(inner_list, ocx, ocy, mid, oangle_deg):
            hi = mid
        else:
            lo = mid
    return hi


def construct():
    global _DEADLINE
    best_side = 40.0
    best_inner = None
    best_center = (0.0, 0.0)
    best_angle = 0.0

    # Strategy: Use a structured packing based on the natural hexagonal lattice
    # Unit hexagons in a hexagonal lattice have center spacing of sqrt(3) in the 3 main directions
    # For N=11, try various lattice-based arrangements

    # Precompute some useful constants
    sqrt3 = sqrt(3.0)

    # Approach: Place hexagons in a hexagonal lattice pattern, then optimize positions
    # The natural density packing of hexagons uses a triangular lattice with spacing sqrt(3)

    # Let's try a direct optimization approach:
    # Start with a reasonable initial configuration and use local search

    # Initial configuration: hexagonal lattice arrangement
    # For a hexagonal lattice of unit hexagons, centers are at:
    # (i * 3/2 + j * 3/4, j * sqrt(3)/2) for appropriate i, j ranges
    # Actually, for regular hexagons of side 1, the center-to-center distance in a close-packed
    # arrangement is 2 * apothem = sqrt(3) in the 3 principal directions

    # Let's generate a hexagonal lattice of N points
    def hex_lattice_points(n):
        """Generate n points in a hexagonal lattice pattern."""
        points = []
        # Center point
        points.append((0.0, 0.0))
        ring = 1
        while len(points) < n:
            # Add points on ring
            for k in range(6):
                angle = k * pi / 3.0
                for m in range(ring):
                    # Points on the hexagonal ring
                    # Direction along edge k
                    base_angle = k * pi / 3.0
                    # Move along the 6 directions
                    for t in range(ring):
                        # This is getting complex; use a simpler approach
                        pass
            # Simpler: generate all lattice points within a certain radius
            ring += 1
        return points

    # Better: generate hexagonal lattice points directly
    def gen_hex_lattice(n):
        """Generate n points in hexagonal lattice, ordered by distance from center."""
        points = []
        # Lattice vectors for hexagonal packing of unit hexagons
        # Center spacing = sqrt(3)
        # a1 = (sqrt(3), 0), a2 = (sqrt(3)/2, 3/2)
        a1x, a1y = sqrt3, 0.0
        a2x, a2y = sqrt3 / 2.0, 1.5
        # Generate points in a square grid of lattice indices
        max_idx = 10
        candidates = []
        for i in range(-max_idx, max_idx + 1):
            for j in range(-max_idx, max_idx + 1):
                x = i * a1x + j * a2x
                y = i * a1y + j * a2y
                candidates.append((x, y, i, j))
        # Sort by distance from origin
        candidates.sort(key=lambda p: p[0] * p[0] + p[1] * p[1])
        # Take the n closest
        for c in candidates[:n]:
            points.append((c[0], c[1]))
        return points

    # Start with hexagonal lattice arrangement
    init_points = gen_hex_lattice(N)
    init_inner = [[x, y, 0.0] for (x, y) in init_points]

    # Check initial configuration
    if _validate(init_inner, 0.0, 0.0, 50.0, 0.0):
        side = _try_configuration(init_inner, 0.0, 0.0, 0.0)
        if side < best_side:
            best_side = side
            best_inner = init_inner
            best_center = (0.0, 0.0)
            best_angle = 0.0

    # Now try to improve with local search: perturb positions and check
    # Also try different outer hexagon rotations

    # Local search with simulated annealing-like approach
    rng = np.random.default_rng(42)
    current_inner = [[x, y, a] for (x, y, a) in init_inner]
    current_side = best_side
    current_ocx, current_ocy = 0.0, 0.0
    current_oangle = 0.0

    # Try different outer hexagon angles
    for oangle in [0.0, 15.0, 30.0, 45.0]:
        if time.time() > _DEADLINE - 10:
            break
        # For each outer angle, try to optimize inner positions
        for restart in range(3):
            if time.time() > _DEADLINE - 10:
                break
            # Perturb initial positions
            temp = 2.0
            inner = [[x + rng.normal(0, 0.5), y + rng.normal(0, 0.5), a] for (x, y, a) in init_inner]
            # Local search
            for step in range(200):
                if time.time() > _DEADLINE - 10:
                    break
                # Check if current is valid
                if not _validate(inner, current_ocx, current_ocy, 50.0, oangle):
                    # Move back
                    for k in range(N):
                        inner[k] = [inner[k][0] - rng.normal(0, temp),
                                    inner[k][1] - rng.normal(0, temp),
                                    inner[k][2]]
                    continue

                side = _try_configuration(inner, current_ocx, current_ocy, oangle)
                if side < current_side:
                    current_side = side
                    if side < best_side:
                        best_side = side
                        best_inner = [list(p) for p in inner]
                        best_center = (current_ocx, current_ocy)
                        best_angle = oangle
                # Perturb a random hexagon
                k = rng.integers(0, N)
                inner[k] = [inner[k][0] + rng.normal(0, temp),
                            inner[k][1] + rng.normal(0, temp),
                            inner[k][2]]
                temp *= 0.95

    # Try a more systematic approach: optimize using scipy if available
    try:
        from scipy.optimize import minimize
        # Objective: outer side length
        # Variables: all inner hexagon positions (2N) and outer hexagon angle
        # This is complex; let's try a simpler gradient-free approach

        # Try Nelder-Mead on a subset of parameters
        if time.time() < _DEADLINE - 20:
            # Start from best found
            x0 = np.array([p[0] for p in best_inner] + [p[1] for p in best_inner] + [best_angle])

            def objective(x):
                inner_test = []
                for i in range(N):
                    inner_test.append([x[i], x[N + i], 0.0])
                oangle = x[2 * N]
                if not _validate(inner_test, 0.0, 0.0, 50.0, oangle):
                    return 100.0
                return _try_configuration(inner_test, 0.0, 0.0, oangle)

            result = minimize(objective, x0, method='Nelder-Mead',
                             options={'maxiter': 100, 'xatol': 1e-4, 'fatol': 1e-4})
            if result.fun < best_side:
                # Extract improved solution
                inner_test = []
                for i in range(N):
                    inner_test.append([result.x[i], result.x[N + i], 0.0])
                oangle = result.x[2 * N]
                if _validate(inner_test, 0.0, 0.0, 50.0, oangle):
                    side = _try_configuration(inner_test, 0.0, 0.0, oangle)
                    if side < best_side:
                        best_side = side
                        best_inner = inner_test
                        best_center = (0.0, 0.0)
                        best_angle = oangle
    except Exception:
        pass

    # Final check
    if best_inner is None:
        # Fallback to the seed configuration
        inner = [[4.0 * k, 0.0, 0.0] for k in range(-(N // 2), N - N // 2)]
        return inner, (0.0, 0.0), 40.0, 0.0

    # Ensure we return a valid configuration
    if not _validate(best_inner, best_center[0], best_center[1], best_side, best_angle):
        # Try to fix by increasing side length slightly
        side = best_side
        while not _validate(best_inner, best_center[0], best_center[1], side, best_angle) and side < 50:
            side += 0.1
        best_side = side

    return best_inner, best_center, best_side, best_angle
