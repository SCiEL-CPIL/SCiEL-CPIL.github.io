"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import math
import numpy as np
from scipy.optimize import minimize

N = 11

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s limit

    # Unit hexagon vertices (flat-top orientation, side length 1)
    # For a regular hexagon with side length s, the vertices are at angles 0, 60, 120, 180, 240, 300 degrees
    # For "pointy-top" or "flat-top" depends on convention. Let's use standard:
    # A regular hexagon centered at origin with side length 1:
    # Vertices at distance 1 from center (circumradius = side length for regular hexagon)
    # Angles: 0, 60, 120, 180, 240, 300 degrees
    
    def hex_vertices(cx, cy, side_len, angle_deg):
        """Return vertices of a regular hexagon centered at (cx, cy) with given side length and rotation."""
        angle_rad = math.radians(angle_deg)
        # Circumradius equals side length for regular hexagon
        R = side_len
        verts = []
        for k in range(6):
            theta = angle_rad + k * math.pi / 3.0
            x = cx + R * math.cos(theta)
            y = cy + R * math.sin(theta)
            verts.append((x, y))
        return np.array(verts)

    def min_distance_between_hexagons(c1, s1, a1, c2, s2, a2):
        """
        Compute the minimum distance between two regular hexagons (Minkowski difference approach).
        Returns a smooth-ish lower bound or actual minimum distance between the two convex polygons.
        
        For efficiency, we compute the minimum distance between the two convex hexagons.
        Since both are regular hexagons, we can use the fact that the minimum distance
        between two convex polygons equals the minimum over all pairs of vertices/edges,
        but a more efficient method is to compute the Minkowski difference and find
        the minimum norm point.
        
        Actually, for the optimization, we need a smooth constraint. Let's use the
        distance from center1 to the boundary of hex2 in the direction of (c1-c2), 
        plus similar from the other side. But this is complex.
        
        Alternative: use the fact that for two regular hexagons, the minimum distance
        can be computed by checking all 6*6 = 36 vertex-vertex pairs and 6*6 edge-vertex pairs,
        but that's expensive.
        
        For a simpler approach: the Minkowski sum of two regular hexagons (possibly rotated)
        is another hexagon (if same orientation) or a dodecagon. The minimum distance
        between the two hexagons is related to the minimum distance from the origin to
        the boundary of the Minkowski difference, shifted by the center offset.
        
        Let's use a simpler smooth approximation: 
        d = |c1 - c2| - (R1 + R2) is a lower bound but not tight for all orientations.
        
        Better: For two regular hexagons with circumradii R1, R2 and orientations a1, a2,
        the support function in direction u is h1(u) + h2(-u), and the minimum distance
        is min over directions of (|c1-c2| * cos(theta) - h1(u) - h2(-u)) where theta is angle
        between (c1-c2) and u. This is complex.
        
        For practical purposes in optimization, let's use:
        d_approx = |c1 - c2| - R1 - R2
        This is valid when the hexagons are axis-aligned in the same orientation, but for
        arbitrary rotations, the effective "radius" in the direction of separation matters.
        
        Actually, for regular hexagons, the inradius is s*sqrt(3)/2 and circumradius is s.
        The minimum distance between two identical regular hexagons with the same orientation
        and centers separated by distance d is:
        - If d <= 0: they overlap (distance is negative)
        - The actual minimum gap depends on the angle of separation relative to the hexagon axes.
        
        For simplicity and smoothness, let's use:
        d_eff = |c1 - c2| - 2 * (side_len) * max(1, ...) 
        
        Let me use a different approach: compute the minimum distance between the two convex
        polygons exactly but vectorized as much as possible.
        """
        v1 = hex_vertices(c1[0], c1[1], s1, a1)
        v2 = hex_vertices(c2[0], c2[1], s2, a2)
        
        # Compute minimum distance between two convex polygons
        # For convex polygons, min distance is min of:
        # 1. Vertex-vertex distances
        # 2. Vertex-edge distances (vertex of one to edge of other, and vice versa)
        
        def point_to_segment_dist(p, a, b):
            """Distance from point p to segment ab."""
            ab = b - a
            ap = p - a
            t = np.dot(ap, ab) / (np.dot(ab, ab) + 1e-12)
            t = np.clip(t, 0, 1)
            closest = a + t * ab
            return np.linalg.norm(p - closest)
        
        min_d = np.inf
        
        # Check all vertex-vertex pairs
        for v in v1:
            for w in v2:
                d = np.linalg.norm(v - w)
                if d < min_d:
                    min_d = d
        
        # Check vertex of poly1 to edges of poly2
        for v in v1:
            for i in range(6):
                a = v2[i]
                b = v2[(i + 1) % 6]
                d = point_to_segment_dist(v, a, b)
                if d < min_d:
                    min_d = d
        
        # Check vertex of poly2 to edges of poly1
        for w in v2:
            for i in range(6):
                a = v1[i]
                b = v1[(i + 1) % 6]
                d = point_to_segment_dist(w, a, b)
                if d < min_d:
                    min_d = d
        
        return min_d

    # For N=11, place hexagons in a hexagonal lattice pattern
    # A hexagonal packing of 11 unit hexagons: 
    # The densest packing of regular hexagons is the hexagonal lattice.
    # For N hexagons, arrange them in a hexagonal cluster.
    # 
    # Hexagonal numbers: 1, 7, 19, 37, ...
    # 11 is not a hexagonal number, so we need a non-perfect hexagonal cluster.
    # 
    # Let's use a simple grid arrangement first, then optimize.
    # For a flat-top hexagonal lattice, the spacing between centers is:
    # Horizontal: 2 * side * cos(30°) = sqrt(3) * side
    # Vertical: 1.5 * side
    # 
    # Actually, for regular hexagons of side length s in a hexagonal packing:
    # The center-to-center distance is 2 * s * sin(60°) = s * sqrt(3)
    
    s = 1.0  # side length of inner hexagons
    center_dist = s * math.sqrt(3)  # minimum center-to-center distance for non-overlap
    
    # Generate initial positions in a hexagonal cluster
    positions = []
    # Center hexagon
    positions.append((0.0, 0.0, 0.0))
    
    # First ring: 6 hexagons
    for k in range(6):
        angle = k * 60.0
        x = center_dist * math.cos(math.radians(angle))
        y = center_dist * math.sin(math.radians(angle))
        positions.append((x, y, 0.0))
    
    # Second ring: need 5 more to reach 11
    # Place them in gaps or extend the pattern
    for k in range(6):
        angle = k * 60.0 + 30.0  # Offset by 30 degrees
        r = center_dist * math.sqrt(3)
        x = r * math.cos(math.radians(angle))
        y = r * math.sin(math.radians(angle))
        positions.append((x, y, 0.0))
    
    # We have 1 + 6 + 6 = 13 positions, trim to N
    positions = positions[:N]
    positions = np.array(positions)
    
    def check_no_overlap(pos, eps=0.01):
        """Check if all hexagons are non-overlapping."""
        n = len(pos)
        for i in range(n):
            for j in range(i + 1, n):
                d = min_distance_between_hexagons(pos[i][:2], s, pos[i][2], pos[j][:2], s, pos[j][2])
                if d < eps:
                    return False
        return True

    def enclosing_hex_side_length(pos, outer_center, outer_angle):
        """Compute the minimum side length of an outer regular hexagon that contains all inner hexagons."""
        all_verts = []
        for px, py, ang in pos:
            verts = hex_vertices(px, py, s, ang)
            all_verts.append(verts)
        all_verts = np.vstack(all_verts)
        
        angle_rad = math.radians(outer_angle)
        normals = []
        for k in range(6):
            a = angle_rad + k * math.pi / 3
            normals.append(np.array([math.cos(a), math.sin(a)]))
        normals = np.array(normals)
        
        oc = np.array(outer_center)
        inradius_needed = 0
        for n_vec in normals:
            proj = np.dot(all_verts - oc, n_vec)
            inradius_needed = max(inradius_needed, np.max(proj))
        
        # For a regular hexagon, side length = inradius * 2 / sqrt(3)
        return 2 * inradius_needed / math.sqrt(3)

    # Optimize positions to minimize the enclosing hexagon side length
    # Parameters: all positions (x, y, angle) for each hexagon
    n_hex = N
    dim = n_hex * 3
    
    def objective(params):
        if time.time() > deadline:
            return 1e10
        pos = params.reshape(n_hex, 3)
        
        # Penalty for overlap
        penalty = 0.0
        for i in range(n_hex):
            for j in range(i + 1, n_hex):
                d = min_distance_between_hexagons(pos[i][:2], s, pos[i][2], pos[j][:2], s, pos[j][2])
                if d < 0:
                    penalty += 100.0 * d * d
        
        # Enclosing hexagon
        # Try a few outer angles and take the minimum
        best_side = 1e10
        for outer_angle in [0, 30, 60, 90, 45, 15, 75, 120]:
            # Center the outer hexagon at the centroid of inner hexagons
            centroid = np.mean(pos[:, :2], axis=0)
            side = enclosing_hex_side_length(pos, tuple(centroid), outer_angle)
            if side < best_side:
                best_side = side
        
        return best_side + penalty

    # Initial parameter vector
    x0 = positions.flatten().copy()
    
    # Use multiple starting points and Nelder-Mead
    best_result = None
    best_obj = 1e10
    
    # Try different initial configurations
    for start_angle_offset in [0, 15, 30, 45]:
        if time.time() > deadline:
            break
        
        # Slightly perturb the initial positions
        rng = np.random.RandomState(42 + start_angle_offset)
        x_try = x0.copy()
        x_try += rng.normal(0, 0.5, dim)
        
        try:
            res = minimize(objective, x_try, method='Nelder-Mead',
                          options={'maxiter': 2000, 'xatol': 1e-8, 'fatol': 1e-10,
                                   'adaptive': True})
            if res.fun < best_obj:
                best_obj = res.fun
                best_result = res.x
        except Exception:
            pass
    
    # If optimization failed, use initial configuration
    if best_result is None:
        best_result = x0
    
    pos = best_result.reshape(n_hex, 3)
    
    # Verify no overlap
    if not check_no_overlap(pos, eps=0.005):
        # Fall back to the initial lattice configuration
        pos = positions.copy()
    
    # Find optimal outer hexagon
    centroid = np.mean(pos[:, :2], axis=0)
    best_side = 1e10
    best_center = (0, 0)
    best_angle = 0
    
    for outer_angle in range(0, 180, 15):
        # Try a few center offsets
        for dx in [-0.5, 0, 0.5]:
            for dy in [-0.5, 0, 0.5]:
                oc = (centroid[0] + dx, centroid[1] + dy)
                side = enclosing_hex_side_length(pos, oc, outer_angle)
                if side < best_side:
                    best_side = side
                    best_center = oc
                    best_angle = outer_angle
    
    # Refine the outer hexagon parameters
    def outer_obj(params):
        if time.time() > deadline:
            return 1e10
        cx, cy, angle = params
        return enclosing_hex_side_length(pos, (cx, cy), angle)
    
    try:
        res_outer = minimize(outer_obj, list(best_center) + [best_angle],
                           method='Nelder-Mead',
                           options={'maxiter': 500, 'xatol': 1e-8, 'fatol': 1e-10})
        if res_outer.fun < best_side:
            best_side = res_outer.fun
            best_center = (res_outer.x[0], res_outer.x[1])
            best_angle = res_outer.x[2]
    except Exception:
        pass
    
    # Build return value
    inner_list = [[float(x), float(y), float(a)] for x, y, a in pos]
    
    return inner_list, (float(best_center[0]), float(best_center[1])), float(best_side), float(best_angle)
# EVOLVE-BLOCK-END
