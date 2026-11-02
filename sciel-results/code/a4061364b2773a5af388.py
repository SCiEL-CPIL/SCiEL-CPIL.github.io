"""Program for packing unit hexagons inside a regular hexagon, minimizing outer side length."""
import numpy as np
import time
import math

N = 11
SQRT3 = math.sqrt(3.0)
DEADLINE_FRACTION = 0.90  # Use 90% of the 120s budget


def construct():
    start_time = time.time()
    deadline = start_time + 120.0 * DEADLINE_FRACTION

    # --- Geometry helpers ---

    def hex_vertex_offsets(angle_deg=0.0):
        """Return the 6 vertices of a unit regular hexagon centered at origin,
        rotated by angle_deg degrees. Vertices are at distance 1 from center."""
        theta = math.radians(angle_deg)
        # A regular hexagon has vertices at angles 0, 60, 120, ... from center
        verts = []
        for k in range(6):
            phi = theta + k * math.pi / 3.0
            verts.append((math.cos(phi), math.sin(phi)))
        return verts

    def point_in_hexagon(px, py, cx, cy, s, angle_deg):
        """Check if point (px,py) is inside a regular hexagon with center (cx,cy),
        side length s, rotation angle_deg.
        
        We transform to local coords, then check against the hexagon's half-planes.
        A regular hexagon with side s has apothem (inradius) = s * sqrt(3)/2.
        In local frame (centered, unrotated), the hexagon is the intersection of
        6 half-planes. For a flat-top or pointy-top hexagon, we can check using
        the fact that it's a regular hexagon.
        
        Actually, for a regular hexagon with vertices at distance s (circumradius = s),
        the inradius is s * sqrt(3)/2.
        
        Let's use the vertex-based approach: a point is inside if it's on the same side
        of each edge as the center.
        """
        # Transform to local coordinates
        dx = px - cx
        dy = py - cy
        # Rotate by -angle
        theta = -math.radians(angle_deg)
        cos_t = math.cos(theta)
        sin_t = math.sin(theta)
        lx = dx * cos_t - dy * sin_t
        ly = dx * sin_t + dy * cos_t

        # In local frame, hexagon has vertices at angles 0, 60, 120, 180, 240, 300
        # with circumradius = s
        # The 6 edges are lines. For each edge, check if point is on the correct side.
        # Edge between vertex k and vertex k+1.
        # Normal to edge k points outward.
        # For a regular hexagon with circumradius s, the edge k (from vertex at angle (k-1)*60 to k*60 degrees... 
        # Let me use a simpler formulation:
        # The hexagon is { (x,y) : |x| <= s, |y| <= s*sqrt(3)/2, |x/2 + y*sqrt(3)/2| <= s, ... }
        # Actually for a hexagon with flat sides at top/bottom (pointy left/right):
        # Vertices at (s, 0), (s/2, s*sqrt(3)/2), (-s/2, s*sqrt(3)/2), (-s, 0), (-s/2, -s*sqrt(3)/2), (s/2, -s*sqrt(3)/2)
        # This is a "pointy-top" if we consider x-axis horizontal... 
        # Let me just use the half-plane method with the 6 edge normals.
        
        # For a regular hexagon with circumradius s, centered at origin, with one vertex at angle 0:
        # Vertices: (s*cos(k*pi/3), s*sin(k*pi/3)) for k=0..5
        # Edge between vertex k and vertex k+1 (mod 6)
        # The outward normal to edge k is at angle (k + 0.5) * pi / 3
        # Distance from center to each edge (apothem) = s * cos(pi/6) = s * sqrt(3)/2
        
        apothem = s * SQRT3 / 2.0
        for k in range(6):
            # Normal angle for edge k
            normal_angle = (k + 0.5) * math.pi / 3.0
            nx = math.cos(normal_angle)
            ny = math.sin(normal_angle)
            # Project local point onto normal
            proj = lx * nx + ly * ny
            if proj > apothem:
                return False
        return True

    def max_dist_point_to_hexagon_boundary(px, py, cx, cy, s, angle_deg):
        """Not needed for containment check."""
        pass

    def contains_hexagon(inner_pts, outer_cx, outer_cy, outer_s, outer_angle):
        """Check if all vertices of inner hexagons are inside the outer hexagon."""
        for (ix, iy, iangle) in inner_pts:
            for (vx, vy) in hex_vertex_offsets(iangle):
                px = ix + vx
                py = iy + vy
                if not point_in_hexagon(px, py, outer_cx, outer_cy, outer_s, outer_angle):
                    return False
        return True

    def check_no_overlap(inner_pts):
        """Check if any two unit hexagons overlap.
        Two unit hexagons (possibly rotated) overlap if and only if
        one of them has a vertex inside the other, OR their edges cross.
        For regular hexagons of the same size, if they don't contain each other's
        vertices and don't share edges, they're disjoint.
        
        Actually, a simpler sufficient and necessary condition for two convex polygons
        to be disjoint is that they are separable by a line. For regular hexagons,
        we can check:
        1. No vertex of A is inside B
        2. No vertex of B is inside A
        3. No edge of A intersects any edge of B
        
        But for simplicity and since these are same-size regular hexagons, if the
        minimum distance between any vertex of A and the boundary of B is > 0 AND
        vice versa, they're disjoint.
        
        Actually the cleanest check: two convex polygons are disjoint iff their
        Minkowski difference doesn't contain the origin, or equivalently, we can
        use the separating axis theorem.
        
        For efficiency, let's use: two unit hexagons are disjoint iff no vertex of
        either is inside the other AND no edges intersect.
        
        For simplicity in the initial construction, let's just check vertex-in-hexagon
        in both directions, which catches most overlaps for near-convex configurations.
        """
        n = len(inner_pts)
        for i in range(n):
            for j in range(i + 1, n):
                ix1, iy1, ia1 = inner_pts[i]
                ix2, iy2, ia2 = inner_pts[j]
                # Check if any vertex of hex i is inside hex j
                for (vx, vy) in hex_vertex_offsets(ia1):
                    px = ix1 + vx
                    py = iy1 + vy
                    if point_in_hexagon(px, py, ix2, iy2, 1.0, ia2):
                        return False
                # Check if any vertex of hex j is inside hex i
                for (vx, vy) in hex_vertex_offsets(ia2):
                    px = ix2 + vx
                    py = iy2 + vy
                    if point_in_hexagon(px, py, ix1, iy1, 1.0, ia1):
                        return False
        return True

    def compute_min_outer_side(inner_pts, outer_cx=0.0, outer_cy=0.0, outer_angle=0.0):
        """Compute the minimum side length s such that all inner hexagon vertices
        fit inside a regular hexagon centered at (outer_cx, outer_cy) with rotation
        outer_angle.
        
        For each inner vertex (px, py), we need the smallest s such that
        point_in_hexagon(px, py, outer_cx, outer_cy, s, outer_angle) is True.
        
        In local coordinates (centered at outer center, rotated by -outer_angle),
        a point (lx, ly) is inside a hexagon of side s iff:
        For each of the 6 edge normals at angle (k+0.5)*pi/3, the projection 
        lx*cos(angle) + ly*sin(angle) <= s * sqrt(3)/2.
        
        So s >= 2/sqrt(3) * max over k of (lx*cos(angle_k) + ly*sin(angle_k))
        where we take the positive part, but actually all 6 constraints must hold,
        so s >= 2/sqrt(3) * max_k |...|... no wait.
        
        The constraint is proj_k <= apothem = s*sqrt(3)/2 for all k.
        So s >= 2/sqrt(3) * max_k(proj_k) where proj_k can be negative (no issue).
        But we need all projections to be <= s*sqrt(3)/2.
        So s >= 2/sqrt(3) * max_k(proj_k).
        
        Wait, but proj_k is signed. If proj_k is negative, it's automatically satisfied.
        So s >= 2/sqrt(3) * max(0, max_k(proj_k)).
        
        Actually no - we need s*sqrt(3)/2 >= proj_k for ALL k.
        So s >= 2/sqrt(3) * max_k(proj_k).
        If all proj_k are negative, then s >= negative number, so s >= 0 suffices.
        But s must be positive, so s = max(2/sqrt(3) * max_k(proj_k), small_positive).
        """
        theta = -math.radians(outer_angle)
        cos_t = math.cos(theta)
        sin_t = math.sin(theta)
        
        max_proj = 0.0
        for (ix, iy, iangle) in inner_pts:
            for (vx, vy) in hex_vertex_offsets(iangle):
                px = ix + vx
                py = iy + vy
                # Transform to outer local coords
                dx = px - outer_cx
                dy = py - outer_cy
                lx = dx * cos_t - dy * sin_t
                ly = dx * sin_t + dy * cos_t
                for k in range(6):
                    normal_angle = (k + 0.5) * math.pi / 3.0
                    nx = math.cos(normal_angle)
                    ny = math.sin(normal_angle)
                    proj = lx * nx + ly * ny
                    if proj > max_proj:
                        max_proj = proj
        
        s = 2.0 / SQRT3 * max_proj
        return max(s, 0.001)

    # --- Initial construction: honeycomb packing ---
    
    def generate_honeycomb(n, offset_x=0.0, offset_y=0.0, angle=0.0):
        """Generate n unit hexagons in a honeycomb pattern.
        For flat-top hexagons with circumradius 1 (side 1):
        - Horizontal spacing between centers in same row: 3 (distance between parallel edges * 2 = 3? No.)
        
        Actually, for regular hexagons of side 1:
        - Circumradius = 1
        - Inradius = sqrt(3)/2
        - Distance between centers of adjacent hexagons in a honeycomb = 2 * inradius = sqrt(3)
        
        For pointy-top hexagons (vertex at top):
        - Horizontal neighbors: distance = sqrt(3) apart, offset vertically by 1.5
        - Actually let me think more carefully.
        
        For a regular hexagon with side 1, if we orient it with a flat top:
        Vertices at angles 30, 90, 150, 210, 270, 330 degrees (or equivalently, 
        flat sides at top and bottom).
        
        Honeycomb tiling with side-1 hexagons:
        - Centers form a triangular lattice
        - Nearest neighbor distance = sqrt(3)
        - Three directions at 60 degrees apart
        
        Let me use the standard honeycomb with horizontal rows:
        Row 0: centers at (0,0), (sqrt(3), 0), (2*sqrt(3), 0), ...
        Row 1: centers at (sqrt(3)/2, 3/2), (3*sqrt(3)/2, 3/2), ...
        Row 2: centers at (0, 3), (sqrt(3), 3), ...
        
        Wait, that's for pointy-top hexagons. Let me verify:
        Pointy-top hex with side 1: vertices at (0,1), (sqrt(3)/2, 1/2), (sqrt(3)/2, -1/2), (0,-1), (-sqrt(3)/2, -1/2), (-sqrt(3)/2, 1/2)
        Two adjacent hexagons sharing a vertical edge: centers at distance sqrt(3) apart horizontally.
        Two hexagons sharing a diagonal edge: centers offset by (sqrt(3)/2, 3/2) or similar.
        
        For a compact arrangement, let's use:
        Row spacing (vertical): 1.5 (for pointy-top, the vertical distance between row centers)
        Column spacing within a row: sqrt(3)
        Odd rows shifted by sqrt(3)/2 horizontally.
        """
        pts = []
        count = 0
        row = 0
        while count < n:
            # Number of hexagons in this row
            row_width = n  # We'll trim later
            for col in range(n):
                if count >= n:
                    break
                x = col * SQRT3 + (row % 2) * (SQRT3 / 2.0)
                y = row * 1.5
                # Center the whole thing
                pts.append((x, y, angle))
                count += 1
            row += 1
        
        # Center the configuration
        if pts:
            xs = [p[0] for p in pts]
            ys = [p[1] for p in pts]
            cx = (min(xs) + max(xs)) / 2.0
            cy = (min(ys) + max(ys)) / 2.0
            pts = [(p[0] - cx + offset_x, p[1] - cy + offset_y, p[2]) for p in pts]
        
        return pts

    # Generate initial honeycomb packing
    inner = generate_honeycomb(N)
    
    # Compute initial outer side
    initial_s = compute_min_outer_side(inner, 0.0, 0.0, 0.0)
    
    # --- Local optimization: adjust outer center and try to reduce side ---
    # The outer hexagon center and rotation can be adjusted.
    # For a given inner configuration, the optimal outer hexagon is determined by
    # the "envelope" of all inner vertices. We want to minimize the side length.
    
    # Let's try a few different outer rotations and find the best side length
    best_s = float('inf')
    best_outer = (0.0, 0.0, 0.0)
    
    for trial in range(12):
        outer_angle = trial * 30.0  # Try every 30 degrees
        s = compute_min_outer_side(inner, 0.0, 0.0, outer_angle)
        if s < best_s:
            best_s = s
            best_outer = (0.0, 0.0, outer_angle)
    
    # Also try shifting the outer center slightly
    # The optimal center is roughly the center of the bounding region
    # Let's do a coarse grid search for outer center
    best_s2 = best_s
    best_outer2 = best_outer
    outer_angle_fixed = best_outer[2]
    
    # Estimate center from inner points
    cx_est = sum(p[0] for p in inner) / N
    cy_est = sum(p[1] for p in inner) / N
    
    for dcx in [-0.5, -0.25, 0.0, 0.25, 0.5]:
        for dcy in [-0.5, -0.25, 0.0, 0.25, 0.5]:
            if time.time() > deadline:
                break
            ox = cx_est + dcx
            oy = cy_est + dcy
            s = compute_min_outer_side(inner, ox, oy, outer_angle_fixed)
            if s < best_s2:
                best_s2 = s
                best_outer2 = (ox, oy, outer_angle_fixed)
    
    best_s = best_s2
    best_outer = best_outer2
    
    # --- Try a compacted arrangement: reduce vertical spacing ---
    # In a honeycomb, the vertical spacing is 1.5 for unit hexagons.
    # If we allow slight overlap-free compression by rotating hexagons, we might do better.
    # But for now, let's try a simpler approach: use a more compact initial layout.
    
    # Alternative: place hexagons in a hexagonal cluster pattern
    def generate_hex_cluster(n):
        """Generate n hexagons in a hexagonal cluster (centered hexagonal numbers)."""
        pts = [(0.0, 0.0, 0.0)]  # Center
        if n == 1:
            return pts
        
        # Rings of hexagons around the center
        ring = 1
        while len(pts) < n:
            # Ring r has 6*r hexagons
            ring_count = 6 * ring
            for i in range(ring_count):
                if len(pts) >= n:
                    break
                angle = i * (2 * math.pi / ring_count)
                # Position on ring
                # For a hexagonal lattice, ring r hexagons are at distance r*sqrt(3)
                # But the angular positions need to align with the lattice
                # Simplified: place at angle with radius ring * sqrt(3)
                r_dist = ring * SQRT3
                x = r_dist * math.cos(angle)
                y = r_dist * math.sin(angle)
                pts.append((x, y, 0.0))
            ring += 1
        
        return pts[:n]

    # Compare honeycomb vs hex cluster
    inner_cluster = generate_hex_cluster(N)
    best_s_cluster = float('inf')
    best_outer_cluster = (0.0, 0.0, 0.0)
    
    for trial in range(12):
        outer_angle = trial * 30.0
        s = compute_min_outer_side(inner_cluster, 0.0, 0.0, outer_angle)
        if s < best_s_cluster:
            best_s_cluster = s
            best_outer_cluster = (0.0, 0.0, outer_angle)
    
    # Choose the better configuration
    if best_s_cluster < best_s:
        inner = inner_cluster
        best_s = best_s_cluster
        best_outer = best_outer_cluster
    
    # --- Final answer ---
    # Convert inner to the required format: list of [x, y, angle_degrees]
    inner_final = [[p[0], p[1], p[2]] for p in inner]
    outer_cx, outer_cy, outer_angle = best_outer
    
    return inner_final, (outer_cx, outer_cy), best_s, outer_angle


# Test the construction
if __name__ == "__main__":
    result = construct()
    inner, center, side, angle = result
    print(f"Outer center: {center}")
    print(f"Outer side length: {side}")
    print(f"Outer rotation: {angle}")
    print(f"Number of inner hexagons: {len(inner)}")
    for i, h in enumerate(inner):
        print(f"  Hex {i}: x={h[0]:.4f}, y={h[1]:.4f}, angle={h[2]:.1f}")
