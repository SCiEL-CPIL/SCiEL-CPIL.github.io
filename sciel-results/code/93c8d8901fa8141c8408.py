import math
import time
import random
import numpy as np
import itertools

N = 11

# EVOLVE-BLOCK-START
def construct():
    deadline = time.time() + 145.0
    
    # Precompute base vertices for a unit hexagon (side 1)
    # Vertices at angles 0, 60, 120, 180, 240, 300
    base_verts = [(math.cos(math.radians(60 * i)), math.sin(math.radians(60 * i))) for i in range(6)]
    
    def transform_verts(cx, cy, S, angle_deg):
        a = math.radians(angle_deg)
        ca, sa = math.cos(a), math.sin(a)
        verts = []
        for vx, vy in base_verts:
            rx = vx * ca - vy * sa
            ry = vx * sa + vy * ca
            verts.append((cx + S * rx, cy + S * ry))
        return verts

    def point_in_polygon(p, polygon):
        x, y = p
        n = len(polygon)
        inside = False
        j = n - 1
        for i in range(n):
            xi, yi = polygon[i]
            xj, yj = polygon[j]
            if ((yi > y) != (yj > y)) and (x < (xj - xi) * (y - yi) / (yj - yi) + xi):
                inside = not inside
            j = i
        return inside

    def segments_intersect(p1, p2, p3, p4):
        def cross(o, a, b):
            return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])
        d1 = cross(p3, p4, p1)
        d2 = cross(p3, p4, p2)
        d3 = cross(p1, p2, p3)
        d4 = cross(p1, p2, p4)
        if ((d1 > 0 and d2 < 0) or (d1 < 0 and d2 > 0)) and \
           ((d3 > 0 and d4 < 0) or (d3 < 0 and d4 > 0)):
            return True
        return False

    def hexagons_disjoint(verts_a, verts_b):
        for v in verts_a:
            if point_in_polygon(v, verts_b):
                return False
        for v in verts_b:
            if point_in_polygon(v, verts_a):
                return False
        for i in range(6):
            a1, a2 = verts_a[i], verts_a[(i + 1) % 6]
            b1, b2 = verts_b[i], verts_b[(i + 1) % 6]
            if segments_intersect(a1, a2, b1, b2):
                return False
        return True

    def compute_outer_hex(inner, ocx, ocy, oangle):
        """
        Computes the minimum side length S of an outer hexagon centered at (ocx, ocy) 
        with rotation oangle that contains all inner hexagons.
        """
        inner_verts = []
        for (ix, iy, iangle) in inner:
            inner_verts.append(transform_verts(ix, iy, 1.0, iangle))
        
        a = math.radians(oangle)
        apothem_factor = math.sqrt(3) / 2
        max_S = 0.0
        for i in range(6):
            # Normals to the sides of the outer hexagon
            theta = a + math.radians(30 + 60 * i)
            nx, ny = math.cos(theta), math.sin(theta)
            max_proj = 0.0
            for iv in inner_verts:
                for v in iv:
                    proj = (v[0] - ocx) * nx + (v[1] - ocy) * ny
                    if proj > max_proj:
                        max_proj = proj
            if max_proj > 0:
                S_needed = max_proj / apothem_factor
                if S_needed > max_S:
                    max_S = S_needed
        return max_S

    def all_hexagons_fit(inner, ocx, ocy, S, oangle):
        outer_verts = transform_verts(ocx, ocy, S, oangle)
        inner_verts = []
        for (ix, iy, iangle) in inner:
            iv = transform_verts(ix, iy, 1.0, iangle)
            inner_verts.append(iv)
        
        # Check containment
        for iv in inner_verts:
            for v in iv:
                if not point_in_polygon(v, outer_verts):
                    return False
        
        # Check disjointness
        for i in range(len(inner_verts)):
            for j in range(i + 1, len(inner_verts)):
                if not hexagons_disjoint(inner_verts[i], inner_verts[j]):
                    return False
        return True

    def generate_hex_clusters(n_points, max_radius=4):
        """
        Generates all sets of n_points lattice points that form a "hexagonal" cluster.
        We use axial coordinates (q, r).
        Distance from origin (0,0) in axial coords:
        dist(q, r) = (abs(q) + abs(r) + abs(q+r)) / 2
        """
        points = []
        for q in range(-max_radius, max_radius + 1):
            for r in range(-max_radius, max_radius + 1):
                dist = (abs(q) + abs(r) + abs(q + r)) / 2
                if dist <= max_radius:
                    points.append((q, r))
        
        # Sort by distance, then by angle for determinism
        points.sort(key=lambda p: (p[0]**2 + p[1]**2 + p[0]*p[1], p[0], p[1]))
        
        # We want the N closest points.
        # However, simply taking the first N might not be the only "blob".
        # For small N (11), the "hexagonal number" shapes are very constrained.
        # 1, 7, 19 are centered hexagonal numbers.
        # For 11, we take the 11 closest.
        # Let's verify if there are other compact shapes.
        # The 11 closest points on a triangular lattice are unique up to rotation/reflection.
        # They form a hexagon of "radius" 2 with 2 corners missing? 
        # Radius 1: 1 point.
        # Radius 2: 7 points (center + 6 neighbors).
        # Radius 3: 19 points.
        # 11 points: 7 + 4 from the next ring.
        # The 4 points from the next ring (radius 3) can be placed in different ways?
        # No, to minimize the bounding box (and thus the outer hex size), we want to fill the corners.
        # The 11 closest points are actually a specific shape.
        
        # Let's just take the first N points from the sorted list.
        selected = points[:n_points]
        
        # Convert axial to cartesian
        # x = size * (q + r/2)
        # y = size * (r * sqrt(3)/2)
        # Here size is the distance between centers. For unit hexagons, min dist is sqrt(3).
        # Let's use spacing = sqrt(3) for touching hexagons.
        spacing = math.sqrt(3)
        
        cart_points = []
        for q, r in selected:
            x = spacing * (q + r / 2.0)
            y = spacing * (r * math.sqrt(3) / 2.0)
            cart_points.append((x, y))
        
        return cart_points

    def optimize_outer_hex(inner, seed=0):
        """
        Given a fixed configuration of inner hexagons, find the best outer hexagon
        (center and rotation) that minimizes the side length.
        """
        # Calculate bounding box of inner hexagons to estimate center
        all_x = []
        all_y = []
        for ix, iy, ia in inner:
            verts = transform_verts(ix, iy, 1.0, ia)
            for v in verts:
                all_x.append(v[0])
                all_y.append(v[1])
        
        min_x, max_x = min(all_x), max(all_x)
        min_y, max_y = min(all_y), max(all_y)
        cx0 = (min_x + max_x) / 2.0
        cy0 = (min_y + max_y) / 2.0
        
        best_S = float('inf')
        best_cx = cx0
        best_cy = cy0
        best_ang = 0.0
        
        # Coarse search
        # Center search range: around centroid, +/- 1.0
        # Angle search range: 0 to 60, step 5
        
        # First, try angles 0, 30, 60 with optimized center
        for ang in [0, 30, 60]:
            # Optimize center for this angle
            # The optimal center minimizes the max projection.
            # This is a convex optimization problem (minimize max of linear functions).
            # We can use a simple grid search or scipy.
            # Grid search:
            # Range: cx0 +/- 1.5, cy0 +/- 1.5
            # Step: 0.2
            found = False
            for cx in np.arange(cx0 - 1.5, cx0 + 1.5, 0.2):
                for cy in np.arange(cy0 - 1.5, cy0 + 1.5, 0.2):
                    S = compute_outer_hex(inner, cx, cy, ang)
                    if S < best_S:
                        best_S = S
                        best_cx = cx
                        best_cy = cy
                        best_ang = ang
                        found = True
            
            # Refine center with smaller step around best found so far for this angle
            if found:
                for cx in np.arange(best_cx - 0.2, best_cx + 0.2, 0.05):
                    for cy in np.arange(best_cy - 0.2, best_cy + 0.2, 0.05):
                        S = compute_outer_hex(inner, cx, cy, ang)
                        if S < best_S:
                            best_S = S
                            best_cx = cx
                            best_cy = cy
                            best_ang = ang

        # Try other angles if we have time
        if time.time() < deadline - 20:
            for ang in np.arange(0, 60, 5):
                if ang in [0, 30, 60]:
                    continue
                # Quick check with initial center
                S = compute_outer_hex(inner, best_cx, best_cy, ang)
                if S < best_S + 0.1: # Only refine if promising
                    # Refine center
                    for cx in np.arange(best_cx - 0.5, best_cx + 0.5, 0.1):
                        for cy in np.arange(best_cy - 0.5, best_cy + 0.5, 0.1):
                            S = compute_outer_hex(inner, cx, cy, ang)
                            if S < best_S:
                                best_S = S
                                best_cx = cx
                                best_cy = cy
                                best_ang = ang
                                # Refine more
                                for cx2 in np.arange(cx - 0.1, cx + 0.1, 0.02):
                                    for cy2 in np.arange(cy - 0.1, cy + 0.1, 0.02):
                                        S = compute_outer_hex(inner, cx2, cy2, ang)
                                        if S < best_S:
                                            best_S = S
                                            best_cx = cx2
                                            best_cy = cy2
                                            best_ang = ang

        return best_S, best_cx, best_cy, best_ang

    # Main Strategy:
    # 1. Generate the canonical "blob" of 11 hexagons on the lattice.
    # 2. Try different global rotations of this blob.
    # 3. For each, optimize the outer hexagon.
    
    best_S = float('inf')
    best_inner = None
    best_center = (0.0, 0.0)
    best_angle = 0.0
    best_valid = False

    # Generate the cluster
    # The 11 closest points on a triangular lattice are unique in shape.
    # We generate them and then rotate the whole set.
    cluster_points = generate_hex_clusters(N)
    
    # Center the cluster at origin
    cx0 = sum(p[0] for p in cluster_points) / N
    cy0 = sum(p[1] for p in cluster_points) / N
    centered_points = [(p[0] - cx0, p[1] - cy0) for p in cluster_points]
    
    # Try global rotations of the cluster
    # The cluster has 6-fold symmetry, so we only need to search 0 to 60 degrees.
    # However, the outer hexagon also rotates.
    # Let's try global rotations of the cluster in steps of 5 degrees.
    
    for rot_deg in range(0, 60, 5):
        if time.time() > deadline - 5:
            break
            
        rad = math.radians(rot_deg)
        ca, sa = math.cos(rad), math.sin(rad)
        
        inner = []
        for x, y in centered_points:
            rx = x * ca - y * sa
            ry = x * sa + y * ca
            # The individual hexagons are aligned with the lattice (angle 0 relative to lattice).
            # If we rotate the lattice, we should rotate the hexagons too?
            # The task allows individual rotation.
            # If we keep them aligned with the "lattice frame", their absolute angle is rot_deg.
            # But since they are regular hexagons, angle mod 60 is what matters.
            # Let's set their angle to rot_deg to maintain the lattice structure relative to each other.
            # Actually, if we just rotate the positions, the "shape" is preserved.
            # The inner hexagons can be at any angle.
            # To maintain the "touching" property of the lattice packing, they should be rotated by rot_deg.
            inner.append([rx, ry, float(rot_deg)])
        
        S, ocx, ocy, oang = optimize_outer_hex(inner)
        
        if all_hexagons_fit(inner, ocx, ocy, S, oang):
            if S < best_S:
                best_S = S
                best_inner = inner
                best_center = (ocx, ocy)
                best_angle = oang
                best_valid = True

    # If the rigid cluster didn't work or wasn't found, try a few other simple configurations
    # e.g. a line, or a different blob
    if not best_valid:
        # Fallback: Line
        spacing = math.sqrt(3)
        inner = [[i * spacing, 0.0, 0.0] for i in range(N)]
        S, ocx, ocy, oang = optimize_outer_hex(inner)
        if all_hexagons_fit(inner, ocx, ocy, S, oang):
            best_S = S
            best_inner = inner
            best_center = (ocx, ocy)
            best_angle = oang
            best_valid = True

    if not best_valid:
        # Emergency fallback
        inner = [[0.0, 0.0, 0.0] for _ in range(N)]
        S = 11.0
        return inner, (0.0, 0.0), S, 0.0

    return best_inner, best_center, best_S, best_angle
# EVOLVE-BLOCK-END
