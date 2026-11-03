import math
import time
import random
import numpy as np

N = 12

# EVOLVE-BLOCK-START
def construct():
    deadline = time.time() + 140.0  # 150s limit, keep 10s margin
    
    # 1. Generate a compact triangular cluster of 12 unit hexagons
    # A hexagonal number arrangement for 12 items isn't a perfect hexagon (which would be 19 or 7),
    # but a "flower" shape or a compact triangle is efficient.
    # Let's use a specific known compact packing for 12 hexagons:
    # It can be viewed as a hexagon of side 3 (19 cells) with 7 removed? No, that's not minimal.
    # Better: A 1-2-3-4-2 arrangement? Or just a dense rectangular-ish block?
    # Actually, the most compact packing of n congruent hexagons is a hexagonal cluster if n is a hexagonal number.
    # 12 is not a hexagonal number.
    # Let's try a "rhombus" or "parallelogram" of hexagons, or a specific optimized cluster.
    # A very compact shape for 12 is a 3x4 rhombus of hexagons?
    # Hexagons tile with 60 degree angles.
    # Let's place them in a grid defined by axial coordinates (q, r).
    # Distance between centers in axial coords (q, r) is:
    # dx = 1.5 * q + 0.5 * r
    # dy = sqrt(3)/2 * r
    # Wait, standard hex grid spacing for side length 1:
    # Center distance is sqrt(3) for adjacent hexes.
    # Let's use offset coordinates for a "pointy top" or "flat top" hex.
    # Flat top hexagons:
    # x = 3/2 * size * q
    # y = sqrt(3) * size * (r + q/2)
    # Here size is side length 1.
    
    # Let's construct a cluster that is roughly circular to minimize the bounding hexagon.
    # A 12-hex cluster can be formed by a central hex + 5 neighbors + 6 more?
    # Let's just use a simple dense packing: a 3x4 parallelogram of hexagons.
    # This shape is convex and easy to bound.
    # Or even better: a "hexagon" of side 2 has 7 cells. We have 12.
    # Let's try a shape that fits well inside a larger hexagon.
    # The minimal bounding hexagon for a set of points is determined by the furthest points in the 6 directions.
    
    # Let's generate coordinates for a 3x4 block of hexagons (flat-top).
    # q ranges 0..2, r ranges 0..3.
    points = []
    for q in range(3):
        for r in range(4):
            # Center of hex at (q, r)
            x = 1.5 * q
            y = math.sqrt(3) * (r + q / 2.0)
            points.append((x, y))
            
    # This gives 12 points.
    # We need to find the minimum side length 'L' of a regular hexagon that contains all these unit hexagons.
    # The unit hexagons have side 1. Their vertices extend to radius 1 from their centers.
    # So we need to enclose the set of points expanded by radius 1 (Minkowski sum with a circle? No, with a hexagon).
    # Actually, if we enclose the centers in a hexagon of side L-1 (roughly), the vertices of the inner hexagons will fit.
    # More precisely: The condition is that for every inner hexagon center c_i and every vertex v of the inner hexagon (relative to c_i),
    # the point c_i + v must be inside the outer hexagon.
    # This is equivalent to: The set of all points (c_i + v) for all i and all 6 vertices v must be inside the outer hexagon.
    # This is equivalent to: The Minkowski sum of the set of centers C and the unit hexagon H (side 1) must be inside the outer hexagon H_outer (side L).
    
    # The Minkowski sum of a set of points and a regular hexagon is the union of hexagons centered at those points.
    # To find the minimal L, we can compute the "extent" of this union in the 6 directions normal to the sides of the outer hexagon.
    # For a regular hexagon centered at origin with side L, the distance from center to each side (apothem) is A_outer = L * sqrt(3)/2.
    # The distance from center to each vertex is L.
    # The condition for a point p to be inside the outer hexagon (aligned with axes, flat-top) is:
    # |p.x| * 0.5 + |p.y| * sqrt(3)/2 <= A_outer ? No, that's for the hexagon defined by |x| + |y/sqrt(3)| ...
    # Let's use the support function.
    # The outer hexagon H_L is the intersection of 6 half-planes.
    # For a flat-top hexagon, the normals are at angles 0, 60, 120, 180, 240, 300.
    # The distance from origin to the side with normal n is A_outer.
    # So, max_{p in Union} (p . n) <= A_outer for all 6 normals n.
    # Union = C + H_1.
    # max_{c in C, v in H_1} ((c + v) . n) = max_{c in C} (c . n) + max_{v in H_1} (v . n).
    # max_{v in H_1} (v . n) is the support function of the unit hexagon in direction n.
    # For a unit hexagon (side 1), the support function in direction n (unit vector) depends on the angle of n relative to the hexagon.
    # If n is normal to a side, the max dot product is the apothem of the unit hexagon = sqrt(3)/2.
    # If n is towards a vertex, the max dot product is the circumradius = 1.
    # Since the outer hexagon is also regular and we can rotate it, we should align the outer hexagon such that its sides are parallel to the sides of the "worst-case" inner hexagons?
    # Actually, the optimal outer hexagon orientation is usually aligned with the symmetry of the cluster.
    # Let's assume the outer hexagon is "flat-top" (sides horizontal/vertical-ish).
    # The normals are at 0, 60, 120, 180, 240, 300 degrees.
    # Let's compute the required apothem A_outer for this orientation.
    
    # Support function of unit hexagon (flat-top, centered at 0) in direction theta:
    # Vertices of unit flat-top hex: (1,0), (1/2, sqrt(3)/2), (-1/2, sqrt(3)/2), (-1,0), (-1/2, -sqrt(3)/2), (1/2, -sqrt(3)/2)
    # Wait, flat-top means vertices are at 0, 60...?
    # Standard flat-top: top and bottom are flat. Vertices at 30, 90, 150...? No.
    # Flat-top: Sides are horizontal. Vertices at 0, 60, 120, 180, 240, 300?
    # If vertices are at 0 deg (1,0), then side between 0 and 60 is not horizontal.
    # Let's define "flat-top" as having horizontal top and bottom sides.
    # Vertices: (1/2, sqrt(3)/2), (-1/2, sqrt(3)/2), (-1, 0), (-1/2, -sqrt(3)/2), (1/2, -sqrt(3)/2), (1, 0).
    # This has vertical sides? No.
    # Let's stick to the definition: A regular hexagon with side 1.
    # The support function h(theta) = max_{v in H} v . (cos theta, sin theta).
    # For a regular hexagon, h(theta) is a periodic function with period 60 deg.
    # At theta = 0 (normal to a side if aligned), h(0) = apothem = sqrt(3)/2.
    # At theta = 30 deg (towards vertex), h(30) = circumradius = 1.
    
    # Let's calculate the required A_outer for the 6 normals of the outer hexagon.
    # We can rotate the outer hexagon by angle alpha.
    # The normals will be at alpha, alpha+60, ...
    # We want to minimize L = 2 * A_outer / sqrt(3).
    # So we want to minimize max_{k=0..5} ( max_{c in C} (c . n_k) + h(alpha + 60k) ).
    # Note: h is the support of the *inner* unit hexagon. The inner hexagons can also be rotated?
    # The task says "each inner hexagon may translate and rotate".
    # To minimize the bounding box, we should align all inner hexagons with the outer hexagon (or in a way that minimizes the support function in the critical directions).
    # If we align all inner hexagons with the outer hexagon, then the support function of the inner hexagon in the direction of the outer normal is simply the apothem of the inner hexagon (if the normal is perpendicular to a side) or circumradius (if towards vertex).
    # Actually, if we align them, the "Minkowski sum" shape is just the cluster of hexagons.
    # The extent in direction n is max_c (c . n) + apothem_inner (if n is normal to side of inner hex).
    # If we rotate the inner hexagons, we might reduce the extent in some directions but increase in others.
    # However, for a regular hexagon, the support function is minimized in the "normal to side" directions and maximized in "vertex" directions.
    # Since the outer hexagon has 6 normals, and the inner hexagon has 6 normals, aligning them means that in each of the 6 outer normal directions, the inner hexagon presents its "flat" side (apothem) if the outer normal coincides with an inner normal, or its "vertex" if it coincides with an inner vertex normal.
    # Wait, the normals of a regular hexagon are 60 deg apart. The vertices are 30 deg offset from the normals.
    # If we align inner and outer hexagons, the outer normals coincide with the inner normals.
    # Therefore, in each of the 6 critical directions, the inner hexagon contributes its apothem (sqrt(3)/2) to the extent.
    # This is the minimal possible contribution for a regular hexagon in any direction (the average support is apothem, max is radius).
    # So, aligning inner and outer hexagons is optimal for minimizing the bounding size.
    
    # So, for each of the 6 directions n_k (angle alpha + 60k):
    # Extent_k = max_{c in C} (c . n_k) + sqrt(3)/2
    # We need A_outer >= Extent_k for all k.
    # So A_outer = max_k ( max_{c in C} (c . n_k) + sqrt(3)/2 )
    # L = 2 * A_outer / sqrt(3)
    
    # We can optimize over alpha (rotation of the outer hexagon relative to the cluster).
    # The cluster C is fixed (the 3x4 block).
    # Let's compute max_c (c . n) for the 6 directions as a function of alpha.
    
    # Centers of the 3x4 block:
    # q in 0..2, r in 0..3
    # x = 1.5 * q
    # y = sqrt(3) * (r + q/2)
    
    centers = np.array([(1.5 * q, math.sqrt(3) * (r + q / 2.0)) for q in range(3) for r in range(4)])
    
    # We need to find alpha that minimizes max_k ( max_c (centers[c] . n_k(alpha)) + sqrt(3)/2 )
    # n_k(alpha) = (cos(alpha + 60k), sin(alpha + 60k))
    
    # Since the function is periodic with 60 deg, we can search alpha in [0, 60).
    # Actually, due to symmetry of the 3x4 block, it might be less symmetric.
    # Let's do a fine grid search for alpha.
    
    best_L = float('inf')
    best_alpha = 0.0
    
    # The term sqrt(3)/2 is constant for all k if aligned.
    # Let apothem_inner = math.sqrt(3) / 2.0
    
    # Grid search alpha from 0 to 60 degrees
    # Step size 0.1 degrees for precision
    alphas = np.linspace(0, math.pi/3, 600)
    
    for alpha in alphas:
        max_extent = 0.0
        for k in range(6):
            theta = alpha + k * math.pi / 3.0
            nx = math.cos(theta)
            ny = math.sin(theta)
            # Project centers onto normal
            projections = centers[:, 0] * nx + centers[:, 1] * ny
            max_proj = np.max(projections)
            extent = max_proj + apothem_inner
            if extent > max_extent:
                max_extent = extent
        L_candidate = 2.0 * max_extent / math.sqrt(3)
        if L_candidate < best_L:
            best_L = L_candidate
            best_alpha = alpha
            
    # best_L is the side length of the outer hexagon.
    # We need to return:
    # inner: list of [x, y, angle_degrees] for 12 hexagons
    # center: (x, y) of outer hexagon
    # side: best_L
    # angle: best_alpha (in degrees? The interface says angle_degrees for inner, and "outer ... rotation" which usually implies degrees or radians? 
    # The seed returns 0.0 for angle. Let's assume degrees for consistency with inner angle_degrees.)
    
    # The centers we computed are relative to the "origin" of our coordinate system for the cluster.
    # We should center the cluster at the origin to make the outer hexagon center (0,0).
    # Calculate centroid of the centers.
    cx_avg = np.mean(centers[:, 0])
    cy_avg = np.mean(centers[:, 1])
    
    # Shift centers so centroid is at (0,0)
    shifted_centers = centers - np.array([cx_avg, cy_avg])
    
    # The outer hexagon center is (0,0).
    # The inner hexagons are at shifted_centers, with angle 0.0 (aligned with outer).
    # Wait, the outer hexagon is rotated by best_alpha.
    # If we define the outer hexagon in the global frame with rotation best_alpha,
    # and the inner hexagons are aligned with the outer hexagon, then their global angle is also best_alpha.
    # However, the coordinates of the inner hexagons (x,y) are in the global frame.
    # Our `shifted_centers` are in the frame where the cluster is defined.
    # If we rotate the outer hexagon by alpha, do we rotate the cluster too?
    # No, the cluster shape is fixed. We found the optimal orientation of the *bounding box* relative to the cluster.
    # So, in the global frame:
    # 1. Place the cluster centers at `shifted_centers`.
    # 2. The inner hexagons have angle `best_alpha` (to align with outer).
    # 3. The outer hexagon has center (0,0), side `best_L`, and angle `best_alpha`.
    # Let's verify if this fits.
    # The condition was derived for the outer hexagon normals at alpha + 60k.
    # The inner hexagons are aligned with the outer, so their normals are also at alpha + 60k.
    # The support function calculation holds.
    
    inner = []
    for cx, cy in shifted_centers:
        inner.append([float(cx), float(cy), float(math.degrees(best_alpha))])
        
    outer_center = (0.0, 0.0)
    outer_side = float(best_L)
    outer_angle = float(math.degrees(best_alpha))
    
    return inner, outer_center, outer_side, outer_angle
# EVOLVE-BLOCK-END
