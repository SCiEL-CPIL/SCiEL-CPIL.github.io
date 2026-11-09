"""Packing 12 unit hexagons in minimal regular hexagon."""
import math
import time
import itertools

N = 12

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 130.0  # Safety margin before 150s kill

    sqrt3 = math.sqrt(3.0)

    # Helper: Check if two unit hexagons overlap
    # Distance between centers must be >= 2 * apothem = sqrt(3)
    # Actually, for two regular hexagons of side 1, the minimum center distance
    # to be disjoint (touching allowed) is 2 * (sqrt(3)/2) = sqrt(3) if aligned?
    # No. The distance between parallel sides is sqrt(3).
    # If they touch side-to-side, center distance is sqrt(3).
    # If they touch vertex-to-vertex?
    # The "diameter" of a hexagon is 2. The "width" is sqrt(3).
    # Two hexagons are disjoint if the distance between centers is at least
    # the sum of their "radii" in the direction of the center line.
    # For regular polygons, the minimum distance to avoid overlap is
    # 2 * apothem = sqrt(3) if the centers align with apothems.
    # However, if rotated, the distance can be smaller? No, the convex hull
    # of a unit hexagon has inradius sqrt(3)/2.
    # The Minkowski sum of two unit hexagons (rotated arbitrarily) has
    # inradius at least sqrt(3).
    # So, distance >= sqrt(3) is a necessary condition.
    # Is it sufficient?
    # If two convex sets A and B are disjoint, dist(0, A-B) > 0.
    # A-B contains the Minkowski sum of their support functions.
    # For regular hexagons, the support function is periodic.
    # The minimum distance between centers of two disjoint unit hexagons is sqrt(3).
    # This occurs when they touch side-to-side.
    # If they touch vertex-to-vertex, the distance is 2 (larger).
    # So dist >= sqrt(3) is the correct condition for non-overlap.

    min_dist_sq = 3.0  # (sqrt(3))^2

    def overlaps(p1, p2):
        dx = p1[0] - p2[0]
        dy = p1[1] - p2[1]
        return dx*dx + dy*dy < min_dist_sq - 1e-9

    # Generate candidate positions on a hexagonal grid
    # Grid spacing: sqrt(3) is the minimum distance.
    # We can use a grid with step size sqrt(3)/2 in x and y?
    # Standard hex grid axial to cartesian:
    # x = 1.5 * q, y = sqrt(3) * (r + q/2)
    # This grid has nearest neighbor distance sqrt(3).
    # Let's generate a dense set of candidate centers.
    
    # We need to pack 12 hexagons.
    # A cluster of 12 hexagons fits in a hexagon of side 3?
    # Side 3 hexagon has 1 + 6 + 12 = 19 cells.
    # So 12 is less than 19.
    # We can search for 12 non-overlapping points from the 19 available in a side-3 hexagon.
    # Or we can search a larger area.
    
    # Let's generate candidates in a bounding box.
    # The optimal packing likely fits in a hexagon of side L ~ 4-5.
    # Let's generate points in a range [-5, 5].
    
    # Using a fine grid might be too slow for brute force 12-choose-N.
    # Instead, let's use a specific known good configuration or a heuristic search.
    
    # Strategy: 
    # 1. Define a set of candidate points (hex grid).
    # 2. Use a backtracking search to place 12 hexagons.
    # 3. For each valid placement, calculate the minimal enclosing hexagon side length.
    # 4. Keep the best one.

    # Candidate points: Hex grid with spacing sqrt(3).
    # Axial coordinates (q, r).
    # x = 1.5 * q
    # y = sqrt(3) * (r + 0.5 * q)
    
    max_coord = 6
    candidates = []
    for q in range(-max_coord, max_coord + 1):
        for r in range(-max_coord, max_coord + 1):
            x = 1.5 * q
            y = sqrt3 * (r + 0.5 * q)
            candidates.append((x, y))
            
    # To make the search tractable, we should restrict the number of candidates.
    # Or use a smarter search.
    # With ~100 candidates, choosing 12 is hard.
    # Let's try a local search or simulated annealing on the positions.
    # But the prompt suggests a different strategy.
    # The "linear programming" hint was one option. 
    # My chosen strategy: "Grid-based discrete search".
    
    # Let's refine the grid search.
    # We can limit the search space to a smaller cluster.
    # The 12 hexagons must fit in the outer hexagon.
    # If we assume the optimal outer hexagon is centered at (0,0) and aligned,
    # we can just find the max distance of the inner hexagons to the boundary.
    
    # Let's try to find a packing of 12 hexagons in a fixed grid, then compute the required outer size.
    # We will use a backtracking algorithm to find a set of 12 non-overlapping points.
    # To speed it up, we can limit the candidates to those within a certain radius.
    
    # Let's pick the 19 centers of a side-3 hexagon (radius 2 in axial coords).
    # This is a very dense packing.
    # We just need to pick 12 of them.
    # Since they are all non-overlapping (distance sqrt(3)), ANY 12 of them form a valid packing.
    # To minimize the outer hexagon, we want the 12 points to be as "compact" as possible.
    # The "compactness" is measured by the minimal enclosing hexagon.
    
    # The 19 points are:
    hex_centers = []
    for q in range(-2, 3):
        for r in range(-2, 3):
            if abs(q + r) <= 2:
                x = 1.5 * q
                y = sqrt3 * (r + 0.5 * q)
                hex_centers.append((x, y))
                
    # There are 19 points. We need to choose 12.
    # C(19, 12) = C(19, 7) = 50388. This is small enough to brute force!
    
    best_side = float('inf')
    best_config = None
    
    # Function to calculate minimal enclosing regular hexagon side length
    # for a set of points, assuming the outer hexagon is centered at (0,0) 
    # and can be rotated? 
    # The task says: "outer hexagon center (x, y), side length and rotation".
    # So we can optimize center and rotation too.
    # However, for a symmetric set of points, center (0,0) is likely optimal.
    # And rotation 0 or 30 might be optimal.
    
    # Let's first assume center (0,0) and try a few rotations.
    # The distance from a point (x,y) to the boundary of a regular hexagon
    # centered at origin, side L, rotation theta:
    # The hexagon is defined by 6 inequalities.
    # For a flat-top hexagon (rotation 0), the boundaries are:
    # x <= L
    # x >= -L
    # y <= L * sqrt(3)
    # y >= -L * sqrt(3)
    # Wait, for a regular hexagon of side L:
    # Width (vertex to vertex) = 2L.
    # Height (flat to flat) = L * sqrt(3).
    # If flat-top:
    # x in [-L, L]
    # y in [-L*sqrt(3)/2 * 2? No.]
    
    # Let's use the support function.
    # A regular hexagon with side L centered at origin, rotated by theta.
    # The distance from origin to the boundary in direction phi is:
    # r(phi) = (L * sqrt(3)) / (2 * cos((phi - theta) mod 60 - 30)) ?
    # Actually, the apothem a = L * sqrt(3) / 2.
    # The distance to the boundary in direction phi is a / cos(delta),
    # where delta is the angle difference from the normal to the side.
    # The normals are at angles theta + k * 60.
    # So for a point (x, y) with polar angle phi and radius r,
    # it is inside if r <= a / cos((phi - theta) mod 60 - 30) is not quite right.
    
    # Simpler:
    # The hexagon is the intersection of 6 half-planes.
    # For a side L, the distance from center to each side (apothem) is a = L * sqrt(3) / 2.
    # The normals to the sides are at angles alpha_k = theta + k * pi/3.
    # A point (x,y) is inside if x * cos(alpha_k) + y * sin(alpha_k) <= a for all k.
    # So for a given set of points, and a given rotation theta,
    # the required apothem a is max over all points and all k of (x_i cos(alpha_k) + y_i sin(alpha_k)).
    # Then L = 2 * a / sqrt(3).
    # We want to minimize L, so we minimize a.
    # We can search over theta in [0, pi/3) with some step size.
    
    def min_enclosing_hex_side(points, theta):
        max_a = 0.0
        for x, y in points:
            for k in range(6):
                alpha = theta + k * math.pi / 3.0
                val = x * math.cos(alpha) + y * math.sin(alpha)
                if val > max_a:
                    max_a = val
        return 2.0 * max_a / sqrt3

    # We also need to consider the center of the outer hexagon.
    # Shifting the outer hexagon is equivalent to shifting the points.
    # To minimize the max distance, the center should be near the centroid.
    # For now, let's assume the outer hexagon is centered at the centroid of the inner points.
    # Or even simpler: center at (0,0) and see if it works.
    # Given the symmetry of the hex grid, (0,0) is a good guess.
    
    # Let's iterate over all combinations of 12 points from the 19.
    count = 0
    for combo in itertools.combinations(range(19), 12):
        if time.time() > deadline:
            break
        points = [hex_centers[i] for i in combo]
        
        # Calculate centroid
        cx = sum(p[0] for p in points) / 12.0
        cy = sum(p[1] for p in points) / 12.0
        
        # Center the points at origin for calculation
        centered_points = [(p[0] - cx, p[1] - cy) for p in points]
        
        # Try a few rotations
        best_L_for_combo = float('inf')
        for theta in [0.0, math.pi/6, math.pi/12, math.pi/4]:
            L = min_enclosing_hex_side(centered_points, theta)
            if L < best_L_for_combo:
                best_L_for_combo = L
                
        if best_L_for_combo < best_side:
            best_side = best_L_for_combo
            best_config = (points, (cx, cy), best_L_for_combo, theta)
            
        count += 1
        if count % 1000 == 0:
            pass # print(f"Checked {count}, best: {best_side}")

    if best_config is None:
        # Fallback to a simple valid configuration if search fails or times out
        # Just pick the first 12 from the 19
        points = hex_centers[:12]
        cx = sum(p[0] for p in points) / 12.0
        cy = sum(p[1] for p in points) / 12.0
        centered_points = [(p[0] - cx, p[1] - cy) for p in points]
        L = min_enclosing_hex_side(centered_points, 0.0)
        return [[p[0], p[1], 0.0] for p in points], (cx, cy), L, 0.0

    points, center, L, theta = best_config
    # Return inner hexagons in global coordinates
    inner = [[p[0], p[1], 0.0] for p in points]
    return inner, center, L, theta * 180.0 / math.pi

# EVOLVE-BLOCK-END
