import itertools
import math
import time

N = 11
# Lattice vectors for flat-top hexagons
# v1 = (sqrt(3), 0)
# v2 = (sqrt(3)/2, 1.5)
sqrt3 = math.sqrt(3)
v1 = (sqrt3, 0.0)
v2 = (sqrt3/2.0, 1.5)

# Generate points within radius 3 (axial coords)
# Axial to Cartesian: x = sqrt3 * q + sqrt3/2 * r, y = 1.5 * r
points = []
for q in range(-3, 4):
    for r in range(-3, 4):
        # Check hex distance
        dist = (abs(q) + abs(r) + abs(q+r)) / 2.0
        if dist <= 3.0:
            x = sqrt3 * q + sqrt3/2.0 * r
            y = 1.5 * r
            points.append((x, y, q, r))

# We have 19 points.
# We need to choose 11.
# To speed up, we can pre-compute projections.
# u1 = (0, 1)
# u2 = (sqrt3/2, 0.5)
# u3 = (sqrt3/2, -0.5)

u1 = (0.0, 1.0)
u2 = (sqrt3/2.0, 0.5)
u3 = (sqrt3/2.0, -0.5)

projections = []
for p in points:
    x, y, q, r = p
    p1 = x * u1[0] + y * u1[1]
    p2 = x * u2[0] + y * u2[1]
    p3 = x * u3[0] + y * u3[1]
    projections.append((p1, p2, p3, p))

def _construct_impl():
    best_score = float('inf')
    best_combo = None
    best_outer = 0.0

    start_time = time.time()
    limit = 140.0

    # Iterate over combinations
    # 19 choose 11 is ~75k. Very fast.
    for combo in itertools.combinations(range(len(points)), N):
        if time.time() - start_time > limit:
            break
        
        # Calculate spans
        min_p1 = min(projections[i][0] for i in combo)
        max_p1 = max(projections[i][0] for i in combo)
        span1 = max_p1 - min_p1
        
        min_p2 = min(projections[i][1] for i in combo)
        max_p2 = max(projections[i][1] for i in combo)
        span2 = max_p2 - min_p2
        
        min_p3 = min(projections[i][2] for i in combo)
        max_p3 = max(projections[i][2] for i in combo)
        span3 = max_p3 - min_p3
        
        max_span = max(span1, span2, span3)
        outer_side = max_span / sqrt3 + 1.0
        
        if outer_side < best_score:
            best_score = outer_side
            best_combo = combo
            best_outer = outer_side

    # Construct output
    if best_combo is not None:
        inner = []
        for i in best_combo:
            x, y, q, r = points[i]
            inner.append([x, y, 0.0])
        
        # Center of the outer hexagon should be the centroid of the inner hexagons?
        # Or the center of the bounding box?
        # For a regular hexagon container, the center is the center of symmetry.
        # If the packing is symmetric, the centroid is the center.
        # If not, we should center the container such that the spans are balanced?
        # The formula S = Span/sqrt(3) + 1 assumes the container is centered at the origin
        # if the spans are calculated from the origin?
        # No, Span is max - min.
        # The container must enclose the points.
        # If the points are not centered at 0, the container center should be shifted.
        # The side length S depends only on the spans.
        # The center of the container should be the midpoint of the bounding box in each direction?
        # For a hexagon, the center is unique.
        # Let's compute the center as the average of the min/max projections?
        # Actually, the center of the hexagon that minimizes S for a given set of points
        # is the center of the minimum enclosing hexagon.
        # For aligned hexagons, the center coordinates can be found by averaging the midpoints.
        
        cx = (min(projections[i][3][0] for i in best_combo) + max(projections[i][3][0] for i in best_combo)) / 2.0
        cy = (min(projections[i][3][1] for i in best_combo) + max(projections[i][3][1] for i in best_combo)) / 2.0
        
        # However, the x/y bounding box is not the hexagonal bounding box.
        # We should center it such that the projections are centered.
        # Center projection in u1: (min_p1 + max_p1)/2
        # Center projection in u2: (min_p2 + max_p2)/2
        # Center projection in u3: (min_p3 + max_p3)/2
        # If the set is symmetric, these are all 0 (if centered at 0).
        # If not, we need to find (cx, cy) such that the projections of (cx, cy) match the midpoints.
        # This is a linear system.
        
        # Let's just use the centroid of the points as an approximation, or solve for the center.
        # Solve for C = (cx, cy)
        # C . u1 = (min_p1 + max_p1) / 2
        # C . u2 = (min_p2 + max_p2) / 2
        
        m1 = (min(projections[i][0] for i in best_combo) + max(projections[i][0] for i in best_combo)) / 2.0
        m2 = (min(projections[i][1] for i in best_combo) + max(projections[i][1] for i in best_combo)) / 2.0
        
        # u1 = (0, 1) => cy = m1
        # u2 = (sqrt3/2, 0.5) => sqrt3/2 * cx + 0.5 * cy = m2
        cy = m1
        cx = (m2 - 0.5 * cy) / (sqrt3 / 2.0)
        
        return inner, (cx, cy), best_score, 0.0
    else:
        # Fallback
        return [[0.0, 0.0, 0.0]], (0.0, 0.0), 1.0, 0.0

def construct():
    return _construct_impl()
