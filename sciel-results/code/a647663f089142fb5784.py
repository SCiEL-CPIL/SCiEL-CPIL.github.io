import numpy as np
from scipy.optimize import minimize

N = 12

# EVOLVE-BLOCK-START
def construct():
    # Generate a compact cluster of 12 unit hexagons on a triangular lattice.
    # We use a "hexagonal" packing arrangement.
    # For flat-top hexagons with side 1:
    # Width = 2, Height = sqrt(3)
    # Horizontal spacing between centers in same row: 2? No, adjacent horizontal distance is sqrt(3) * ?
    # Let's stick to standard hex grid coordinates.
    # Axial coordinates (q, r).
    # For pointy-top hexagons:
    # x = size * sqrt(3) * (q + r/2)
    # y = size * 3/2 * r
    # Let's use pointy-top for easier standard formulas or flat-top.
    # Let's use flat-top.
    # x = size * 3/2 * q
    # y = size * sqrt(3) * (r + q/2)
    
    size = 1.0
    # We need 12 hexagons.
    # A cluster of radius 1 has 7 hexagons.
    # We add 5 more to make 12.
    # Let's generate a list of axial coords for a compact shape.
    # Center (0,0)
    # Ring 1: 6 neighbors
    # Total 7.
    # We need 5 more. Let's attach them to the ring 1 to keep it compact.
    # This is essentially finding a minimal perimeter polyhex of size 12.
    # Known minimal perimeter for 12 hexes is achieved by a shape close to a hexagon.
    # A hexagon of radius 2 has 1+6+12 = 19 hexes.
    # Let's just hardcode a good set of axial coordinates for 12 hexes.
    # Or generate them algorithmically.
    
    # Let's use a simple generation:
    # Start with center (0,0).
    # Add neighbors in a spiral or BFS until we have 12.
    # This might not be optimal but is decent.
    
    # Alternatively, use a known compact shape.
    # Let's try to build a "hexagonal" shape of radius 1 (7 hexes) plus 5 attached.
    # To minimize the bounding box, we should extend in a direction that aligns with the outer hexagon's flat side or vertex.
    
    # Let's try this specific set of axial coordinates (q, r):
    # Ring 0: (0,0)
    # Ring 1: (1,0), (1,-1), (0,-1), (-1,0), (-1,1), (0,1)
    # That's 7.
    # Add 5 more from Ring 2 that keep the shape convex-ish.
    # Ring 2 neighbors of Ring 1:
    # (2,0), (2,-1), (1,-2), (0,-2), (-1,-1), (-2,0), (-2,1), (-1,2), (0,2), (1,1), (2,1), (1,0)-wait
    
    # Let's just use a heuristic: place them in a 3-4-3-2 grid pattern?
    # Or 4-4-4?
    
    # Let's try a specific layout in Cartesian coordinates directly for flat-top hexagons.
    # Flat-top hex centers:
    # dx = 1.5
    # dy = sqrt(3)
    
    # Layout:
    # Row 0 (y=0): x = -1.5, 0, 1.5, 3.0  (4 hexes)
    # Row 1 (y=sqrt(3)/2 * 2 = sqrt(3)? No, vertical spacing is sqrt(3) for flat-top?
    # Flat top:
    # Width 2. Height sqrt(3).
    # Centers:
    # x = 1.5 * q
    # y = sqrt(3) * (r + q/2)
    
    # Let's generate 12 points using this formula and pick a compact set.
    # q in [-2, 2], r in [-2, 2]
    # Let's try:
    coords = []
    for q in range(-2, 3):
        for r in range(-2, 3):
            coords.append((q, r))
            
    # We want 12 of these that form a compact cluster.
    # The 12 cells closest to (0,0) in hex distance?
    # Hex distance: (|q| + |r| + |q+r|) / 2
    # Let's sort by hex distance and take first 12.
    # This creates a "hexagonal" shape but truncated.
    # Distance 0: (0,0) -> 1
    # Distance 1: 6 cells -> Total 7
    # Distance 2: 12 cells. We need 5 of them.
    # To keep it symmetric/compact, we pick 5 of the 12 at distance 2.
    # This might create a jagged edge.
    
    # Let's try a different shape: a "rhombus" or "parallelogram" of hexagons?
    # A 3x4 rhombus has 12 hexes.
    # In axial coords, a rhombus can be defined by vectors.
    # Let's just generate a grid and pick a contiguous block.
    
    # Let's try this specific layout of 12 flat-top hexes:
    # 4 columns, each with 3 hexes? Or 3 columns of 4?
    # Let's use axial coords:
    # q from 0 to 2 (3 columns)
    # r from 0 to 3 (4 rows)
    # This forms a 3x4 block.
    # Let's convert these to Cartesian and center them.
    
    points = []
    for q in range(3):
        for r in range(4):
            # Flat top
            x = 1.5 * q
            y = np.sqrt(3) * (r + q / 2.0)
            points.append((x, y))
            
    # Convert to numpy array
    pts = np.array(points)
    
    # Center the points
    center = pts.mean(axis=0)
    pts = pts - center
    
    # Now we have 12 points (centers of unit flat-top hexagons).
    # The hexagons have side length 1.
    # We need to find the minimum side length S of a regular hexagon (outer) that contains all these unit hexagons.
    # The outer hexagon can be rotated.
    
    # A regular hexagon with side length S and rotation angle theta can be defined by its support function.
    # For a point (x, y) to be inside the outer hexagon (centered at origin for now), it must satisfy:
    # |x cos(theta) + y sin(theta)| <= S * cos(pi/6) ? No.
    # Let's derive the inequality.
    # A regular hexagon with side S has inradius r_in = S * sqrt(3) / 2.
    # It is the intersection of 6 strips.
    # The normals to the sides are at angles theta, theta + 60, ..., theta + 300.
    # A point (x,y) is inside if for all k in 0..5:
    # x * cos(theta + k*pi/3) + y * sin(theta + k*pi/3) <= r_in
    
    # So, for a given theta, the required r_in is max over all points p and all k of p.dot(n_k).
    # Then S = r_in / (sqrt(3)/2) = 2 * r_in / sqrt(3).
    
    # We want to minimize S(theta).
    
    def required_side(theta):
        # Normals
        angles = theta + np.arange(6) * np.pi / 3.0
        normals = np.column_stack((np.cos(angles), np.sin(angles)))
        
        # Max projection of any point onto any normal
        # pts is (12, 2), normals is (6, 2)
        # We want max over i (points) and j (normals) of pts[i].dot(normals[j])
        # This is max over j of max over i of pts[i].dot(normals[j])
        
        max_proj = 0.0
        for n in normals:
            # Project all points onto n
            projs = pts @ n
            max_p = np.max(projs)
            if max_p > max_proj:
                max_proj = max_p
        # The hexagon is centered at 0. The points are centered at 0.
        # However, the points might not be symmetric.
        # The condition is p.dot(n) <= r_in for ALL p and n.
        # So r_in must be >= max(p.dot(n)) for all p, n.
        # Is it also true for -n? Yes, because n covers 360 degrees.
        # So r_in = max_proj.
        
        return 2.0 * max_proj / np.sqrt(3.0)
    
    # Optimize theta
    # S(theta) is periodic with period pi/3 (60 deg) because the hexagon has 6-fold symmetry.
    # Actually, the set of normals rotates. The function is periodic with pi/3.
    # Let's minimize over [0, pi/3].
    
    best_theta = 0.0
    best_side = np.inf
    
    # Sample theta
    thetas = np.linspace(0, np.pi / 3.0, 100, endpoint=False)
    sides = [required_side(t) for t in thetas]
    idx = np.argmin(sides)
    best_theta = thetas[idx]
    best_side = sides[idx]
    
    # Refine with scipy
    res = minimize(lambda t: required_side(t[0]), [best_theta], method='Nelder-Mead',
                   options={'xatol': 1e-6, 'fatol': 1e-6})
    
    if res.success:
        best_theta = res.x[0]
        best_side = res.fun
    
    # The outer hexagon side length is best_side.
    # The outer hexagon center is (0,0) in our local frame, which is the centroid of the inner hexes.
    # In global frame, we placed them at (0,0).
    
    # Construct the return values.
    # inner: list of [x, y, angle_degrees]
    # The inner hexagons are flat-top. What is their angle?
    # The problem says "inner rows (x, y, angle_degrees)".
    # Usually angle is the rotation of the hexagon.
    # A standard flat-top hexagon has a side horizontal.
    # If we define angle=0 as flat-top, then we return 0.
    # If we define angle=0 as pointy-top, we need to adjust.
    # Let's assume angle=0 means the hexagon is in its "default" orientation.
    # Let's check the seed program.
    # Seed: inner = [[4.0 * k, 0.0, 0.0] ...]
    # Side 40, angle 0.
    # It's ambiguous, but usually angle 0 corresponds to the orientation used in the coordinate generation.
    # Let's use angle 0 for all inner hexagons since they are all flat-top.
    
    inner_list = []
    for p in pts:
        inner_list.append([p[0], p[1], 0.0])
    
    # Outer hexagon center is (0,0)
    outer_center = (0.0, 0.0)
    outer_side = best_side
    outer_angle = best_theta * 180.0 / np.pi
    
    return inner_list, outer_center, outer_side, outer_angle
# EVOLVE-BLOCK-END
