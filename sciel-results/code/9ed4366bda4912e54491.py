"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import math

N = 12

# EVOLVE-BLOCK-START
def construct():
    # Place 12 unit hexagons in a hexagonal arrangement (1 + 6 + 5)
    # Using axial coordinates for a regular hex grid
    # Center hexagon + first ring (6) + 5 from second ring
    
    s3 = math.sqrt(3)
    
    # Axial to Cartesian conversion for flat-top hexagons with side 1
    # x = s3 * (q + r/2)
    # y = 1.5 * r
    # Each hexagon occupies a cell; centers are at these positions
    
    positions_axial = [
        (0, 0),      # center
        (1, 0),      # ring 1
        (0, 1),
        (-1, 1),
        (-1, 0),
        (0, -1),
        (1, -1),
        # 5 from ring 2 (asymmetric selection to potentially reduce bounding box)
        (2, 0),
        (2, -1),
        (1, -2),
        (0, -2),
        (-1, -1),
    ]
    
    inner = []
    for q, r in positions_axial:
        x = s3 * (q + r / 2.0)
        y = 1.5 * r
        inner.append([x, y, 0.0])
    
    # Compute bounding box of all vertices of all hexagons to find minimal containing hexagon
    
    def hex_vertices(cx, cy, side, angle_deg=0.0):
        """Return vertices of a regular hexagon centered at (cx, cy) with given side and rotation."""
        verts = []
        for i in range(6):
            theta = math.radians(angle_deg + 60 * i)
            vx = cx + side * math.cos(theta)
            vy = cy + side * math.sin(theta)
            verts.append((vx, vy))
        return verts
    
    # Collect all vertices
    all_verts = []
    for x, y, ang in inner:
        verts = hex_vertices(x, y, 1.0, ang)
        all_verts.extend(verts)
    
    # Find the minimum side length of a regular hexagon (with any rotation) that contains all vertices
    # A regular hexagon with side S contains point p if for all 6 directions (every 60 deg), 
    # the projection of p onto that direction is within [-S, S] (for a hexagon centered at origin)
    # More precisely: a regular hexagon with circumradius R = S has support function h(theta) = S * cos(theta - k*60deg)
    # Actually for a regular hexagon with side length S, the apothem is S * sqrt(3)/2
    # The condition for a point (px, py) to be inside a regular hexagon centered at (cx, cy) with side S and rotation phi:
    # For each of the 6 face normals (at angles phi + k*60 for k=0..5), the dot product of (p - c) with the normal 
    # must be <= apothem = S * sqrt(3)/2
    
    # We need to find minimum S such that there exists a center (cx, cy) and rotation phi where all points fit.
    # This is a 2D optimization problem. Let's use a grid search over rotation and then solve for center and S.
    
    # For a given rotation phi, the minimal S is determined by the maximum over all points of the 
    # "width" in each of the 6 directions.
    # For a fixed phi, we need to find (cx, cy) that minimizes max over all vertices of max over k=0..5 of 
    #   |(v - c) . n_k| / apothem_ratio
    # where n_k are the 6 face normals.
    
    # Actually, let's think differently. For a fixed rotation, the containing hexagon problem reduces to:
    # Find the smallest S such that the intersection of 6 half-planes (defined by the hexagon sides) contains all points.
    # The optimal center is the centroid of the bounding box in each pair of opposite directions.
    
    # For a fixed rotation phi, define the 6 normal directions:
    # n_k = (cos(phi + k*pi/3), sin(phi + k*pi/3)) for k=0..5
    # For each direction n_k, the projection of all points onto n_k gives [min_k, max_k]
    # The width in direction n_k is max_k - min_k
    # For the hexagon to contain all points, we need:
    #   S * sqrt(3)/2 >= max_k (projection of all points onto n_k) - min_k (projection of all points onto n_k)
    # Wait, that's not quite right. The hexagon centered at c with side S contains all points iff for each k:
    #   (p - c) . n_k <= S*sqrt(3)/2 for all p, and (p - c) . n_k >= -S*sqrt(3)/2 for all p
    # So: c . n_k + max_p (p . n_k) <= S*sqrt(3)/2  and  c . n_k + min_p (p . n_k) >= -S*sqrt(3)/2
    # => c . n_k <= S*sqrt(3)/2 - max_p (p . n_k)
    # => c . n_k >= -S*sqrt(3)/2 - min_p (p . n_k)
    # For a solution c to exist, we need:
    #   -S*sqrt(3)/2 - min_p (p . n_k) <= S*sqrt(3)/2 - max_p (p . n_k) for all k
    #   => max_p (p . n_k) - min_p (p . n_k) <= S*sqrt(3)
    #   => S >= (max_p (p . n_k) - min_p (p . n_k)) / sqrt(3)
    
    # So for a fixed rotation phi, the minimum S is:
    #   S_min(phi) = max_k [ (max_p (p . n_k) - min_p (p . n_k)) / sqrt(3) ]
    
    # Let's search over phi in [0, pi/3) (since hexagon has 6-fold symmetry, period is pi/3)
    
    best_S = float('inf')
    best_phi = 0.0
    
    # Grid search over phi
    n_steps = 360
    for i in range(n_steps):
        phi = 2 * math.pi * i / (6 * n_steps)  # range [0, pi/3)
        S_candidate = 0.0
        for k in range(6):
            theta = phi + k * math.pi / 3
            nx = math.cos(theta)
            ny = math.sin(theta)
            projections = [vx * nx + vy * ny for vx, vy in all_verts]
            proj_range = max(projections) - min(projections)
            S_candidate = max(S_candidate, proj_range / s3)
        if S_candidate < best_S:
            best_S = S_candidate
            best_phi = phi
    
    # Now find the center: for the optimal phi, the center should be at the midpoint of the projections
    phi = best_phi
    cx = 0.0
    cy = 0.0
    for k in range(6):
        theta = phi + k * math.pi / 3
        nx = math.cos(theta)
        ny = math.sin(theta)
        projections = [vx * nx + vy * ny for vx, vy in all_verts]
        proj_mid = (max(projections) + min(projections)) / 2.0
        # c . n_k = proj_mid
        # This is a system of 6 equations (but only 3 are independent due to symmetry)
        # We can use least squares to find (cx, cy)
    
    # Solve for center using least squares
    import numpy as np
    A = np.zeros((6, 2))
    b = np.zeros(6)
    for k in range(6):
        theta = phi + k * math.pi / 3
        nx = math.cos(theta)
        ny = math.sin(theta)
        A[k] = [nx, ny]
        projections = [vx * nx + vy * ny for vx, vy in all_verts]
        b[k] = (max(projections) + min(projections)) / 2.0
    
    center, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
    cx, cy = center[0], center[1]
    
    return inner, (cx, cy), best_S, math.degrees(phi)
# EVOLVE-BLOCK-END
