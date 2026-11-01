"""Program for placing unit hexagons inside a regular hexagon, minimizing outer side length."""
import numpy as np
from scipy.spatial import ConvexHull

N = 11

# EVOLVE-BLOCK-START
def construct():
    """
    Place N unit regular hexagons (side length 1) inside a regular hexagon,
    minimizing the outer hexagon's side length.
    
    Strategy: Use a hexagonal lattice arrangement for the inner hexagons.
    For N=11, we can use a compact hexagonal cluster.
    
    The inner hexagons are placed on a triangular lattice with spacing 
    sufficient to avoid overlap. Two unit hexagons with the same orientation 
    need a center-to-center distance of at least 2*sqrt(3) in the direction 
    perpendicular to the shared edge (i.e., the distance between parallel edges).
    
    Actually, for two unit regular hexagons with the same orientation, the 
    minimum center distance to avoid overlap is 2 * (apothem) = 2 * (sqrt(3)/2) = sqrt(3) 
    in the direction of the normal to a shared edge. But for general positions, 
    we need to ensure no overlap.
    
    For simplicity and robustness, we place them on a triangular lattice with 
    spacing d = 2 * sqrt(3) + epsilon in the axial directions, which guarantees 
    no overlap for same-oriented hexagons.
    
    Wait, let me reconsider. Two unit hexagons with the same orientation: the 
    distance between centers must be at least 2 * apothem = sqrt(3) in the 
    direction perpendicular to any shared edge. For a triangular lattice with 
    spacing s, the minimum distance between any two centers is s. So we need 
    s >= sqrt(3). But actually, the constraint is more nuanced. Let's use a 
    safe spacing.
    
    For a regular hexagon of side 1, the distance from center to vertex is 1, 
    and from center to edge (apothem) is sqrt(3)/2. Two such hexagons with the 
    same orientation, if their centers are distance d apart in any direction, 
    they overlap if d < 2 * apothem = sqrt(3) in the direction normal to a 
    shared edge. For a triangular lattice, the closest neighbors are at distance 
    s. So s >= sqrt(3) should suffice for same-oriented hexagons on a triangular 
    lattice.
    
    Let's use a hexagonal cluster arrangement. For N=11, a good arrangement is 
    a central hexagon surrounded by 6, with 4 more added. Or we can use a 
    rectangular-ish arrangement on a hex grid.
    
    Let's use axial coordinates on a hex grid. The positions in Cartesian 
    coordinates for a hex grid with spacing s (distance between adjacent centers) 
    are:
    x = s * (q + r/2)
    y = s * (sqrt(3)/2) * r
    
    where q, r are axial coordinates.
    
    For N=11, let's use a compact arrangement. One option is a "lozenge" shape 
    or a small hexagonal cluster.
    
    Let's try: a central hexagon, 6 around it (total 7), then 4 more. 
    Alternatively, use a 3x3 block minus corners, or similar.
    
    Actually, let's use a simple approach: place them in a hexagonal pattern. 
    For N=11, we can use:
    Row 0: 3 hexagons
    Row 1: 4 hexagons  
    Row 2: 3 hexagons
    Row 3: 1 hexagon
    That's 11. Or some other arrangement.
    
    Let's just use a rectangular grid on the hex lattice and compute the 
    bounding hexagon.
    
    For simplicity, let's place them in a line first (like the seed) but then 
    optimize. Actually, let's use a proper hexagonal packing.
    
    I'll use a triangular lattice with spacing s = sqrt(3) + 0.01 for safety.
    Arrange the 11 hexagons in a compact cluster.
    """
    s = np.sqrt(3) + 0.01  # spacing between adjacent hex centers
    
    # Define axial coordinates for 11 hexagons in a compact arrangement
    # Using a "hexagon of hexagons" pattern: center + 6 ring + 4 more
    # Or use a 3-4-3-1 arrangement
    
    # Let's use a simple rectangular arrangement on hex grid:
    # 3 columns x 4 rows = 12, remove one for 11
    # Or 2x5 + 1
    
    # Axial coords (q, r) for a compact 11-hex cluster:
    axial_coords = [
        (0, 0),
        (1, 0), (2, 0),
        (0, 1), (1, 1), (2, 1),
        (0, 2), (1, 2),
        (0, 3), (1, 3), (2, 3)
    ]
    # That's 11 points in a 3x4 rectangle minus one corner
    
    # Convert to Cartesian
    inner = []
    for q, r in axial_coords:
        x = s * (q + r / 2.0)
        y = s * (np.sqrt(3) / 2.0) * r
        inner.append([x, y, 0.0])
    
    # Find the bounding regular hexagon
    # The outer hexagon must contain all inner hexagons.
    # Since all inner hexagons have the same orientation (0 degrees), 
    # the outer hexagon should also be at 0 degrees for minimal size.
    
    # For a regular hexagon of side L centered at (cx, cy) with orientation 0,
    # it contains point (x, y) if the point is within the hexagon.
    # We need to find the minimum L such that all inner hexagons (including 
    # their extent) are inside.
    
    # Each inner hexagon of side 1 extends sqrt(3)/2 from its center in the 
    # apothem direction. So effectively, we need the outer hexagon to contain 
    # all points at distance up to 1 (vertex distance) from each inner center.
    
    # Actually, the inner hexagons have vertices at distance 1 from their centers.
    # So the outer hexagon must contain all vertices of all inner hexagons.
    
    # Let's compute all vertices of all inner hexagons.
    all_points = []
    for x, y, ang in inner:
        # Vertices of a unit hexagon centered at (x, y) with orientation ang
        for k in range(6):
            theta = np.radians(ang + 60 * k)
            vx = x + np.cos(theta)
            vy = y + np.sin(theta)
            all_points.append((vx, vy))
    
    all_points = np.array(all_points)
    
    # Now find the minimum regular hexagon (orientation 0) containing all points.
    # A regular hexagon with orientation 0, center (cx, cy), side L has the property 
    # that a point (x, y) is inside if:
    # |x - cx| <= L
    # |y - cy| <= L * sqrt(3)/2
    # |(x - cx) * sqrt(3) + (y - cy)| <= L * sqrt(3)
    # Wait, let me derive this properly.
    
    # A regular hexagon with flat top (orientation 0) centered at origin, side L:
    # The six vertices are at angles 0, 60, 120, 180, 240, 300 degrees at distance L.
    # The hexagon is the intersection of 6 half-planes.
    # The constraints are:
    # x * cos(0) + y * sin(0) <= L * apothem_factor
    # Actually, for a regular polygon, the distance from center to each edge is the apothem.
    # For a hexagon of side L, apothem = L * sqrt(3)/2.
    # The 6 edges are at angles 30, 90, 150, 210, 270, 330 degrees (normals).
    # So a point (x, y) is inside if:
    # x * cos(30) + y * sin(30) <= L * sqrt(3)/2
    # x * cos(90) + y * sin(90) <= L * sqrt(3)/2
    # x * cos(150) + y * sin(150) <= L * sqrt(3)/2
    # x * cos(210) + y * sin(210) <= L * sqrt(3)/2
    # x * cos(270) + y * sin(270) <= L * sqrt(3)/2
    # x * cos(330) + y * sin(330) <= L * sqrt(3)/2
    
    # For center (cx, cy), replace (x, y) with (x-cx, y-cy).
    
    # We need to find (cx, cy, L) minimizing L such that all points satisfy these.
    
    # This is a convex optimization problem. Let's use a simple approach:
    # For a given orientation (0 degrees), the minimal enclosing hexagon can be 
    # found by considering the support functions in the 6 normal directions.
    
    normals = []
    for k in range(6):
        theta = np.radians(30 + 60 * k)
        normals.append((np.cos(theta), np.sin(theta)))
    
    # For each normal direction, compute the max and min projection of all points.
    # The hexagon must span from min to max in each normal direction.
    # For a regular hexagon centered at (cx, cy) with side L, the extent in 
    # direction theta is: cx*cos(theta) + cy*sin(theta) +/- L*sqrt(3)/2
    
    # So for each normal direction k:
    # min_proj_k <= cx*cos(theta_k) + cy*sin(theta_k) - L*sqrt(3)/2
    # max_proj_k >= cx*cos(theta_k) + cy*sin(theta_k) + L*sqrt(3)/2
    
    # From these:
    # L*sqrt(3)/2 >= max_proj_k - (cx*cos(theta_k) + cy*sin(theta_k))
    # L*sqrt(3)/2 >= (cx*cos(theta_k) + cy*sin(theta_k)) - min_proj_k
    
    # So L >= (2/sqrt(3)) * max_k [max(max_proj_k - proj_c_k, proj_c_k - min_proj_k)]
    # where proj_c_k = cx*cos(theta_k) + cy*sin(theta_k)
    
    # To minimize L, we want to choose (cx, cy) to minimize the maximum of 
    # these quantities.
    
    # Let's compute min and max projections for each normal.
    min_projs = []
    max_projs = []
    for nx, ny in normals:
        projs = all_points[:, 0] * nx + all_points[:, 1] * ny
        min_projs.append(np.min(projs))
        max_projs.append(np.max(projs))
    
    # Now, for a given center (cx, cy), the required L is:
    # L = (2/sqrt(3)) * max_k [max(max_projs[k] - (cx*nx_k + cy*ny_k), (cx*nx_k + cy*ny_k) - min_projs[k])]
    
    # This is a convex function of (cx, cy). We can minimize it using scipy.
    
    from scipy.optimize import minimize
    
    def objective(params):
        cx, cy = params
        max_val = 0
        for k in range(6):
            nx, ny = normals[k]
            proj_c = cx * nx + cy * ny
            val = max(max_projs[k] - proj_c, proj_c - min_projs[k])
            max_val = max(max_val, val)
        return (2.0 / np.sqrt(3)) * max_val
    
    # Initial guess: centroid of all points
    cx0 = np.mean(all_points[:, 0])
    cy0 = np.mean(all_points[:, 1])
    
    result = minimize(objective, [cx0, cy0], method='Nelder-Mead')
    
    cx, cy = result.x
    L = objective([cx, cy])
    
    return inner, (float(cx), float(cy)), float(L), 0.0

# EVOLVE-BLOCK-END
