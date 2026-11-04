import numpy as np
from scipy.spatial import ConvexHull

N = 11

# EVOLVE-BLOCK-START
def construct():
    # Define 11 unit hexagon centers in a compact hexagonal arrangement
    # Hexagonal lattice spacing: distance between adjacent hex centers = sqrt(3)
    d = np.sqrt(3.0)
    
    # Compact arrangement: center + 6 in first ring + 4 in second ring (compact)
    # Using axial coordinates converted to Cartesian
    # Axial coords (q, r) -> Cartesian: x = d * (q + r/2), y = d * (r * sqrt(3)/2)
    def axial_to_cart(q, r):
        x = d * (q + r / 2.0)
        y = d * (r * np.sqrt(3.0) / 2.0)
        return np.array([x, y])
    
    # 11 hexagons in a compact pattern
    # Center: (0,0)
    # First ring: 6 hexagons
    # Second ring: 4 hexagons to make compact shape
    axial_coords = [
        (0, 0),      # center
        (1, 0), (1, -1), (0, -1), (-1, 0), (-1, 1), (0, 1),  # first ring (6)
        (2, 0), (2, -1), (1, -2), (0, -2)  # 4 more in second ring
    ]
    
    centers = np.array([axial_to_cart(q, r) for q, r in axial_coords])
    
    # Center the cluster at origin
    centroid = np.mean(centers, axis=0)
    centers = centers - centroid
    
    # Compute convex hull of the centers
    # But we need to account for the hexagon shape, not just points
    # For now, use the convex hull of centers as approximation
    # Better: compute the Minkowski sum of each hexagon with its center
    
    # Each unit hexagon has vertices at distance 1 from center in 6 directions
    # The convex hull of all hexagon vertices gives the true extent
    
    # Generate all vertices of all hexagons
    all_vertices = []
    for cx, cy in centers:
        for k in range(6):
            angle = np.pi / 3.0 * k
            vx = cx + np.cos(angle)
            vy = cy + np.sin(angle)
            all_vertices.append([vx, vy])
    
    all_vertices = np.array(all_vertices)
    
    # Compute convex hull
    hull = ConvexHull(all_vertices)
    hull_vertices = all_vertices[hull.vertices]
    
    # Function to compute minimal enclosing hexagon side length for given rotation
    def min_hex_side(angle_theta, hull_pts):
        """
        For a regular hexagon with rotation angle_theta, find minimal side length
        such that it contains all hull_pts. Optimize center position.
        """
        # 6 edge normal directions
        normals = []
        for k in range(6):
            ang = angle_theta + k * np.pi / 3.0
            normals.append(np.array([np.cos(ang), np.sin(ang)]))
        normals = np.array(normals)
        
        # For optimal center, we need to minimize max over normals of 
        # (max projection - min projection) / 2, and center at midpoint
        # The required apothem for normal k is (max_proj_k - min_proj_k) / 2
        # The side length s must satisfy s * sqrt(3)/2 >= max over k of required apothem_k
        
        projections = hull_pts @ normals.T  # shape (n_hull, 6)
        max_projs = np.max(projections, axis=0)
        min_projs = np.min(projections, axis=0)
        
        # Required apothem for each direction
        required_apothems = (max_projs - min_projs) / 2.0
        max_required_apothem = np.max(required_apothems)
        
        # Side length: apothem = s * sqrt(3) / 2, so s = 2 * apothem / sqrt(3)
        side_length = 2.0 * max_required_apothem / np.sqrt(3.0)
        
        # Center is at the midpoint of projections
        center = np.mean(hull_pts, axis=0)  # approximate; more precise below
        
        return side_length
    
    # More precise: for each angle, compute optimal center and side length
    def compute_side_and_center(angle_theta, hull_pts):
        normals = []
        for k in range(6):
            ang = angle_theta + k * np.pi / 3.0
            normals.append(np.array([np.cos(ang), np.sin(ang)]))
        normals = np.array(normals)
        
        projections = hull_pts @ normals.T
        max_projs = np.max(projections, axis=0)
        min_projs = np.min(projections, axis=0)
        
        required_apothems = (max_projs - min_projs) / 2.0
        max_required_apothem = np.max(required_apothems)
        side_length = 2.0 * max_required_apothem / np.sqrt(3.0)
        
        # Compute optimal center: for each normal, center projection should be at midpoint
        # Center in projection space: mid_projs = (max_projs + min_projs) / 2
        mid_projs = (max_projs + min_projs) / 2.0
        
        # Find center c such that c @ normals.T = mid_projs
        # This is an overdetermined system; use least squares
        # normals.T @ c = mid_projs
        # c = pinv(normals.T) @ mid_projs
        try:
            center = np.linalg.lstsq(normals.T, mid_projs, rcond=None)[0]
        except:
            center = np.mean(hull_pts, axis=0)
        
        return side_length, center
    
    # Ternary search over rotation angle [0, pi/3) since hexagon has 60° symmetry
    # Actually, the optimal rotation could be anywhere in [0, 2*pi), but due to symmetry
    # of the hexagon, we can search [0, pi/3)
    
    def objective(theta):
        side, center = compute_side_and_center(theta, hull_vertices)
        return side
    
    # Ternary search
    a, b = 0.0, np.pi / 3.0
    for _ in range(100):
        m1 = a + (b - a) / 3.0
        m2 = b - (b - a) / 3.0
        f1 = objective(m1)
        f2 = objective(m2)
        if f1 < f2:
            b = m2
        else:
            a = m1
    
    theta_opt = (a + b) / 2.0
    side_opt, center_opt = compute_side_and_center(theta_opt, hull_vertices)
    
    # Construct inner hexagons
    inner = []
    for cx, cy in centers:
        inner.append([float(cx), float(cy), 0.0])
    
    # Outer hexagon: center, side length, rotation
    outer_center = [float(center_opt[0]), float(center_opt[1])]
    outer_side = float(side_opt)
    outer_rotation = float(theta_opt)
    
    return inner, outer_center, outer_side, outer_rotation
# EVOLVE-BLOCK-END
