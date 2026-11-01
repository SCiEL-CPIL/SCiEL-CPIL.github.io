"""Packing 11 unit hexagons inside a minimal regular hexagon."""
import numpy as np
from scipy.optimize import minimize

N = 11

# EVOLVE-BLOCK-START
def construct():
    # We pack 11 unit hexagons on a hexagonal lattice (centers at integer (u,v) coords in axial system)
    # Axial to Cartesian: x = sqrt(3)*(u + v/2), y = 3/2*v  (for flat-top? let's use pointy-top standard)
    # Actually let's use the standard: hexagon side 1, flat-top orientation
    # Center at (0,0), vertices at angles 0, 60, ... 300 degrees
    # Distance from center to vertex = 1 (circumradius = side for regular hexagon)
    # Distance from center to side midpoint = sqrt(3)/2 (apothem)
    
    # For two non-overlapping unit hexagons (both regular, possibly rotated), the minimum center distance
    # for them to not overlap is 2 * apothem = sqrt(3) if aligned, but with rotation it can be less.
    # However, for simplicity and to ensure no overlap, we place them on a hex lattice with center distance >= 2*sqrt(3)/2 = sqrt(3) * ... 
    # Wait, two unit hexagons on hex lattice: adjacent centers are sqrt(3) apart. This is exactly the minimum
    # distance for two flat-top unit hexagons to touch edge-to-edge.
    
    # So we use axial coordinates (q, r) and convert to Cartesian:
    # x = sqrt(3) * (q + r/2), y = 3/2 * r  (this is for "pointy-top" hexagons where flat sides are left/right)
    # Let me verify: hexagon with flat top has vertices at (±1, 0), (±1/2, ±sqrt(3)/2)
    # Apothem (distance to side) = sqrt(3)/2 in y direction for top/bottom sides? No.
    # Flat-top hexagon: sides are horizontal top and bottom.
    # Vertices: (1, 0), (0.5, sqrt(3)/2), (-0.5, sqrt(3)/2), (-1, 0), (-0.5, -sqrt(3)/2), (0.5, -sqrt(3)/2)
    # So width (x-direction) = 2, height (y-direction) = sqrt(3)
    # Two such hexagons side by side horizontally: centers 2 apart? No, they touch at edges.
    # The distance between centers of two adjacent flat-top hexagons (sharing a vertical edge) is 2 * apothem_horizontal.
    # The apothem in the x-direction (to left/right vertices) is 1. So center distance = 2.
    # The apothem in the y-direction (to top/bottom edges) is sqrt(3)/2. 
    # For hexagons stacked vertically (sharing horizontal edge), center distance = 2 * sqrt(3)/2 = sqrt(3).
    
    # So the hex lattice spacing:
    # In q-direction: 2 (horizontal)
    # In r-direction: sqrt(3) (vertical for stacked)
    # Axial to Cartesian (flat-top): x = 2*q + sqrt(3)*r*... 
    # Let me just use a simple grid. I'll place 11 hexagons in a compact shape.
    # A 3x3 block minus 1 = 8, not enough. 
    # Let's use a "flower" pattern: center + 6 around + 4 more? 
    # Or a 2x3 rectangle + extras.
    
    # Let's try to enumerate small clusters and find the one with minimum enclosing hexagon.
    # For 11 hexagons, a good shape is a 3x4 rectangle minus 1, or similar.
    
    # Actually, let's just try a few known compact shapes and compute the enclosing hexagon.
    # Shape 1: 3x4 rectangle (12 hexes) minus 1 corner = 11 hexes
    # Shape 2: "Hexagon" of radius 2 is 1+6+12=19, too big. Radius 1 is 7.
    # 7 + 4 extra attached.
    
    # Let's use axial coordinates. q is x-axis, r is y-axis.
    # Cartesian: x = 1.5*q + sqrt(3)/2*r, y = sqrt(3)*r  (pointy-top, spacing sqrt(3))
    # This is simpler. Pointy-top hexagons have vertices at (0,1), (sqrt(3)/2, 1/2), (sqrt(3)/2, -1/2), (0,-1), (-sqrt(3)/2, -1/2), (-sqrt(3)/2, 1/2)
    # Width (x) = sqrt(3), Height (y) = 2.
    # Adjacent centers: sqrt(3) apart.
    
    # Let's define 11 positions in axial (q, r):
    # A 3x3 block is 9. Add 2 more.
    # q: -1, 0, 1
    # r: -1, 0, 1
    # That's 9. Add (2, 0) and (-2, 0)? Or (0, 2) and (0, -2)?
    # Let's try:
    positions = [
        (-1, -1), (0, -1), (1, -1),
        (-1, 0),  (0, 0),  (1, 0),
        (-1, 1),  (0, 1),  (1, 1),
        (2, 0),
        (2, 1)
    ]
    
    # Convert to Cartesian (pointy-top)
    sqrt3 = np.sqrt(3)
    centers = []
    for q, r in positions:
        x = 1.5 * q + 0.5 * sqrt3 * r
        y = sqrt3 * r
        centers.append((x, y))
    
    centers = np.array(centers)
    
    # Now find the minimal enclosing regular hexagon.
    # A regular hexagon of side S, center (cx, cy), rotation theta.
    # The hexagon contains a point (x, y) if for all 6 directions, the projection is within S.
    # Actually, for a regular hexagon with apothem A = S * sqrt(3)/2, 
    # the condition is: max_k |dot(point - center, n_k)| <= A
    # where n_k are the 6 unit normals to the sides.
    
    # We want to minimize S = 2 * A / sqrt(3).
    # So minimize A = max_k |dot(p_k - c, n_k)| over c, theta.
    # For a fixed theta, the optimal c is the center of the minimal bounding box in the 6 directions.
    # But this is coupled. Let's use scipy.optimize to minimize over (cx, cy, theta, A).
    # Actually, we can fix theta and solve for c and A analytically? No, c affects A.
    
    # Let's use a 3D optimization: variables are (cx, cy, theta).
    # Objective: compute the minimal A required for these parameters.
    # A_min = max_k ( (max_p dot(p, n_k) - min_p dot(p, n_k)) / 2 )
    # Wait, the hexagon is centered at (cx, cy).
    # Let d_p = p - c. We need max_k |dot(d_p, n_k)| <= A for all p.
    # This is equivalent to: for each k, max_p dot(p, n_k) - dot(c, n_k) <= A and dot(c, n_k) - min_p dot(p, n_k) <= A.
    # So A >= max_p dot(p, n_k) - dot(c, n_k) and A >= dot(c, n_k) - min_p dot(p, n_k).
    # Thus A >= max( max_p dot(p, n_k) - dot(c, n_k), dot(c, n_k) - min_p dot(p, n_k) ).
    # Let M_k = max_p dot(p, n_k), m_k = min_p dot(p, n_k).
    # A >= max(M_k - c·n_k, c·n_k - m_k).
    # To minimize A for fixed c, we choose A_k = max(M_k - c·n_k, c·n_k - m_k) and A = max_k A_k.
    # We want to minimize A over c.
    
    # This is a convex optimization problem in c for fixed theta!
    # We can use scipy.optimize.minimize.
    
    def compute_A(cx, cy, theta):
        # Normals for a regular hexagon rotated by theta
        # Base normals (pointy-top, theta=0): 
        # Sides are at angles 30, 90, 150, 210, 270, 330 degrees from x-axis?
        # Pointy-top hexagon: vertices at 90, 150, 210, 270, 330, 30.
        # Normals point to midpoints of sides: 60, 0, -60, -120, 180, 120? 
        # Let's just use 6 directions: theta + k*pi/3 for k=0..5.
        angles = theta + np.arange(6) * np.pi / 3
        normals = np.column_stack([np.cos(angles), np.sin(angles)])
        
        # Project centers onto normals
        dots = centers @ normals.T  # shape (11, 6)
        M = dots.max(axis=0)  # shape (6,)
        m = dots.min(axis=0)  # shape (6,)
        
        c_dot_n = np.array([cx, cy]) @ normals.T
        A_k = np.maximum(M - c_dot_n, c_dot_n - m)
        A = A_k.max()
        return A
    
    def objective(params):
        cx, cy, theta = params
        return compute_A(cx, cy, theta)
    
    # Initial guess: centroid, theta=0
    cx0 = centers[:, 0].mean()
    cy0 = centers[:, 1].mean()
    res = minimize(objective, [cx0, cy0, 0.0], method='Nelder-Mead', options={'maxiter': 10000, 'xatol': 1e-8, 'fatol': 1e-10})
    
    cx, cy, theta = res.x
    A = compute_A(cx, cy, theta)
    S = 2 * A / sqrt3
    
    # The inner hexagons are at centers with 0 rotation (since we placed them on a hex lattice, they are aligned)
    inner = [[float(x), float(y), 0.0] for x, y in centers]
    
    return inner, (float(cx), float(cy)), float(S), float(np.degrees(theta))
# EVOLVE-BLOCK-END
