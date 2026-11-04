import numpy as np

def construct():
    # Use a small epsilon to ensure disjointness
    # 1e-7 is safe for floating point precision while minimizing side length
    eps = 1e-7
    
    # Centers
    centers = []
    # Origin
    centers.append((0.0, 0.0))
    
    # 6 neighbors at dist sqrt(3) + eps
    for k in range(6):
        angle = np.deg2rad(k * 60)
        x = (np.sqrt(3) + eps) * np.cos(angle)
        y = (np.sqrt(3) + eps) * np.sin(angle)
        centers.append((x, y))
        
    # 4 more at dist 3 + 2*eps
    # Angles 30, 90, 150, 210
    for k in range(4):
        angle = np.deg2rad(30 + k * 60)
        x = (3.0 + 2*eps) * np.cos(angle)
        y = (3.0 + 2*eps) * np.sin(angle)
        centers.append((x, y))
        
    inner = []
    for x, y in centers:
        inner.append([x, y, 30.0])
        
    # Calculate required outer side length
    # The cluster has 6-fold symmetry. The bounding hexagon must contain all 11 inner hexagons.
    # For a regular hexagon of side s centered at origin with rotation theta, the distance to boundary
    # in direction phi is: r(phi) = s * cos(60*(phi - theta)) / cos(60) ... wait, simpler:
    # A regular hexagon with side s has inradius r = s * sqrt(3)/2 and circumradius R = s.
    # The distance from center to the boundary in direction phi (measured from a vertex direction)
    # is: d(phi) = s * cos(pi/6) / cos(phi - theta_vertex) where |phi - theta_vertex| <= pi/6.
    # Actually, for a regular hexagon with side s, the support function in direction u is:
    # h(u) = s * cos(pi/6) / cos(alpha) where alpha is the angle between u and the nearest vertex normal.
    
    # Let's use the vertex-based approach.
    # For each inner hexagon, find its 6 vertices.
    # For each vertex, compute the required side length of the outer hexagon (with given rotation) 
    # such that the vertex is inside.
    
    # The outer hexagon has side s, center (0,0), rotation angle theta.
    # Its vertices are at distance s from center, at angles theta + k*60 for k=0..5.
    # The edges are perpendicular to the directions theta + 30 + k*60.
    # The distance from center to each edge (inradius) is s * sqrt(3)/2.
    
    # A point (x, y) is inside the outer hexagon if and only if for all 6 edge normals n_k,
    # dot((x,y), n_k) <= s * sqrt(3)/2.
    # The edge normals are at angles theta + 30 + k*60.
    
    # So for a point p, the required s is:
    # s >= max_k ( dot(p, n_k) ) / (sqrt(3)/2)
    # where n_k = (cos(theta + 30 + k*60), sin(theta + 30 + k*60))
    
    # We need to minimize s over theta.
    
    # Compute all vertices of all 11 inner hexagons.
    # Each inner hexagon has side 1, so its vertices are at distance 1 from its center,
    # at angles (inner_angle + k*60) for k=0..5.
    
    all_vertices = []
    for x, y, ang_deg in inner:
        ang_rad = np.deg2rad(ang_deg)
        for k in range(6):
            v_ang = ang_rad + k * np.pi / 3
            vx = x + np.cos(v_ang)
            vy = y + np.sin(v_ang)
            all_vertices.append((vx, vy))
    
    all_vertices = np.array(all_vertices)
    
    # For a given theta, compute the required side length.
    def required_side(theta):
        # Edge normals of outer hexagon
        normals = []
        for k in range(6):
            n_ang = theta + np.pi/6 + k * np.pi/3
            normals.append((np.cos(n_ang), np.sin(n_ang)))
        normals = np.array(normals)
        
        # For each vertex, compute max dot product with normals
        # dot products: (N x 2) @ (M x 2).T = N x M
        dots = normals @ all_vertices.T  # shape (6, M)
        max_dots = np.max(dots, axis=0)  # shape (M,)
        
        # Required s = max_dots / (sqrt(3)/2)
        s_required = max_dots / (np.sqrt(3)/2)
        return np.max(s_required)
    
    # Minimize required_side(theta) using ternary search.
    # The function is periodic with period 60 degrees (pi/3).
    # Due to symmetry, we can search in [0, pi/3].
    
    # Ternary search
    a = 0.0
    b = np.pi / 3
    for _ in range(100):
        m1 = a + (b - a) / 3
        m2 = b - (b - a) / 3
        if required_side(m1) < required_side(m2):
            b = m2
        else:
            a = m1
    
    theta_opt = (a + b) / 2
    side = required_side(theta_opt)
    
    # Add a tiny safety margin for floating point
    side += 1e-9
    
    return inner, (0.0, 0.0), side, np.rad2deg(theta_opt)
