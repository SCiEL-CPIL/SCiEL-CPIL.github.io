"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
from scipy.optimize import minimize

N = 11

# EVOLVE-BLOCK-START

def hex_vertices(cx, cy, side, angle_deg):
    """Return vertices of a regular hexagon centered at (cx, cy) with given side length and rotation."""
    angle = np.radians(angle_deg)
    vertices = []
    for i in range(6):
        theta = angle + i * np.pi / 3
        x = cx + side * np.cos(theta)
        y = cy + side * np.sin(theta)
        vertices.append((x, y))
    return np.array(vertices)

def support_function(points, direction):
    """Compute the support function: max dot product of points with direction."""
    return np.max(points @ direction)

def compute_convex_hull(points):
    """Compute convex hull using scipy."""
    from scipy.spatial import ConvexHull
    if len(points) < 3:
        return points
    try:
        hull = ConvexHull(points)
        return points[hull.vertices]
    except Exception:
        return points

def outer_hexagon_side_from_points(points):
    """
    Given a set of points, find the minimum side length of a regular hexagon
    (with any rotation) that contains all points.
    
    We minimize over rotation angle theta the maximum support function
    in the 6 directions of the hexagon.
    """
    def objective(theta):
        # The 6 outward normals of a regular hexagon with rotation theta
        # A regular hexagon has flat sides; the normals are at angles theta + k*pi/3 for k=0..5
        # Actually, let's think carefully. A regular hexagon with "pointy" orientation
        # has vertices at angles theta + k*pi/3. The sides are perpendicular to directions
        # theta + pi/6 + k*pi/3.
        # 
        # For a regular hexagon with circumradius R (distance from center to vertex),
        # the inradius (distance from center to side) is R * cos(pi/6) = R * sqrt(3)/2.
        # The side length s = R.
        # 
        # Wait, for a regular hexagon, side length = circumradius.
        # The inradius = s * sqrt(3)/2.
        #
        # The hexagon can be described as: for each of 6 directions n_k (unit normals to sides),
        # the support function of the hexagon in direction n_k is the inradius = s * sqrt(3)/2.
        #
        # But we need the hexagon to contain all points. So for each direction n_k,
        # the support function of the points in direction n_k must be <= inradius.
        #
        # So we need: max_k support(points, n_k) <= s * sqrt(3)/2
        # => s >= 2/sqrt(3) * max_k support(points, n_k)
        #
        # We also need to consider the opposite directions, but since we take max over all 6
        # and the hexagon is centrally symmetric, we should consider all 6 outward normals.
        
        theta_rad = np.radians(theta)
        normals = []
        for k in range(6):
            # Normals to the sides of a flat-top hexagon rotated by theta
            # For a regular hexagon, the 6 outward normals are at angles theta + k*pi/3
            # if we define theta appropriately. Let's use:
            # Vertices at angles theta + k*pi/3, so side normals at theta + pi/6 + k*pi/3
            normal_angle = theta_rad + np.pi/6 + k * np.pi/3
            normals.append(np.array([np.cos(normal_angle), np.sin(normal_angle)]))
        normals = np.array(normals)
        
        # Also include the opposite directions to ensure containment
        all_normals = np.vstack([normals, -normals])
        
        supports = points @ all_normals.T
        max_support = np.max(supports)
        
        # The inradius must be at least max_support
        # inradius = s * sqrt(3)/2
        # s = 2 * inradius / sqrt(3)
        return 2.0 * max_support / np.sqrt(3.0)
    
    # Optimize over theta
    best_side = float('inf')
    for theta_init in np.linspace(0, 180, 13):
        result = minimize(objective, [theta_init], method='Nelder-Mead', 
                          options={'xatol': 0.1, 'fatol': 1e-6, 'maxiter': 200})
        if result.success or result.fun < best_side:
            best_side = min(best_side, result.fun)
    
    return best_side

def construct():
    """
    Place 11 unit hexagons inside a regular hexagon, minimizing the outer side length.
    
    Strategy: Start with a hexagonal cluster arrangement (center + 6 around + 4 more),
    then optimize positions via gradient descent to minimize the enclosing hexagon side length.
    """
    np.random.seed(42)
    
    # Initial arrangement: center + 6 in first ring + 4 in second ring (partial)
    # For unit hexagons, the distance between centers of adjacent hexagons is 2*inradius = 2*sqrt(3)/2 = sqrt(3)
    # Actually, for two unit hexagons to be tangent, the distance between centers is 2 * inradius = 2 * (sqrt(3)/2) = sqrt(3)
    
    d = np.sqrt(3)  # distance between centers of adjacent unit hexagons
    
    # Hexagonal grid positions for 11 hexagons
    # Center: (0,0)
    # Ring 1: 6 hexagons at distance d, angles 0, 60, 120, 180, 240, 300
    # Ring 2: 4 hexagons at distance 2d, angles 30, 90, 150, 210 (or similar)
    
    positions = []
    angles = [0.0] * 11
    
    # Center
    positions.append([0.0, 0.0])
    
    # Ring 1: 6 hexagons
    for k in range(6):
        angle = k * np.pi / 3
        x = d * np.cos(angle)
        y = d * np.sin(angle)
        positions.append([x, y])
    
    # Ring 2: 4 hexagons at 2d, at angles 30, 90, 150, 210 degrees
    for k in range(4):
        angle = np.radians(30 + k * 60)
        x = 2 * d * np.cos(angle)
        y = 2 * d * np.sin(angle)
        positions.append([x, y])
    
    positions = np.array(positions)
    
    def get_all_vertices(positions, angles):
        """Get all vertices of all hexagons."""
        all_verts = []
        for i in range(len(positions)):
            cx, cy = positions[i]
            ang = angles[i]
            verts = hex_vertices(cx, cy, 1.0, ang)
            all_verts.append(verts)
        return np.vstack(all_verts)
    
    def objective(params):
        """Objective: minimize the outer hexagon side length."""
        pos = params[:22].reshape(11, 2)
        angs = params[22:]
        
        all_verts = get_all_vertices(pos, angs)
        side = outer_hexagon_side_from_points(all_verts)
        return side
    
    # Initial parameters
    init_params = np.concatenate([positions.flatten(), angles])
    
    # Optimize
    result = minimize(objective, init_params, method='Nelder-Mead',
                      options={'xatol': 0.01, 'fatol': 1e-8, 'maxiter': 5000, 'adaptive': True})
    
    if result.success or result.fun < 100:
        pos = result.x[:22].reshape(11, 2)
        angs = result.x[22:]
    else:
        pos = positions
        angs = angles
    
    # Convert to required format
    inner = []
    for i in range(11):
        inner.append([pos[i][0], pos[i][1], angs[i] * 180.0 / np.pi])
    
    # Find center and side of enclosing hexagon
    all_verts = get_all_vertices(pos, angs)
    center = np.mean(all_verts, axis=0)
    
    # Compute the optimal side length
    side = outer_hexagon_side_from_points(all_verts - center)
    
    # Find the optimal rotation by checking which theta gives the minimum
    def find_best_rotation(points):
        best_side = float('inf')
        best_theta = 0
        for theta in np.linspace(0, 180, 37):
            theta_rad = np.radians(theta)
            normals = []
            for k in range(6):
                normal_angle = theta_rad + np.pi/6 + k * np.pi/3
                normals.append(np.array([np.cos(normal_angle), np.sin(normal_angle)]))
            normals = np.array(normals)
            all_normals = np.vstack([normals, -normals])
            supports = points @ all_normals.T
            max_support = np.max(supports)
            s = 2.0 * max_support / np.sqrt(3.0)
            if s < best_side:
                best_side = s
                best_theta = theta
        return best_side, best_theta
    
    side, rot = find_best_rotation(all_verts - center)
    
    return inner, (float(center[0]), float(center[1])), float(side), float(rot)

# EVOLVE-BLOCK-END
