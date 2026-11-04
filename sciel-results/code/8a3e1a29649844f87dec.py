"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
from scipy.optimize import minimize

N = 12

# EVOLVE-BLOCK-START

def hex_vertices(cx, cy, side, angle_deg):
    """Return vertices of a regular hexagon centered at (cx, cy) with given side length and rotation."""
    angle_rad = np.radians(angle_deg)
    angles = angle_rad + np.pi / 6 + np.arange(6) * np.pi / 3
    return np.column_stack([cx + side * np.cos(angles), cy + side * np.sin(angles)])

def min_enclosing_hexagon_side(points, center, angle_deg):
    """
    Calculate the minimum side length of a regular hexagon with given center and angle
    that contains all points.
    points: Nx2 array of points.
    center: (cx, cy)
    angle_deg: rotation of the hexagon.
    """
    cx, cy = center
    dx = points[:, 0] - cx
    dy = points[:, 1] - cy
    
    angle_rad = np.radians(angle_deg)
    # Normals to the sides of the hexagon
    # For a hexagon with apothem a, the half-planes are:
    # n_k . p <= a
    # where n_k are unit vectors at angles angle_rad + k * pi/3
    # Wait, the vertices are at angle_rad + pi/6 + k*pi/3.
    # The sides are perpendicular to the radii to the vertices? No.
    # The apothem is perpendicular to the side.
    # The angle of the apothem to side k is angle_rad + k * pi/3.
    # Let's verify.
    # If angle_rad = 0, vertices are at 30, 90, 150, 210, 270, 330.
    # Side 1 is between 30 and 90. Midpoint angle is 60.
    # So the normal to the side is at 60 degrees.
    # So normals are at angle_rad + pi/6 + k * pi/3?
    # Let's check: angle_rad=0. Normals at 30, 90, 150...
    # If normal is at 30 deg, the side is perpendicular to it, so at 120 deg?
    # Let's use the property:
    # A regular hexagon centered at origin with one vertex at (s, 0) has vertices:
    # (s, 0), (s/2, s*sqrt(3)/2), ...
    # The side connecting (s,0) and (s/2, s*sqrt(3)/2) has midpoint (3s/4, s*sqrt(3)/4).
    # Vector to midpoint: (3, sqrt(3)) -> angle 30 deg.
    # So the normal to that side is at 30 deg.
    # So if we rotate the hexagon by angle_rad, the normals are at angle_rad + 30 deg + k*60 deg.
    # i.e., angle_rad + pi/6 + k*pi/3.
    
    normals_angles = angle_rad + np.pi/6 + np.arange(6) * np.pi/3
    cos_n = np.cos(normals_angles)
    sin_n = np.sin(normals_angles)
    
    # For each normal, compute the max projection of points
    # proj_i_k = dx_i * cos_n_k + dy_i * sin_n_k
    # We need a >= max(proj_i_k) for all i, k
    # a is the apothem.
    
    # Matrix multiplication: (N, 6)
    projs = dx[:, np.newaxis] * cos_n[np.newaxis, :] + dy[:, np.newaxis] * sin_n[np.newaxis, :]
    max_proj = np.max(projs)
    
    # The apothem a must be at least max_proj.
    # Side length s = 2 * a / sqrt(3)
    a = max_proj
    if a < 0:
        a = 0
    return 2 * a / np.sqrt(3)

def objective(params):
    """
    params: [x1, y1, a1, ..., x12, y12, a12, outer_angle]
    """
    inner_positions = []
    for i in range(N):
        x, y, a = params[3*i], params[3*i+1], params[3*i+2]
        inner_positions.append((x, y, a))
    
    outer_angle = params[3*N]
    
    # Collect all vertices of inner hexagons
    all_vertices = []
    for i in range(N):
        x, y, a = inner_positions[i]
        verts = hex_vertices(x, y, 1.0, a)
        all_vertices.append(verts)
    all_vertices = np.vstack(all_vertices)
    
    # Calculate minimum outer hexagon side length
    # Center is fixed at (0,0) for symmetry
    side_len = min_enclosing_hexagon_side(all_vertices, (0, 0), outer_angle)
    
    # Calculate overlap penalty
    # Centers of inner hexagons
    centers = np.array([[p[0], p[1]] for p in inner_positions])
    penalty = 0.0
    min_dist = np.sqrt(3) # Minimum distance between centers of disjoint unit hexagons
    
    for i in range(N):
        for j in range(i+1, N):
            dx = centers[i, 0] - centers[j, 0]
            dy = centers[i, 1] - centers[j, 1]
            dist = np.sqrt(dx*dx + dy*dy)
            if dist < min_dist:
                penalty += (min_dist - dist) ** 2
                
    return side_len + 10.0 * penalty

def construct():
    """Construct 12 unit hexagons inside a minimal outer hexagon."""
    # Initial guess: Hexagonal cluster
    # Center at (0,0)
    # Ring 1: 6 hexagons at distance sqrt(3)
    # Ring 2: 6 hexagons at distance 2*sqrt(3)? No, that's too far.
    # Let's use axial coordinates.
    # (0,0)
    # (1,0), (0,1), (-1,1), (-1,0), (0,-1), (1,-1) -> 7 hexagons
    # We need 12.
    # Add 5 more from the next ring.
    # Next ring positions: (2,0), (1,1), (0,2), (-1,2), (-2,1), (-2,0), (-1,-1), (0,-2), (1,-2), (2,-1)
    # Let's pick 5 of these to make it compact.
    # Symmetric choice: (2,0), (0,2), (-2,1), (-2,0), (0,-2)?
    
    sqrt3 = np.sqrt(3)
    # Axial to Cartesian:
    # x = 1.5 * q
    # y = sqrt(3) / 2 * q + sqrt(3) * r
    
    def axial_to_cart(q, r):
        x = 1.5 * q
        y = sqrt3 / 2 * q + sqrt3 * r
        return x, y

    hex_coords = [
        (0, 0),
        (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
        (2, 0), (1, 1), (0, 2), (-1, 2), (-2, 1)
    ]
    
    # Total 12
    initial_positions = []
    for q, r in hex_coords:
        x, y = axial_to_cart(q, r)
        initial_positions.append([x, y, 0.0])
        
    initial_params = []
    for x, y, a in initial_positions:
        initial_params.extend([x, y, a])
    initial_params.append(0.0) # outer angle
    
    # Optimize
    res = minimize(objective, initial_params, method='L-BFGS-B', options={'maxiter': 1000, 'ftol': 1e-6})
    
    optimal_params = res.x
    
    # Extract results
    inner = []
    for i in range(N):
        x, y, a = optimal_params[3*i], optimal_params[3*i+1], optimal_params[3*i+2]
        inner.append([x, y, a])
        
    outer_angle = optimal_params[3*N]
    
    # Recalculate side length for reporting
    all_vertices = []
    for i in range(N):
        x, y, a = inner[i]
        verts = hex_vertices(x, y, 1.0, a)
        all_vertices.append(verts)
    all_vertices = np.vstack(all_vertices)
    
    side_len = min_enclosing_hexagon_side(all_vertices, (0, 0), outer_angle)
    
    return inner, (0.0, 0.0), side_len, outer_angle

# EVOLVE-BLOCK-END
