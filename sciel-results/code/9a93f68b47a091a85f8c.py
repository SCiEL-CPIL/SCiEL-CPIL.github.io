import numpy as np
from scipy.optimize import minimize_scalar

N = 12

# EVOLVE-BLOCK-START
def construct():
    # Define lattice vectors for flat-top hexagons
    # Distance between centers is sqrt(3)
    u = np.array([np.sqrt(3), 0.0])
    v = np.array([np.sqrt(3)/2.0, 1.5])
    
    # Create a 3x4 grid of lattice points
    # This is a compact cluster
    indices = [(i, j) for i in range(3) for j in range(4)]
    
    # Calculate positions
    points = np.array([i * u + j * v for i, j in indices])
    
    # Center the cluster at the origin
    centroid = np.mean(points, axis=0)
    points = points - centroid
    
    # Inner hexagons are all aligned at 0 degrees
    inner_angles = [0.0] * N
    inner_positions = points.tolist()
    
    # Function to calculate required outer side length L for a given outer rotation theta
    def calc_L(theta):
        # theta is in radians
        # Outer hexagon normals are at theta + k * pi/3
        # Inner hexagon support function h_inner(phi)
        # For a unit hexagon (side 1), apothem is sqrt(3)/2
        # h(phi) = apothem / cos(phi - normal_angle_of_inner)
        # Since inner is aligned at 0, its normals are at 0, 60, 120...
        # But we just need the max projection of the inner hexagon in direction phi.
        # The max projection of a regular hexagon centered at origin in direction phi is:
        # h_inner(phi) = (sqrt(3)/2) / cos( (phi % (pi/3)) - pi/6 ) ? 
        # Let's verify.
        # Normal directions for inner hexagon (flat top): 0, 60, 120, 180, 240, 300.
        # Actually, for flat top, vertices are at 0, 60... Normals are at 30, 90...?
        # No, standard flat top: vertices at (1,0), (0.5, sqrt(3)/2)...
        # Normal to edge between (1,0) and (0.5, sqrt(3)/2) is at 30 deg?
        # Edge midpoint: (0.75, sqrt(3)/4). Vector (0.75, 0.433). Angle 30 deg.
        # So normals are at 30, 90, 150...
        # Let's just compute the max dot product of the vertices.
        
        # Vertices of unit hexagon centered at origin, angle 0
        # v_k = (cos(k*pi/3), sin(k*pi/3)) for k=0..5
        # Wait, if angle is 0, is it flat top or pointy top?
        # The task says "angle_degrees". 
        # Usually 0 means one vertex is on the x-axis (pointy top) or one edge is horizontal (flat top).
        # The seed program uses 0.0. 
        # Let's assume standard math convention: 0 is vertex at (1,0).
        # Then vertices are at angles 0, 60, 120, 180, 240, 300.
        # The support function h_inner(phi) is max_k (v_k . n_phi).
        
        verts = np.array([[np.cos(k * np.pi / 3), np.sin(k * np.pi / 3)] for k in range(6)])
        
        max_extent = 0.0
        for m in range(6):
            angle = theta + m * np.pi / 3.0
            n = np.array([np.cos(angle), np.sin(angle)])
            
            # Max projection of any inner hexagon in direction n
            # This is max over centers Q_k of (Q_k . n + max_v (v . n))
            max_inner_support = max(verts.dot(n))
            
            center_projs = points.dot(n)
            max_total = np.max(center_projs) + max_inner_support
            if max_total > max_extent:
                max_extent = max_total
        
        # Outer hexagon apothem A = sqrt(3)/2 * L
        # We need A >= max_extent
        # L >= 2/sqrt(3) * max_extent
        return (2.0 / np.sqrt(3)) * max_extent

    # Minimize L over theta in [0, pi/6) due to symmetry
    result = minimize_scalar(calc_L, bounds=(0, np.pi / 6), method='bounded')
    optimal_theta = result.x
    optimal_L = result.fun
    
    # The outer center is (0,0)
    outer_center = (0.0, 0.0)
    outer_side = optimal_L
    outer_angle = np.degrees(optimal_theta)
    
    return inner_positions, outer_center, outer_side, outer_angle
# EVOLVE-BLOCK-END
