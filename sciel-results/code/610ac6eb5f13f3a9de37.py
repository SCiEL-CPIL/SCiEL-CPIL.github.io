import numpy as np
import time

N = 12

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    # Define the 3x4 hexagonal lattice cluster
    # Hexagonal lattice vectors for unit hexagons:
    # Nearest neighbors are at distance sqrt(3) * side? No, distance between centers is sqrt(3) for side 1?
    # Distance between centers of touching unit hexagons is 2 * apothem = 2 * (sqrt(3)/2) = sqrt(3).
    # Let a = sqrt(3).
    a = np.sqrt(3)
    
    # Lattice basis vectors
    # u direction: (a, 0)
    # v direction: (a/2, a*sqrt(3)/2) = (sqrt(3)/2, 3/2)
    u = np.array([a, 0.0])
    v = np.array([a / 2.0, a * np.sqrt(3) / 2.0])
    
    # Generate positions for 3x4 grid (3 along u, 4 along v)
    # Let i be 0..2, j be 0..3
    points = []
    for i in range(3):
        for j in range(4):
            p = i * u + j * v
            points.append(p)
            
    points = np.array(points)
    
    # Center the cluster
    centroid = np.mean(points, axis=0)
    points = points - centroid
    
    # Define the 6 normal directions for a regular hexagon
    # Angles: 0, 30, 60, 90, 120, 150 degrees
    angles_deg = [0, 30, 60, 90, 120, 150]
    normals_base = np.array([[np.cos(np.deg2rad(a)), np.sin(np.deg2rad(a))] for a in angles_deg])
    
    best_S = float('inf')
    best_theta = 0.0
    best_points = points.copy()
    
    # Coarse search for optimal rotation
    # Symmetry of hexagon is 60 degrees, but the cluster might have different symmetry.
    # However, the container is symmetric every 60 degrees.
    # We can search 0 to 30 degrees.
    thetas_coarse = np.linspace(0, np.pi / 6, 100)
    
    for theta in thetas_coarse:
        # Check deadline
        if time.time() > deadline:
            break
            
        # Rotation matrix
        c, s = np.cos(theta), np.sin(theta)
        R = np.array([[c, -s], [s, c]])
        
        # Rotate points
        pts_rot = points @ R.T
        
        # Calculate required side length S
        # S = max_k max_i |p_i dot n_k| * (2/sqrt(3))
        # Normals are fixed in space relative to the container.
        # Since we are rotating the points, we are effectively changing the relative angle.
        # Alternatively, rotate the normals. It's equivalent.
        # Let's rotate the points. The container normals remain fixed.
        # Wait, the container can also rotate? No, the score is invariant to global rotation.
        # We can fix the container orientation and rotate the cluster.
        
        # Project rotated points onto fixed normals
        # normals_base are unit vectors.
        projections = np.abs(pts_rot @ normals_base.T)
        max_proj = np.max(projections)
        
        # The distance from center to side of regular hexagon is S * sqrt(3) / 2
        # So max_proj <= S * sqrt(3) / 2
        S_candidate = max_proj * 2.0 / np.sqrt(3)
        
        if S_candidate < best_S:
            best_S = S_candidate
            best_theta = theta
            best_points = pts_rot
            
    # Fine search around best_theta
    if best_theta > 0 and best_theta < np.pi / 6:
        thetas_fine = np.linspace(best_theta - 0.05, best_theta + 0.05, 100)
        thetas_fine = thetas_fine[(thetas_fine >= 0) & (thetas_fine <= np.pi / 6)]
        for theta in thetas_fine:
             if time.time() > deadline:
                break
             c, s = np.cos(theta), np.sin(theta)
             R = np.array([[c, -s], [s, c]])
             pts_rot = points @ R.T
             projections = np.abs(pts_rot @ normals_base.T)
             max_proj = np.max(projections)
             S_candidate = max_proj * 2.0 / np.sqrt(3)
             if S_candidate < best_S:
                 best_S = S_candidate
                 best_theta = theta
                 best_points = pts_rot

    # Construct the return values
    # inner: list of [x, y, angle_degrees]
    # The angle of the inner hexagons:
    # Since the cluster was built on a standard lattice, the hexagons are aligned with the lattice.
    # In our rotated coordinate system (pts_rot), the hexagons are also rotated by best_theta.
    # So their angle is best_theta in degrees.
    angle_deg = np.rad2deg(best_theta)
    
    inner = []
    for p in best_points:
        inner.append([p[0], p[1], angle_deg])
        
    # Outer hexagon center, side, angle
    # Center is (0,0) because we centered the points.
    # Side length is best_S.
    # Angle of outer hexagon: We assumed normals were at 0, 30, ...
    # This corresponds to a hexagon with a flat top/bottom?
    # Normal at 0 deg is (1,0). Side is vertical.
    # This is a standard orientation.
    # The return value asks for outer hexagon rotation.
    # Since we fixed the container normals to 0, 30..., the container angle is 0.
    center = (0.0, 0.0)
    side = float(best_S)
    angle = 0.0
    
    return inner, center, side, angle
# EVOLVE-BLOCK-END
