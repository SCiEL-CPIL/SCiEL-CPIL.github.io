import numpy as np

def construct():
    N = 11
    # Define a few candidate configurations of 11 unit hexagons
    # Coordinates are (x, y, angle)
    # We assume all inner hexagons are aligned (angle 0) for simplicity, 
    # as rotating them usually doesn't reduce the bounding box for a hexagonal container 
    # significantly compared to just placing them well, and the container rotation 
    # handles the alignment.
    
    # Helper to generate points in a triangular lattice
    # x = sqrt(3) * (i + j/2)
    # y = 1.5 * j
    def lattice_point(i, j):
        return (np.sqrt(3) * (i + j / 2.0), 1.5 * j, 0.0)

    # Configuration 1: Central flower (7) + 4 extending
    # Flower: (0,0) and 6 neighbors
    # Neighbors in lattice coords (i,j):
    # (1,0), (-1,0), (0,1), (0,-1), (1,-1), (-1,1)
    pts1 = [
        lattice_point(0,0),
        lattice_point(1,0), lattice_point(-1,0),
        lattice_point(0,1), lattice_point(0,-1),
        lattice_point(1,-1), lattice_point(-1,1)
    ]
    # Add 4 more. Let's try extending the right side.
    # (2,0), (2,1), (2,-1), (3,0)?
    # Let's try a compact shape.
    # How about (1,1), (-1,-1), (1,-1 is already there), (-1,1 is already there).
    # The 7 are:
    # (0,0)
    # (1,0), (-1,0)
    # (0,1), (0,-1)
    # (1,-1), (-1,1)
    # Missing from the 19-hexagon (side 3) are:
    # (2,0), (2,1), (2,-1), (1,2), (1,-2), (0,2), (0,-2), (-1,2), (-1,-2), (-2,0), (-2,1), (-2,-1)
    # Let's pick 4 that keep it compact.
    # (2,0), (2,1), (2,-1) and (1,2)?
    # Let's try: (2,0), (1,1), (-1,-1), (0,2)?
    # Let's try a symmetric addition.
    # Add (2,0), (-2,0), (1,1), (-1,-1)?
    pts1.extend([
        lattice_point(2,0),
        lattice_point(-2,0),
        lattice_point(1,1),
        lattice_point(-1,-1)
    ])
    
    # Configuration 2: Triangular 1,2,3,5? No.
    # Let's try a 3x3 block in lattice coords: i in [-1,1], j in [-1,1] -> 9 points.
    # Add 2 more.
    pts2 = [lattice_point(i, j) for i in range(-1, 2) for j in range(-1, 2)]
    # Add (2,0) and (-2,0)
    pts2.extend([
        lattice_point(2,0),
        lattice_point(-2,0)
    ])
    
    # Configuration 3: A more circular cluster.
    # 7 center.
    # Add (2,0), (2,1), (2,-1), (1,2)?
    pts3 = [
        lattice_point(0,0),
        lattice_point(1,0), lattice_point(-1,0),
        lattice_point(0,1), lattice_point(0,-1),
        lattice_point(1,-1), lattice_point(-1,1)
    ]
    pts3.extend([
        lattice_point(2,0),
        lattice_point(2,1),
        lattice_point(2,-1),
        lattice_point(1,2)
    ])

    candidates = [pts1, pts2, pts3]
    
    best_score = float('inf')
    best_config = None
    best_center = (0.0, 0.0)
    best_side = 0.0
    best_angle = 0.0

    for pts in candidates:
        # Convert to numpy array of (x,y)
        coords = np.array([p[:2] for p in pts])
        # Center at centroid
        center = np.mean(coords, axis=0)
        centered_coords = coords - center
        
        # Find optimal rotation theta for the outer hexagon
        # Normals of a regular hexagon with rotation theta:
        # n_k(theta) = (cos(theta + k*pi/3), sin(theta + k*pi/3))
        # We want to minimize max_{p, k} | p . n_k(theta) |
        # Let f(theta) = max_{p, k} | centered_coords @ n_k(theta) |
        # S = 2/sqrt(3) * f(theta)
        
        min_f = float('inf')
        best_theta = 0.0
        
        # Sample theta from 0 to 60 degrees
        # The function is periodic with period 60 degrees.
        # We can sample finely.
        thetas = np.linspace(0, np.pi/3, 181) # 0.5 deg steps
        
        for theta in thetas:
            # Calculate projections
            # n_k for k=0..5
            # We can vectorize this
            # angles = theta + np.arange(6) * np.pi/3
            # cos_angles = np.cos(angles)
            # sin_angles = np.sin(angles)
            # projections = np.abs(centered_coords[:, 0] * cos_angles[None, :] + centered_coords[:, 1] * sin_angles[None, :])
            # max_proj = np.max(projections)
            
            # Optimization: only need to check 3 normals? No, 6.
            # But due to symmetry, checking 0, 30, 60 might be enough? No.
            # Let's just compute.
            angles = theta + np.arange(6) * (np.pi / 3)
            cos_a = np.cos(angles)
            sin_a = np.sin(angles)
            
            # centered_coords is (11, 2)
            # cos_a is (6,)
            # proj is (11, 6)
            proj = np.abs(centered_coords[:, 0] * cos_a[None, :] + centered_coords[:, 1] * sin_a[None, :])
            max_proj = np.max(proj)
            
            if max_proj < min_f:
                min_f = max_proj
                best_theta = theta
        
        side_length = (2.0 / np.sqrt(3)) * min_f
        
        if side_length < best_score:
            best_score = side_length
            best_config = pts
            best_center = tuple(center)
            best_angle = best_theta # The outer hexagon rotation

    # Return the best configuration
    # inner: list of [x, y, angle_degrees]
    inner = [[p[0], p[1], p[2]] for p in best_config]
    # center: (x, y)
    center = best_center
    # side: side length
    side = best_score
    # angle: outer hexagon rotation in degrees? The prompt says "rotation".
    # Usually, the angle of the outer hexagon is what matters.
    # If the evaluator expects the angle of the outer hexagon in degrees:
    outer_angle_deg = np.degrees(best_angle)
    
    return inner, center, side, outer_angle_deg

