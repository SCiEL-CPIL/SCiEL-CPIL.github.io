# EVOLVE-BLOCK-START
import numpy as np
import time

def hexagon_packing_11():
    """
    Constructs a packing of 11 disjoint unit regular hexagons inside a larger regular hexagon, maximizing 1/outer_hex_side_length.
    Returns
        inner_hex_data: np.ndarray of shape (11,3), where each row is of the form (x, y, angle_degrees) containing the (x,y) coordinates and angle_degree of the respective inner hexagon.
        outer_hex_data: np.ndarray of shape (3,) of form (x,y,angle_degree) containing the (x,y) coordinates and angle_degree of the respective outer hexagon.
        outer_hex_side_length: float representing the side length of the outer hexagon.
    """
    start_time = time.time()
    deadline = start_time + 600
    
    # Scale factor for lattice spacing to ensure separation > 1e-6
    # Distance between centers in honeycomb is sqrt(3).
    # We use spacing = sqrt(3) * (1 + eps)
    eps = 1e-4
    s = np.sqrt(3) * (1 + eps)
    
    def generate_shape_flower_plus_4():
        # Center
        pts = [(0, 0)]
        # Ring 1
        for i in range(6):
            angle = np.pi/3 * i
            pts.append((s * np.cos(angle), s * np.sin(angle)))
        # Extensions on 4 arms (0, 60, 120, 180)
        for i in [0, 1, 2, 3]:
            angle = np.pi/3 * i
            pts.append((2 * s * np.cos(angle), 2 * s * np.sin(angle)))
        return np.array(pts)

    def generate_shape_rhombus_plus_2():
        pts = []
        # Row 0
        for i in range(3):
            pts.append((i * s, 0))
        # Row 1
        for i in range(3):
            pts.append((i * s + s/2, 1.5 * s))
        # Row 2
        for i in range(3):
            pts.append((i * s, 3 * s))
        # Add 2 more to Row -1
        for i in range(2):
            pts.append((i * s + s/2, -1.5 * s))
        return np.array(pts)

    shapes = [generate_shape_flower_plus_4(), generate_shape_rhombus_plus_2()]
    
    best_S = float('inf')
    best_angle = 0
    best_shape_idx = 0
    best_center = np.array([0,0])

    # Scan angles from 0 to 60 degrees
    # Step 0.1 degrees
    angles = np.arange(0, 60.1, 0.1)
    
    for idx, shape in enumerate(shapes):
        # Center the shape at origin to simplify rotation/centering logic
        centroid = np.mean(shape, axis=0)
        centered_shape = shape - centroid
        
        for theta_deg in angles:
            if time.time() > deadline - 10:
                break
            theta = np.deg2rad(theta_deg)
            
            # Normals of the outer hexagon
            # Vertices at theta + k*pi/3
            # Normals are at theta + pi/6 + k*pi/3
            normals = []
            for k in range(6):
                n_angle = theta + np.pi/6 + k * np.pi/3
                normals.append(np.array([np.cos(n_angle), np.sin(n_angle)]))
            normals = np.array(normals)
            
            max_proj = 0
            # For each inner hexagon, check all its vertices
            for cx, cy in centered_shape:
                # Inner hexagon is aligned with angle 0 (vertices at 0, 60, ...)
                # But we can also rotate inner hexagons? 
                # The problem allows angles. 
                # However, for a grid packing, aligning them is usually optimal for bounding box.
                # Let's assume inner hexagons are aligned with angle 0 relative to global frame.
                # Actually, the "angle_deg" in output is for the inner hexagon.
                # If we rotate the outer hexagon, the inner ones stay fixed in orientation?
                # Yes, we can output any angle for inner ones.
                # But the relative orientation matters for the bounding box.
                # If we align inner hexagons with the global axes, their vertices are fixed.
                
                # Vertices of inner hexagon at (cx, cy) with angle 0
                for j in range(6):
                    v_angle = np.pi/3 * j
                    vx = cx + np.cos(v_angle)
                    vy = cy + np.sin(v_angle)
                    p = np.array([vx, vy])
                    
                    # Project onto all 6 normals
                    projs = np.dot(normals, p)
                    # We need max absolute projection? 
                    # The hexagon is centered at origin. 
                    # Condition: |p . n| <= S * sqrt(3)/2
                    # So we need max |p . n|
                    max_abs_proj = np.max(np.abs(projs))
                    if max_abs_proj > max_proj:
                        max_proj = max_abs_proj
            
            S_candidate = (2 / np.sqrt(3)) * max_proj
            
            if S_candidate < best_S:
                best_S = S_candidate
                best_angle = theta_deg
                best_shape_idx = idx
                best_center = centroid # We will shift back

    # Construct final result
    shape = shapes[best_shape_idx]
    # The shape was centered at origin for calculation. 
    # We need to output coordinates relative to the outer hexagon center.
    # The outer hexagon center is at (0,0) in the final output?
    # The task says: "Coordinates are first shifted by the outer center".
    # So we can place the outer hexagon at (0,0) and the inner ones at their relative positions.
    
    # The 'centered_shape' was shape - centroid.
    # So if we output centered_shape, the outer hexagon is at (0,0).
    # This is valid.
    
    inner_hex_data = np.zeros((11, 3))
    for i, (cx, cy) in enumerate(shape - np.mean(shape, axis=0)):
        inner_hex_data[i, 0] = cx
        inner_hex_data[i, 1] = cy
        inner_hex_data[i, 2] = 0.0 # Angle of inner hexagon
        
    outer_hex_data = np.array([0.0, 0.0, best_angle])
    
    return inner_hex_data, outer_hex_data, best_S

# EVOLVE-BLOCK-END
