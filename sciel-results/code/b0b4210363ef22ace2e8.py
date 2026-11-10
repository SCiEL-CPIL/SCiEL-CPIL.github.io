import numpy as np
import time

def _hex_vertices(cx, cy, d, a_deg):
    a = np.deg2rad(a_deg)
    angles = a + np.deg2rad(60 * np.arange(6))
    return np.array([cx + d * np.cos(angles), cy + d * np.sin(angles)]).T

def hexagon_packing_12():
    # Deadline guard
    start_time = time.time()
    
    # Generate 12 hexagon centers in a 3x4 hex grid
    # Flat-top hexagons
    # dx = sqrt(3), dy = 1.5
    # Offset for odd rows: sqrt(3)/2
    
    centers = []
    scale = 1.001 # Small gap to satisfy separation constraint
    for r in range(3):
        for c in range(4):
            x = scale * (c * np.sqrt(3) + (r % 2) * (np.sqrt(3) / 2))
            y = scale * (r * 1.5)
            centers.append((x, y))
            
    centers = np.array(centers)
    
    # Center the cluster at origin
    mean_center = np.mean(centers, axis=0)
    centers = centers - mean_center
    
    # The inner hexagons have side 1, angle 30 (flat-side right/left, pointy top/bottom)
    # This orientation allows the standard honeycomb grid to be non-overlapping.
    # We need to find the minimum side s of the outer hexagon
    # Outer hexagon center is (0,0)
    # We can rotate the outer hexagon.
    
    min_s = np.inf
    best_angle = 0
    
    # The inner hexagons have fixed orientation (angle 0).
    # We vary the outer hexagon angle from 0 to 30 degrees.
    for outer_angle_deg in np.linspace(0, 30, 100):
        # Normals of the outer hexagon edges
        # Outer hexagon vertices are at angles outer_angle + 60*i
        # Edge normals are at angles outer_angle + 30 + 60*i
        normal_angles = np.deg2rad(outer_angle_deg + 30 + 60 * np.arange(6))
        normals = np.array([np.cos(normal_angles), np.sin(normal_angles)]).T
        
        # Get all vertices of all 12 inner hexagons
        all_vertices = []
        for cx, cy in centers:
            verts = _hex_vertices(cx, cy, 1.0, 30.0)
            all_vertices.append(verts)
        
        all_vertices = np.vstack(all_vertices)
        
        # Project all vertices onto all normals
        # proj shape: (72, 6)
        # all_vertices is (72, 2), normals is (6, 2). We need (72, 6) = (72, 2) @ (2, 6)
        projections = all_vertices @ normals.T
        
        # Max projection for each normal
        max_projs = np.max(projections, axis=0)
        
        # The required apothem R is max(max_projs)
        # R = s * sqrt(3) / 2
        # s = 2 * R / sqrt(3)
        R = np.max(max_projs)
        s = 2 * R / np.sqrt(3)
        
        if s < min_s:
            min_s = s
            best_angle = outer_angle_deg
            
        if time.time() - start_time > 590:
            break

    # Construct the result
    # Inner: (12, 3) array of (x, y, angle_deg)
    inner = np.zeros((12, 3))
    for i, (cx, cy) in enumerate(centers):
        inner[i] = [cx, cy, 30.0]
        
    # Outer: (3,) array (x, y, angle_deg)
    # Outer center is (0,0)
    outer = np.array([0.0, 0.0, best_angle])
    
    return inner, outer, min_s
