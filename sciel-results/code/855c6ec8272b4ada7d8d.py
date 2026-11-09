import numpy as np
import math

N = 11

# EVOLVE-BLOCK-START
def construct():
    # Define the 11 inner hexagon centers in a compact cluster
    # Using a "flower" of 7 hexagons plus 4 on the perimeter
    # Add a small epsilon to ensure strict disjointness if required
    eps = 1e-6
    d = math.sqrt(3) + eps # Distance between centers of adjacent unit hexagons
    
    centers = []
    # Central 7
    centers.append((0.0, 0.0))
    centers.append((d, 0.0))
    centers.append((d/2, d*math.sqrt(3)/2))
    centers.append((-d/2, d*math.sqrt(3)/2))
    centers.append((-d, 0.0))
    centers.append((-d/2, -d*math.sqrt(3)/2))
    centers.append((d/2, -d*math.sqrt(3)/2))
    
    # Additional 4 to make 11
    # Placing them to maintain compactness
    centers.append((1.5*d, d*math.sqrt(3)/2))
    centers.append((1.5*d, -d*math.sqrt(3)/2))
    centers.append((0.0, d*math.sqrt(3)))
    centers.append((0.0, -d*math.sqrt(3)))
    
    # Generate vertices for each inner hexagon
    # Unit hexagon vertices relative to center
    # Orientation: Vertex at 0 degrees (pointy right)
    # This makes the "width" (flat-to-flat) along the y-axis? 
    # No, if vertex is at 0, the flat sides are at 30, 90, etc.
    # Let's stick to standard: vertices at 0, 60, 120, 180, 240, 300.
    
    hex_vertices_rel = []
    for k in range(6):
        angle = math.radians(60 * k)
        hex_vertices_rel.append((math.cos(angle), math.sin(angle)))
    
    all_vertices = []
    for cx, cy in centers:
        for vx, vy in hex_vertices_rel:
            all_vertices.append((cx + vx, cy + vy))
    
    all_vertices = np.array(all_vertices)
    
    # Find the minimal enclosing regular hexagon
    # Search over the rotation angle of the outer hexagon
    # Assume center is the centroid of the inner hexagons' centers
    centroid = np.mean(centers, axis=0)
    
    best_side = float('inf')
    best_angle = 0.0
    
    # Search rotation angle from 0 to 60 degrees
    # Step size 0.5 degrees
    for deg in np.arange(0, 60, 0.5):
        theta = math.radians(deg)
        
        # Normals to the faces of the outer hexagon
        # If the hexagon is rotated by theta, its vertices are at theta + k*60.
        # The face normals are at theta + 30 + k*60.
        normals = []
        for k in range(6):
            n_angle = theta + math.radians(30 + 60 * k)
            normals.append((math.cos(n_angle), math.sin(n_angle)))
        normals = np.array(normals)
        
        # Project all vertices onto the normals
        # We want the max projection for each normal relative to the centroid
        rel_vertices = all_vertices - centroid
        projections = rel_vertices @ normals.T
        max_projections = np.max(projections, axis=0)
        
        # The required inradius is the max of the max projections
        r = np.max(max_projections)
        
        if r < 0:
            r = 0
            
        side = 2 * r / math.sqrt(3)
        
        if side < best_side:
            best_side = side
            best_angle = deg
            
    # Return the result
    inner = [[cx, cy, 0.0] for cx, cy in centers]
    center = [centroid[0], centroid[1]]
    side = best_side
    angle = best_angle
    
    return inner, center, side, angle
# EVOLVE-BLOCK-END
