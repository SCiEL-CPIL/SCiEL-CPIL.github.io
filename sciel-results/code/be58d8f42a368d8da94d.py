"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

N = 11
DEADLINE = 145.0  # seconds, safety margin for 150s limit

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    
    # Generate dense hexagonal cluster layout for 11 hexagons
    # Center at origin, neighbors at distance sqrt(3)
    s3 = np.sqrt(3.0)
    
    # 7-hexagon cluster (1 center + 6 neighbors)
    centers = [(0.0, 0.0)]
    for k in range(6):
        angle = np.radians(60.0 * k)
        centers.append((s3 * np.cos(angle), s3 * np.sin(angle)))
    
    # Add 4 more hexagons extending in 4 directions
    # Extend at angles 0, 90, 180, 270 degrees from center
    for angle_deg in [0, 90, 180, 270]:
        angle = np.radians(angle_deg)
        # Place at distance 2*sqrt(3) from center in that direction
        centers.append((2 * s3 * np.cos(angle), 2 * s3 * np.sin(angle)))
    
    # All 11 hexagons have angle 0 (flat-top orientation)
    inner = [[x, y, 0.0] for x, y in centers]
    
    # Compute optimal container size and rotation
    # Vertices of a unit hexagon with flat-top orientation (angle=0)
    hex_vertices = []
    for k in range(6):
        angle = np.radians(60.0 * k)
        hex_vertices.append((np.cos(angle), np.sin(angle)))
    hex_vertices = np.array(hex_vertices)
    
    # All vertices of all inner hexagons
    all_vertices = []
    for cx, cy, _ in inner:
        for vx, vy in hex_vertices:
            all_vertices.append((cx + vx, cy + vy))
    all_vertices = np.array(all_vertices)
    
    # Find minimum enclosing regular hexagon
    best_side = float('inf')
    best_angle = 0.0
    
    # Try container rotations from 0 to 60 degrees
    # The container is a regular hexagon, so rotation symmetry is 60 degrees
    n_rotations = 360
    for i in range(n_rotations):
        theta = np.radians(60.0 * i / n_rotations)
        
        # Rotation matrix for container
        c = np.cos(theta)
        s = np.sin(theta)
        
        # Project vertices onto 6 normals of the container
        # Container normals are at angles theta, theta+60, theta+120, ...
        min_dist = 0.0
        for k in range(6):
            normal_angle = theta + np.radians(60.0 * k)
            nx = np.cos(normal_angle)
            ny = np.sin(normal_angle)
            
            # Project all vertices onto this normal
            projections = all_vertices[:, 0] * nx + all_vertices[:, 1] * ny
            max_proj = np.max(projections)
            min_dist = max(min_dist, max_proj)
        
        # The side length S relates to the apothem a by: a = S * sqrt(3)/2
        # So S = 2*a/sqrt(3)
        side = 2.0 * min_dist / np.sqrt(3.0)
        
        if side < best_side:
            best_side = side
            best_angle = np.degrees(theta)
        
        # Check deadline
        if time.time() - start_time > DEADLINE:
            break
    
    return inner, (0.0, 0.0), best_side, best_angle
# EVOLVE-BLOCK-END
