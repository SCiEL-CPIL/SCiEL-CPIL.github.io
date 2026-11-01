"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
import random

N = 12

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140.0

    # Generate the 19-hexagon cluster positions
    # Center
    positions = [(0.0, 0.0)]
    
    # Ring 1 (6 hexagons)
    for i in range(6):
        theta = i * np.pi / 3
        x = 2.0 * np.cos(theta)
        y = 2.0 * np.sin(theta)
        positions.append((x, y))
        
    # Ring 2 (12 hexagons)
    # 6 Corners (dist 4)
    for i in range(6):
        theta = i * np.pi / 3
        x = 4.0 * np.cos(theta)
        y = 4.0 * np.sin(theta)
        positions.append((x, y))
        
    # 6 Edges (dist 2*sqrt(3))
    for i in range(6):
        theta = i * np.pi / 3 + np.pi / 6
        x = 2.0 * np.sqrt(3) * np.cos(theta)
        y = 2.0 * np.sqrt(3) * np.sin(theta)
        positions.append((x, y))
        
    # We have 1 + 6 + 6 + 6 = 19 positions.
    # Indices:
    # 0: Center
    # 1-6: Ring 1
    # 7-12: Ring 2 Corners
    # 13-18: Ring 2 Edges
    
    # We want to keep 12.
    # Keep Center (0) and Ring 1 (1-6).
    # Keep 5 of the Ring 2 Edges (13-18).
    # Omit 1 Edge and all 6 Corners.
    # To minimize bounding box, omitting any edge is symmetric.
    # Let's omit the edge at index 18 (angle 330 deg).
    
    selected_indices = [0] + list(range(1, 7)) + list(range(13, 18))
    
    # Extract coordinates
    inner_coords = [positions[i] for i in selected_indices]
    
    # All inner hexagons have angle 0
    inner = [[x, y, 0.0] for x, y in inner_coords]
    
    # Compute vertices for all inner hexagons
    # Flat-top hexagon vertices relative to center:
    # (1,0), (0.5, sqrt(3)/2), (-0.5, sqrt(3)/2), (-1,0), (-0.5, -sqrt(3)/2), (0.5, -sqrt(3)/2)
    sqrt3 = np.sqrt(3)
    rel_verts = np.array([
        [1.0, 0.0],
        [0.5, sqrt3/2],
        [-0.5, sqrt3/2],
        [-1.0, 0.0],
        [-0.5, -sqrt3/2],
        [0.5, -sqrt3/2]
    ])
    
    all_verts = []
    for x, y, _ in inner:
        verts = rel_verts + np.array([x, y])
        all_verts.append(verts)
        
    all_verts = np.vstack(all_verts)
    
    # Function to compute required side length for a given outer angle
    def compute_side(angle_deg):
        angle_rad = np.radians(angle_deg)
        max_width = 0.0
        # Normals of the outer hexagon
        for k in range(6):
            normal_angle = angle_rad + np.radians(30) + k * np.pi / 3
            normal = np.array([np.cos(normal_angle), np.sin(normal_angle)])
            
            # Project all vertices onto the normal
            projections = np.dot(all_verts, normal)
            width = np.max(projections) - np.min(projections)
            
            if width > max_width:
                max_width = width
                
        # Side length L such that width = L * sqrt(3)
        return max_width / sqrt3
        
    # Scan for optimal angle
    # The configuration has some symmetry, but not full 6-fold due to omitted edge.
    # We can scan 0 to 60 degrees.
    best_side = float('inf')
    best_angle = 0.0
    
    # Coarse scan
    for deg in np.arange(0, 60, 0.5):
        s = compute_side(deg)
        if s < best_side:
            best_side = s
            best_angle = deg
            
    # Fine scan around best_angle
    for deg in np.arange(best_angle - 0.5, best_angle + 0.5, 0.01):
        s = compute_side(deg)
        if s < best_side:
            best_side = s
            best_angle = deg
            
    center = (0.0, 0.0)
    side = best_side
    angle = best_angle
    
    return inner, center, side, angle
# EVOLVE-BLOCK-END
