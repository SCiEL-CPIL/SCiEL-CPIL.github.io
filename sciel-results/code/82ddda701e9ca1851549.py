import numpy as np

N = 12

# EVOLVE-BLOCK-START
def construct():
    # Generate 12 unit hexagons in a compact lattice arrangement
    # Center + 6 neighbors + 5 from the next ring (avoiding corners to minimize bounding box)
    
    # Axial coordinates (q, r)
    # Center
    points = [(0, 0)]
    # Ring 1
    points += [(1, 0), (1, -1), (0, -1), (-1, 0), (-1, 1), (0, 1)]
    # Ring 2 - Select 5 "side" points to keep the shape compact and avoid the far corners
    # The 12 points in Ring 2 are:
    # (2,0), (2,-1), (2,-2), (1,-2), (0,-2), (-1,-1), (-2,0), (-2,1), (-2,2), (-1,2), (0,2), (1,1)
    # Corners are (2,0), (0,-2), (-2,0), (-2,2), (0,2), (2,-2) ? 
    # Let's pick points that are not at the extreme axial distance 2 in the primary directions.
    # (1, -1) is already in Ring 1? No, (1,-1) is Ring 1.
    # Ring 2 points with axial norm < 2?
    # (1, 1) -> dist sqrt(3)
    # (1, -2) -> dist 2?
    # Let's just pick 5 points from the 12 and calculate the bounding box.
    
    # To minimize the max projection, we should pick points that are "inward" relative to the 6 main axes.
    # The main axes correspond to the 6 corners of the hexagon.
    # We will pick the 5 points from Ring 2 that are adjacent to the corners but not the corners themselves,
    # or just a contiguous block.
    
    # Let's try a specific set:
    ring2_candidates = [
        (2, -1), (1, -2), (0, -2), (-1, -1), # Wait, (-1,-1) is Ring 1? No, (-1,-1) is dist sqrt(3).
        # Let's verify axial distance.
        # dist(q,r) = (|q| + |r| + |q+r|) / 2
        # (1,0) -> (1+0+1)/2 = 1.
        # (1,1) -> (1+1+2)/2 = 2.
        # So (1,1) is Ring 2.
        
        # Let's pick 5 points from Ring 2:
        (1, 1), (0, 2), (-1, 2), (-2, 1), (-2, 0)
    ]
    
    # Actually, to be symmetric and minimal, let's just take the first 5 of a sorted list by some metric.
    # Or simpler: Use the 7-hex cluster and add 5 hexes in the "gaps" if possible? No, gaps are too small.
    # So we must expand.
    
    # Let's use the 5 points: (1,1), (0,2), (-1,2), (-2,1), (-2,0) is a contiguous arc.
    # This might create a flat side, which is good for a hexagon container.
    
    final_points = points + [(1, 1), (0, 2), (-1, 2), (-2, 1), (-2, 0)]
    
    # Convert to Cartesian (Pointy Top)
    # x = sqrt(3) * (q + r/2)
    # y = 1.5 * r
    centers = []
    for q, r in final_points:
        x = np.sqrt(3) * (q + r / 2.0)
        y = 1.5 * r
        centers.append((x, y))
    
    # Calculate minimal enclosing regular hexagon side length S
    # The hexagon is centered at (0,0) to minimize S for this symmetric-ish cluster?
    # We should center the cluster at the origin for the calculation.
    # The current points are centered around (0,0) roughly.
    # Let's find the centroid and shift?
    # For a regular hexagon container, centering the packing at the container center is optimal.
    
    # Calculate max projection on 6 normals
    # Normals are at angles 0, 60, 120, 180, 240, 300 degrees
    # Inradius of unit hex is sqrt(3)/2
    # S * sqrt(3)/2 >= max(proj) + sqrt(3)/2
    # S >= 2/sqrt(3) * max(proj) + 1
    
    max_proj = 0
    for i in range(6):
        angle = np.deg2rad(60 * i)
        nx = np.cos(angle)
        ny = np.sin(angle)
        for cx, cy in centers:
            proj = cx * nx + cy * ny
            if proj > max_proj:
                max_proj = proj
                
    S = (2.0 / np.sqrt(3)) * max_proj + 1.0
    
    # Add a small epsilon to ensure it's strictly inside due to floating point
    S += 1e-6
    
    # Construct inner list
    # Format: (x, y, angle_degrees)
    # All angles 0
    inner = [[x, y, 0.0] for x, y in centers]
    
    # Outer hexagon
    # Center (0,0), Side S, Angle 0
    return inner, (0.0, 0.0), S, 0.0
# EVOLVE-BLOCK-END
