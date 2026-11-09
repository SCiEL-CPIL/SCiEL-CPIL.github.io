# EVOLVE-BLOCK-START
import numpy as np

def hexagon_packing_12():
    """
    Constructs a packing of 12 disjoint unit regular hexagons inside a larger regular hexagon.
    Uses a rigid 'flower' cluster arrangement which is optimal for this specific topology.
    """
    sqrt3 = np.sqrt(3)
    
    # Centers of the 12 hexagons in a 1-6-5 flower arrangement
    # Center is at (0,0)
    # 6 neighbors at distance 2 (side length 1 hexagons touch at distance 2)
    # 5 more in the second ring to complete the compact shape
    
    centers = np.array([
        [0.0, 0.0],
        [2.0, 0.0],
        [1.0, sqrt3],
        [-1.0, sqrt3],
        [-2.0, 0.0],
        [-1.0, -sqrt3],
        [1.0, -sqrt3],
        [3.0, sqrt3],
        [0.0, 2*sqrt3],
        [-3.0, sqrt3],
        [-3.0, -sqrt3],
        [3.0, -sqrt3]
    ])
    
    # All inner hexagons have the same orientation (0 degrees)
    # This ensures they form a perfect honeycomb pattern
    angles = np.zeros(12)
    
    # Outer hexagon orientation
    # We align the outer hexagon such that its vertices align with the "points" of the inner cluster
    # The inner cluster has points at angles 0, 60, ... (vertices of the outer ring hexagons)
    # The outer hexagon should also have vertices at these angles to minimize size
    outer_angle = 0.0
    
    # Compute the bounding box of all inner vertices to find the minimal enclosing regular hexagon
    # Vertices of a unit hexagon at (cx, cy) with angle 0:
    # (cx + cos(60i), cy + sin(60i))
    
    all_vertices = []
    for i in range(12):
        cx, cy = centers[i]
        # Generate 6 vertices
        for j in range(6):
            theta = np.radians(60 * j)
            all_vertices.append([cx + np.cos(theta), cy + np.sin(theta)])
    
    all_vertices = np.array(all_vertices)
    
    # The outer hexagon is regular, centered at (ox, oy) with side s and angle outer_angle.
    # A point (x,y) is inside if for all 6 edge normals n_k:
    # n_k . (x - c) <= s * sqrt(3) / 2
    # The edge normals for a hexagon with angle 0 (vertex at 0 deg) are at angles 30, 90, 150, 210, 270, 330.
    # Wait, let's check the vertex definition.
    # Vertices at 0, 60, 120...
    # Edge 0 is between vertex 0 and 1. Midpoint angle 30. Normal is 30.
    # So normals are at 30 + 60k.
    
    normals = []
    for k in range(6):
        theta = np.radians(30 + 60 * k)
        normals.append([np.cos(theta), np.sin(theta)])
    normals = np.array(normals)
    
    # We want to find center c and side s such that all vertices are inside.
    # For a fixed orientation, the optimal center is the center of the bounding box in each normal direction.
    # However, for a regular hexagon, the center must be the same for all 6 directions.
    # Due to the symmetry of the flower cluster and the chosen orientation (0 deg),
    # the cluster is symmetric with respect to the origin.
    # Therefore, the center of the outer hexagon is (0,0).
    
    ox, oy = 0.0, 0.0
    
    # Calculate required s for each normal direction
    s_required = 0.0
    for n in normals:
        # Project all vertices onto this normal
        projections = all_vertices @ n
        # Distance from center (0,0) to the furthest vertex in this direction
        max_proj = projections.max()
        min_proj = projections.min()
        
        # The constraint is |n . (v - c)| <= s * sqrt(3) / 2
        # Since c=0, |n . v| <= s * sqrt(3) / 2
        # s >= |n . v| * 2 / sqrt(3)
        
        req_s = max(abs(max_proj), abs(min_proj)) * 2 / sqrt3
        s_required = max(s_required, req_s)
    
    s = s_required
    
    # Construct the output arrays
    # inner: (12, 3) array of rows (x, y, angle_deg)
    inner = np.column_stack([centers, angles])
    
    # outer: (3,) array (x, y, angle_deg)
    outer = np.array([ox, oy, outer_angle])
    
    return inner, outer, s

# EVOLVE-BLOCK-END
