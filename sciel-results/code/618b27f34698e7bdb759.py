import numpy as np
import time

def hexagon_packing_12():
    start_time = time.time()
    deadline = 600  # seconds
    
    # Define candidate configurations in axial coordinates (q, r)
    # Axial to Cartesian (flat-top, side 1):
    # x = sqrt(3) * (q + r/2)
    # y = 1.5 * r
    
    candidates = [
        # Config 1: Symmetric "star" extension
        [
            (0,0),
            (1,0), (0,1), (-1,1), (-1,0), (0,-1), (1,-1),
            (2,0), (1,1), (0,2),
            (-2,0), (-1,-1), (0,-2)
        ],
        # Config 2: Symmetric "cross" extension
        [
            (0,0),
            (1,0), (0,1), (-1,1), (-1,0), (0,-1), (1,-1),
            (2,0), (2,-1),
            (-2,0), (-2,1),
            (0,2), (0,-2)
        ],
        # Config 3: Compact block
        [
            (0,0),
            (1,0), (0,1), (-1,1), (-1,0), (0,-1), (1,-1),
            (2,0), (1,1),
            (-2,0), (-1,-1),
            (0,2), (0,-2)
        ]
    ]
    
    best_s = float('inf')
    best_inner = None
    best_outer = None
    
    sqrt3 = np.sqrt(3)
    
    for config in candidates:
        if time.time() - start_time > deadline - 10:
            break
            
        # Convert to Cartesian centers
        centers = []
        for q, r in config:
            x = sqrt3 * (q + r / 2.0)
            y = 1.5 * r
            centers.append((x, y))
        
        # Generate all vertices for all 12 hexagons
        all_vertices = []
        for cx, cy in centers:
            for i in range(6):
                angle = np.deg2rad(60 * i)
                vx = cx + np.cos(angle)
                vy = cy + np.sin(angle)
                all_vertices.append((vx, vy))
        
        all_vertices = np.array(all_vertices)
        
        # Calculate minimal enclosing hexagon side length s
        # The hexagon is centered at the centroid of the vertices? 
        # Actually, the outer hexagon can be shifted.
        # The minimal enclosing hexagon (axis-aligned) for a set of points
        # has side length s such that the projection width in each of the 6 
        # normal directions is <= s * sqrt(3).
        # We can center the outer hexagon at the mean of the vertices to minimize s?
        # Or just calculate the width for a hexagon centered at origin, 
        # but we can shift the inner points.
        # The problem allows shifting inner centers relative to outer.
        # So we can center the cluster at (0,0) for calculation.
        
        # Center the cluster
        center_of_mass = np.mean(all_vertices, axis=0)
        centered_vertices = all_vertices - center_of_mass
        
        # The 6 normal vectors for a hexagon with vertices at 0, 60, ...
        # The sides are perpendicular to these directions.
        # Normals point to the midpoints of the sides.
        # Side midpoints are at angles 30, 90, 150, 210, 270, 330.
        normals = []
        for i in range(6):
            angle = np.deg2rad(60 * i + 30)
            normals.append((np.cos(angle), np.sin(angle)))
        
        max_width = 0
        for nx, ny in normals:
            projections = centered_vertices[:, 0] * nx + centered_vertices[:, 1] * ny
            width = np.max(projections) - np.min(projections)
            if width > max_width:
                max_width = width
        
        # Width = s * sqrt(3)
        s = max_width / sqrt3
        
        if s < best_s:
            best_s = s
            best_inner = np.array([
                [cx, cy, 0] for cx, cy in centers
            ])
            # Outer hexagon is centered at origin (after shifting inner)
            # But we need to return the actual coordinates.
            # The task says: "after shifting all centers by minus the outer center..."
            # So we can set outer center to (0,0) and inner centers to their relative positions.
            best_outer = np.array([0, 0, 0])
    
    # If no valid config found (shouldn't happen), fallback to seed
    if best_inner is None:
        inner_hex_data = np.array(
            [
                [0, 0, 0],
                [-2.5, 0, 0],
                [2.5, 0, 0],
                [-1.25, 2.17, 0],
                [1.25, 2.17, 0],
                [-1.25, -2.17, 0],
                [1.25, -2.17, 0],
                [-3.75, 2.17, 0],
                [3.75, 2.17, 0],
                [-3.75, -2.17, 0],
                [3.75, -2.17, 0],
                [0, -4, 0]
            ]
        )
        outer_hex_data = np.array([0, 0, 0])
        outer_hex_side_length = 8
    else:
        inner_hex_data = best_inner
        outer_hex_data = best_outer
        outer_hex_side_length = best_s
        
    return inner_hex_data, outer_hex_data, outer_hex_side_length
