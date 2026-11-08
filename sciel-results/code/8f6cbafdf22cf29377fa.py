import numpy as np

def construct():
    # Axial coordinates for 12 hexagons
    # A compact cluster: 7-center + 5 extensions
    axial_coords = [
        (0, 0),
        (1, 0), (1, -1), (0, -1), (-1, 0), (-1, 1), (0, 1),
        (2, -1), (2, 0), (-2, 1), (-2, 0), (0, 2)
    ]
    
    # Convert axial to Cartesian
    # x = 1.5 * q
    # y = sqrt(3) * (r + 0.5 * q)
    sqrt3 = np.sqrt(3)
    centers = []
    for q, r in axial_coords:
        x = 1.5 * q
        y = sqrt3 * (r + 0.5 * q)
        centers.append((x, y))
    
    # Calculate minimal enclosing hexagon side length
    # The container is a regular hexagon centered at (0,0) with side L.
    # We need to find L such that all hexagons fit inside.
    # A hexagon at (cx, cy) fits if all its vertices are inside the container.
    # The vertices of a unit hexagon at (cx, cy) are:
    # (cx + cos(theta), cy + sin(theta)) for theta in [0, 60, ..., 300]
    # The container is defined by 6 inequalities.
    # For a hexagon aligned with axes (flat top), the boundaries are:
    # |x| <= L * sqrt(3) / 2  (No, this is for pointy top)
    # Let's use the distance from origin to the boundary.
    # For a regular hexagon of side L, the distance from center to side is R_in = L * sqrt(3) / 2.
    # The distance from center to vertex is R_out = L.
    # The condition for a point (px, py) to be inside is:
    # max( |px| / (L*sqrt(3)/2), |py| / L ) <= 1 ? No.
    # Let's use the polar form or linear inequalities.
    # Inequalities for hexagon with vertices at (L,0), (L/2, L*sqrt(3)/2), ...
    # 1. x <= L
    # 2. x/2 + y*sqrt(3)/2 <= L
    # 3. -x/2 + y*sqrt(3)/2 <= L
    # 4. -x <= L
    # 5. -x/2 - y*sqrt(3)/2 <= L
    # 6. x/2 - y*sqrt(3)/2 <= L
    
    # We need to find the minimum L such that for all centers (cx, cy) and all vertices (vx, vy) of the hexagons,
    # the inequalities hold.
    # This is equivalent to finding the maximum of the 6 linear functions over all vertices.
    
    max_l = 0.0
    for cx, cy in centers:
        # Generate vertices of the unit hexagon at (cx, cy)
        for k in range(6):
            angle = np.pi / 3 * k
            vx = cx + np.cos(angle)
            vy = cy + np.sin(angle)
            
            # Calculate required L for each constraint
            # 1. x <= L  => L >= vx
            # 2. x/2 + y*sqrt(3)/2 <= L => L >= vx/2 + vy*sqrt(3)/2
            # 3. -x/2 + y*sqrt(3)/2 <= L => L >= -vx/2 + vy*sqrt(3)/2
            # 4. -x <= L => L >= -vx
            # 5. -x/2 - y*sqrt(3)/2 <= L => L >= -vx/2 - vy*sqrt(3)/2
            # 6. x/2 - y*sqrt(3)/2 <= L => L >= vx/2 - vy*sqrt(3)/2
            
            l1 = vx
            l2 = vx/2 + vy*sqrt3/2
            l3 = -vx/2 + vy*sqrt3/2
            l4 = -vx
            l5 = -vx/2 - vy*sqrt3/2
            l6 = vx/2 - vy*sqrt3/2
            
            current_max = max(l1, l2, l3, l4, l5, l6)
            if current_max > max_l:
                max_l = current_max
    
    # Add a small epsilon to ensure strict containment
    side = max_l + 1e-6
    
    # Prepare inner hexagons list
    inner = []
    for cx, cy in centers:
        inner.append([cx, cy, 0.0])
        
    return inner, (0.0, 0.0), side, 0.0
