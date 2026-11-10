# EVOLVE-BLOCK-START
import numpy as np

def hexagon_packing_11():
    sqrt3 = np.sqrt(3)
    
    # 1-3-3-3-1 Packing (Pointy-Top Hexagons)
    # Row 0 (y=0): 1 hex at x=0
    # Row 1 (y=1.5): 3 hexes at x=-sqrt(3), 0, sqrt(3)
    # Row 2 (y=3.0): 3 hexes at x=-sqrt(3), 0, sqrt(3)
    # Row 3 (y=4.5): 3 hexes at x=-sqrt(3), 0, sqrt(3)
    # Row 4 (y=6.0): 1 hex at x=0
    # Wait, 1+3+3+3+1 = 11.
    # Vertical spacing 1.5? No, 1.5 causes overlap.
    # Correct vertical spacing for pointy-top touching is 2.0?
    # No, touching side-to-side is sqrt(3).
    # Touching vertex-to-vertex (stacked) is 2.0.
    # But we can interlock.
    # Row 0: (0,0)
    # Row 1: (-sqrt(3)/2, 1.5), (sqrt(3)/2, 1.5) -> Only 2.
    # To get 3, we need (-sqrt(3), 1.5), (0, 1.5), (sqrt(3), 1.5).
    # Dist (0,0) to (0,1.5) is 1.5 < sqrt(3). Overlap.
    # So we must use spacing 2.0 for vertical, or shift x.
    # If we use spacing 2.0:
    # Row 0: (0,0)
    # Row 1: (-sqrt(3), 2), (0, 2), (sqrt(3), 2)
    # Dist (0,0) to (0,2) is 2.0 > sqrt(3). OK.
    # Dist (0,0) to (sqrt(3), 2) is sqrt(3+4)=sqrt(7) > sqrt(3). OK.
    # Dist (-sqrt(3),2) to (0,2) is sqrt(3). OK.
    
    # So:
    coords = []
    # Row 0
    coords.append((0.0, 0.0, 0.0))
    # Row 1
    coords.append((-sqrt3, 2.0, 0.0))
    coords.append((0.0, 2.0, 0.0))
    coords.append((sqrt3, 2.0, 0.0))
    # Row 2
    coords.append((-sqrt3, 4.0, 0.0))
    coords.append((0.0, 4.0, 0.0))
    coords.append((sqrt3, 4.0, 0.0))
    # Row 3
    coords.append((-sqrt3, 6.0, 0.0))
    coords.append((0.0, 6.0, 0.0))
    coords.append((sqrt3, 6.0, 0.0))
    # Row 4
    coords.append((0.0, 8.0, 0.0))
    
    # Center the cluster
    ys = [c[1] for c in coords]
    min_y = min(ys)
    max_y = max(ys)
    cy = (min_y + max_y) / 2.0
    coords = [(x, y - cy, a) for x, y, a in coords]
    
    # Calculate vertices of all inner hexagons
    all_verts = []
    for x, y, a in coords:
        angles = np.radians(a) + np.radians(60.0) * np.arange(6)
        vs = np.column_stack([x + np.cos(angles), y + np.sin(angles)])
        all_verts.append(vs)
    all_verts = np.vstack(all_verts)
    
    # Find minimal outer hexagon side
    # Try different orientations of the outer hexagon
    best_side = np.inf
    best_outer_angle = 0.0
    
    for outer_angle_deg in [0, 30, 60]:
        # Outer hexagon vertices for side S, angle a
        # We want to find S such that all inner verts are inside.
        # For a given direction (normal to an edge), the projection of the outer hexagon
        # must cover the projection of the inner vertices.
        # The width of the hexagon in direction theta is S * (cos(theta - a) ... )
        # Actually, the support function of a hexagon with side S and angle a:
        # h(theta) = S * max(0, cos(theta - a - 60k)) ?
        # Simpler: The distance from center to edge in direction theta is S * sqrt(3)/2 / max(1, ...)
        # Let's just check the distance from origin to each vertex of the inner hexagons.
        # The outer hexagon must contain the convex hull of the inner vertices.
        # The minimal circumscribed hexagon is not simply determined by max radius.
        # But for a regular hexagon, the side length S is determined by the max projection.
        
        # For a regular hexagon with side S and angle a (vertex at angle a),
        # the distance from center to the edge normal at angle phi is:
        # d(phi) = S * sqrt(3)/2 * (1 / max(cos(phi - a - 60k)))?
        # Let's use the property that the hexagon is the intersection of 6 half-planes.
        # Edge i normal angle: a + 60*i + 30?
        # If vertex is at a, edge normal is at a + 30 + 60*i.
        # Distance from center to edge = S * sqrt(3)/2.
        
        # So for each of the 6 edge normals, the projection of all inner vertices
        # onto that normal must be <= S * sqrt(3)/2.
        
        normals = []
        a_rad = np.radians(outer_angle_deg)
        # Vertex at a_rad. Edge normals at a_rad + 30, 90, ...
        for i in range(6):
            n_angle = a_rad + np.radians(30) + np.radians(60) * i
            normals.append(np.array([np.cos(n_angle), np.sin(n_angle)]))
        
        max_proj = 0.0
        for n in normals:
            proj = all_verts @ n
            max_proj = max(max_proj, np.max(np.abs(proj)))
        
        # max_proj = S * sqrt(3)/2
        S = max_proj / (np.sqrt(3) / 2)
        
        if S < best_side:
            best_side = S
            best_outer_angle = outer_angle_deg

    return np.array(coords), np.array([0.0, 0.0, best_outer_angle]), float(best_side)

# EVOLVE-BLOCK-END
