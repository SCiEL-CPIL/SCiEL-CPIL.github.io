"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
from itertools import combinations, product
from math import sqrt, cos, sin, pi, atan2, floor, ceil

N = 11

def hex_vertices(cx, cy, side, angle_deg):
    """Return vertices of a regular hexagon centered at (cx,cy) with given side and rotation."""
    a = np.radians(angle_deg)
    verts = []
    for i in range(6):
        theta = a + i * pi / 3
        verts.append((cx + side * cos(theta), cy + side * sin(theta)))
    return verts

def point_in_hexagon(px, py, cx, cy, side, angle_deg):
    """Check if point (px,py) is inside or on boundary of hexagon."""
    # Transform to hexagon's local coordinate system
    a = np.radians(angle_deg)
    dx = px - cx
    dy = py - cy
    # Rotate by -a
    rx = dx * cos(-a) - dy * sin(-a)
    ry = dx * sin(-a) + dy * cos(-a)
    
    # In local coords, hexagon has vertices at angles 0, 60, 120, 180, 240, 300
    # The hexagon can be described as intersection of 6 half-planes
    # For a regular hexagon with side s centered at origin, the distance from center
    # to each edge (apothem) is s * sqrt(3)/2
    # The edges are at angles 30, 90, 150, 210, 270, 330 degrees from center
    apothem = side * sqrt(3) / 2
    
    # Check each of the 6 edges
    for i in range(6):
        edge_angle = np.radians(30 + i * 60)
        # Normal vector pointing inward
        nx = cos(edge_angle)
        ny = sin(edge_angle)
        # Distance from center to this edge plane
        dist = rx * nx + ry * ny
        if dist > apothem + 1e-10:
            return False
    return True

def hexagon_contains_hexagon(cx1, cy1, s1, a1, cx2, cy2, s2, a2):
    """Check if hexagon 1 contains hexagon 2 (all vertices of 2 inside 1)."""
    verts2 = hex_vertices(cx2, cy2, s2, a2)
    for vx, vy in verts2:
        if not point_in_hexagon(vx, vy, cx1, cy1, s1, a1):
            return False
    return True

def enclosing_hexagon_side(centers, angles, n_inner):
    """
    Find minimum side length of a regular hexagon (with same rotation as outer, which we assume 0)
    that contains all inner hexagons. We'll try different rotations of the outer hexagon.
    """
    # For a given outer rotation, find minimum side and center
    # The optimal outer hexagon will have at least 3 vertices on the boundary of the union
    # Let's use a simple approach: try multiple outer rotations, and for each,
    # find the minimum enclosing hexagon using linear programming or geometric approach
    
    min_side = float('inf')
    
    # Try outer rotations from 0 to 60 degrees in steps
    for outer_angle in np.arange(0, 60, 5):
        # For this outer rotation, we need to find center and side
        # such that all inner hexagons are inside
        
        # The constraint: for each inner hexagon vertex, it must be inside the outer hexagon
        # Outer hexagon: center (cx, cy), side S, rotation outer_angle
        # A point (px, py) is inside if for all 6 edges: dot((px-cx, py-cy), normal_i) <= apothem
        
        # This is a linear feasibility problem in (cx, cy, S) for fixed outer_angle
        # Let's use scipy if available, otherwise grid search
        
        try:
            from scipy.optimize import linprog
            from scipy.optimize import minimize
        except:
            linprog = None
            minimize = None
        
        if minimize is not None:
            # Use optimization
            def objective(params):
                cx, cy, S = params
                if S <= 0:
                    return 1e10
                return S
            
            # Constraints: each vertex of each inner hex must be inside outer
            # We'll use a penalty method or just check
            
            # Grid search over center and side
            # First, estimate bounding box of all inner hex vertices
            all_verts = []
            for i in range(len(centers)):
                cx_i, cy_i = centers[i]
                ang_i = angles[i]
                verts = hex_vertices(cx_i, cy_i, 1.0, ang_i)
                all_verts.extend(verts)
            all_verts = np.array(all_verts)
            x_min, x_max = all_verts[:, 0].min(), all_verts[:, 0].max()
            y_min, y_max = all_verts[:, 1].min(), all_verts[:, 1].max()
            
            # Optimal center is likely near centroid
            cx0 = (x_min + x_max) / 2
            cy0 = (y_min + y_max) / 2
            # Estimate side
            r_max = max(sqrt((vx - cx0)**2 + (vy - cy0)**2) for vx, vy in all_verts)
            S0 = r_max * 2 / sqrt(3) + 1  # rough estimate
            
            # Try a few candidate centers and sides
            best_S = float('inf')
            for dcx in np.linspace(-1, 1, 5):
                for dcy in np.linspace(-1, 1, 5):
                    cx_try = cx0 + dcx
                    cy_try = cy0 + dcy
                    # Find minimum S for this center
                    # For each vertex, compute required S
                    max_req = 0
                    for vx, vy in all_verts:
                        dx = vx - cx_try
                        dy = vy - cy_try
                        # Transform to outer hex local coords
                        a = np.radians(outer_angle)
                        rx = dx * cos(-a) - dy * sin(-a)
                        ry = dx * sin(-a) + dy * cos(-a)
                        # For each of 6 edges, check constraint
                        # dist_i = rx * nx_i + ry * ny_i <= S * sqrt(3)/2
                        # So S >= 2/sqrt(3) * max_i(dist_i)
                        for j in range(6):
                            edge_angle = np.radians(30 + j * 60)
                            nx = cos(edge_angle)
                            ny = sin(edge_angle)
                            dist = rx * nx + ry * ny
                            if dist > max_req:
                                max_req = dist
                    S_try = max_req * 2 / sqrt(3)
                    if S_try < best_S:
                        best_S = S_try
            
            if best_S < min_side:
                min_side = best_S
        else:
            # Fallback: grid search
            all_verts = []
            for i in range(len(centers)):
                cx_i, cy_i = centers[i]
                ang_i = angles[i]
                verts = hex_vertices(cx_i, cy_i, 1.0, ang_i)
                all_verts.extend(verts)
            all_verts = np.array(all_verts)
            x_min, x_max = all_verts[:, 0].min(), all_verts[:, 0].max()
            y_min, y_max = all_verts[:, 1].min(), all_verts[:, 1].max()
            
            cx0 = (x_min + x_max) / 2
            cy0 = (y_min + y_max) / 2
            r_max = max(sqrt((vx - cx0)**2 + (vy - cy0)**2) for vx, vy in all_verts)
            S0 = r_max * 2 / sqrt(3) + 1
            
            best_S = float('inf')
            for dcx in np.linspace(-1, 1, 7):
                for dcy in np.linspace(-1, 1, 7):
                    cx_try = cx0 + dcx
                    cy_try = cy0 + dcy
                    max_req = 0
                    for vx, vy in all_verts:
                        dx = vx - cx_try
                        dy = vy - cy_try
                        a = np.radians(outer_angle)
                        rx = dx * cos(-a) - dy * sin(-a)
                        ry = dx * sin(-a) + dy * cos(-a)
                        for j in range(6):
                            edge_angle = np.radians(30 + j * 60)
                            nx = cos(edge_angle)
                            ny = sin(edge_angle)
                            dist = rx * nx + ry * ny
                            if dist > max_req:
                                max_req = dist
                    S_try = max_req * 2 / sqrt(3)
                    if S_try < best_S:
                        best_S = S_try
            
            if best_S < min_side:
                min_side = best_S
    
    return min_side

def construct():
    """
    Construct 11 unit hexagons inside a regular hexagon, minimizing outer side length.
    """
    # Strategy: Use hexagonal lattice packing
    # For N=11, a good configuration is a hexagonal cluster
    # Center hexagon + 6 surrounding + 4 more
    
    # Hexagonal lattice basis
    # Hexagons of side 1 packed in hexagonal arrangement
    # Distance between centers of adjacent hexagons = sqrt(3)
    
    # Let's try a compact hexagonal cluster
    # Coordinates in hex grid: (q, r) where position = q * v1 + r * v2
    # v1 = (sqrt(3), 0), v2 = (sqrt(3)/2, 3/2)
    
    # For 11 hexagons, try: center + ring of 6 + 4 from next ring
    # Or a different compact shape
    
    # Let's use axial coordinates
    # A hexagonal cluster of "radius" 1 has 7 hexagons (center + 6 neighbors)
    # For 11, we add 4 more
    
    # Positions in axial coords (q, r):
    # Center: (0,0)
    # Ring 1: (1,0), (1,-1), (0,-1), (-1,0), (-1,1), (0,1)
    # Ring 2 partial: (2,0), (2,-1), (1,-2), (0,-2)
    
    # Convert axial to cartesian:
    # x = sqrt(3) * (q + r/2)
    # y = 3/2 * r
    
    sqrt3 = sqrt(3)
    
    # Try several configurations
    
    configs = []
    
    # Config 1: center + 6 + 4 from ring 2
    axial1 = [(0,0), (1,0), (1,-1), (0,-1), (-1,0), (-1,1), (0,1),
              (2,0), (2,-1), (1,-2), (0,-2)]
    configs.append(axial1)
    
    # Config 2: center + 6 + 4 different from ring 2
    axial2 = [(0,0), (1,0), (1,-1), (0,-1), (-1,0), (-1,1), (0,1),
              (2,0), (0,-2), (-2,0), (0,2)]
    configs.append(axial2)
    
    # Config 3: Linear-ish but bent
    axial3 = [(0,0), (1,0), (2,0), (3,0), (4,0),
              (1,-1), (2,-1), (3,-1),
              (1,1), (2,1), (3,1)]
    configs.append(axial3)
    
    # Config 4: More compact
    axial4 = [(0,0), (1,0), (1,-1), (0,-1), (-1,0), (-1,1), (0,1),
              (2,-1), (-2,1), (2,0), (-2,0)]
    configs.append(axial4)
    
    best_score = float('inf')
    best_result = None
    
    for axial in configs:
        # Convert to cartesian
        centers = []
        for q, r in axial:
            x = sqrt3 * (q + r/2)
            y = 1.5 * r
            centers.append((x, y))
        
        # All angles 0
        angles = [0.0] * len(centers)
        
        # Compute enclosing hexagon side
        side = enclosing_hexagon_side(centers, angles, len(centers))
        
        if side < best_score:
            best_score = side
            best_result = (centers, angles, side)
    
    if best_result is None:
        # Fallback: linear arrangement
        centers = [(sqrt3 * i, 0.0) for i in range(N)]
        angles = [0.0] * N
        side = enclosing_hexagon_side(centers, angles, N)
        best_result = (centers, angles, side)
    
    centers, angles, side = best_result
    
    # Center the configuration
    cx = sum(c[0] for c in centers) / len(centers)
    cy = sum(c[1] for c in centers) / len(centers)
    
    inner = []
    for i in range(len(centers)):
        x = centers[i][0] - cx
        y = centers[i][1] - cy
        inner.append([x, y, angles[i]])
    
    return inner, (0.0, 0.0), side, 0.0

# EVOLVE-BLOCK-START
# The construct function is defined above
# EVOLVE-BLOCK-END
