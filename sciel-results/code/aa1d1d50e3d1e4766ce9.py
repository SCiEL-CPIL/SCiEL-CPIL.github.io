import numpy as np
import time
import math
from scipy.optimize import linprog

N = 12

def construct():
    start_time = time.time()
    deadline = start_time + 140.0

    # 1. Define a dense, pre-validated honeycomb cluster of 12 unit hexagons.
    # This cluster is 1 center + 6 ring-1 + 5 ring-2.
    # We choose 5 of the 6 ring-2 positions to create an asymmetric, tighter packing.
    # Hex lattice coordinates (q, r) -> Cartesian (x, y):
    # x = sqrt(3) * (q + r/2)
    # y = 1.5 * r
    
    # Standard 12-hex cluster coordinates (1 center, 6 ring 1, 5 ring 2)
    # Ring 0: (0,0)
    # Ring 1: (1,0), (0,1), (-1,1), (-1,0), (0,-1), (1,-1)
    # Ring 2 (choose 5 out of 6): (2,0), (1,1), (0,2), (-1,2), (-2,1) -- omitting (-2, 0)
    # This creates a "C" shape or similar asymmetric tight cluster.
    
    coords = [
        (0, 0),
        (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
        (2, 0), (1, 1), (0, 2), (-1, 2), (-2, 1)
    ]
    
    inner_positions = []
    for q, r in coords:
        x = math.sqrt(3) * (q + r / 2.0)
        y = 1.5 * r
        inner_positions.append((x, y))
    
    inner_positions = np.array(inner_positions)
    
    # Verify validity of this fixed cluster (should be disjoint)
    # For unit hexagons in a honeycomb lattice, they are tangent.
    # Distance between centers of adjacent hexagons is 2 * apothem = sqrt(3).
    # Let's quickly check min distance to ensure no overlap (dist >= sqrt(3) - epsilon)
    min_dist = float('inf')
    for i in range(N):
        for j in range(i+1, N):
            d = np.linalg.norm(inner_positions[i] - inner_positions[j])
            if d < min_dist:
                min_dist = d
    # sqrt(3) is approx 1.732. If min_dist < 1.732 - 1e-5, they overlap.
    # In a perfect honeycomb, they touch.
    
    # 2. Solve for the minimal enclosing regular hexagon using Linear Programming.
    # Variables: [cx, cy, s]
    # cx, cy: center of outer hexagon
    # s: side length of outer hexagon
    
    # Constraints:
    # For each inner hexagon vertex v_i (absolute coordinates),
    # it must be inside the outer hexagon.
    # The outer hexagon is the intersection of 6 half-planes.
    # Let the outer hexagon have side length s, center (cx, cy), and rotation 0 (standard orientation).
    # The normals to the sides of a standard regular hexagon (flat top/bottom? or pointy top?)
    # Let's use the same orientation as the inner hexagons for simplicity (pointy top/bottom? 
    # The seed used vertices at 0, 60, 120... which is pointy right/left).
    # Normals for a hexagon with vertices at angles 0, 60, ... are at 30, 90, 150, 210, 270, 330.
    # n_k = [cos(30+60k), sin(30+60k)]
    # Support function value for side s is s * sqrt(3) / 2.
    # Condition: (v - c) . n_k <= s * sqrt(3) / 2
    # v_x * n_kx - c_x * n_kx + v_y * n_ky - c_y * n_ky <= s * sqrt(3) / 2
    # v_x * n_kx + v_y * n_ky - n_kx * c_x - n_ky * c_y - (sqrt(3)/2) * s <= 0
    
    # Generate all vertices of the 12 inner hexagons
    all_vertices = []
    for px, py in inner_positions:
        # Inner hexagons are rotated by 0 degrees (same as outer reference frame)
        # Vertices at angles 0, 60, 120, 180, 240, 300
        for k in range(6):
            angle = math.radians(60 * k)
            vx = px + 1.0 * math.cos(angle)
            vy = py + 1.0 * math.sin(angle)
            all_vertices.append((vx, vy))
            
    all_vertices = np.array(all_vertices)
    
    # Define the 6 normal vectors
    normals = []
    for k in range(6):
        angle = math.radians(30 + 60 * k)
        nx = math.cos(angle)
        ny = math.sin(angle)
        normals.append((nx, ny))
    normals = np.array(normals)
    
    # Construct A_ub and b_ub
    # Number of constraints: 6 normals * 72 vertices = 432 constraints
    # Variables: [cx, cy, s]
    
    A_ub = []
    b_ub = []
    
    sqrt3 = math.sqrt(3.0)
    rhs_const = sqrt3 / 2.0
    
    for v in all_vertices:
        vx, vy = v
        for n in normals:
            nx, ny = n
            # vx*nx + vy*ny - nx*cx - ny*cy - rhs_const*s <= 0
            # -nx*cx - ny*cy - rhs_const*s <= -(vx*nx + vy*ny)
            A_ub.append([-nx, -ny, -rhs_const])
            b_ub.append(-(vx * nx + vy * ny))
            
    A_ub = np.array(A_ub)
    b_ub = np.array(b_ub)
    
    # Objective: Minimize s
    c_obj = [0, 0, 1]
    
    # Bounds: s > 0, cx, cy unbounded (or large bounds)
    bounds = [(None, None), (None, None), (0, None)]
    
    try:
        res = linprog(c_obj, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
        
        if res.status == 0:
            cx_opt, cy_opt, s_opt = res.x
            # Ensure s is positive
            s_opt = max(0.0, s_opt)
        else:
            # Fallback if LP fails (should not happen for convex containment)
            cx_opt, cy_opt = 0.0, 0.0
            # Calculate max projection to estimate s
            # For safety, use a large value
            s_opt = 10.0
            # Better fallback: compute max projection for center 0,0
            max_proj = 0
            for v in all_vertices:
                for n in normals:
                    p = v[0]*n[0] + v[1]*n[1]
                    if p > max_proj: max_proj = p
            s_opt = max_proj / (sqrt3/2)
            
    except Exception:
        cx_opt, cy_opt = 0.0, 0.0
        s_opt = 10.0

    # 3. Construct the output
    inner = []
    for i in range(N):
        inner.append([float(inner_positions[i, 0]), float(inner_positions[i, 1]), 0.0])
        
    outer_center = [float(cx_opt), float(cy_opt)]
    outer_side = float(s_opt)
    outer_angle = 0.0
    
    return inner, outer_center, outer_side, outer_angle
