"""Packing 12 unit hexagons in minimal regular hexagon."""
import math
import time
import numpy as np
from scipy.optimize import minimize

N = 12

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 135.0  # Safety margin before 150s kill
    
    sqrt3 = math.sqrt(3.0)
    
    # Generate all 19 centers of the side-3 hexagon cluster
    # Axial coordinates (q, r) with max(|q|, |r|, |q+r|) <= 2
    # Using Flat-Top hexagon grid coordinates
    all_centers = []
    for q in range(-2, 3):
        for r in range(-2, 3):
            if abs(q + r) <= 2:
                # Convert axial to cartesian (flat-top hexagons)
                # x = 1.5 * q, y = sqrt(3) * (r + q/2)
                x = 1.5 * q
                y = sqrt3 * (r + q / 2.0)
                all_centers.append((x, y, q, r))
    
    # Normal directions for flat-top outer hexagon (pointing outward from center)
    # Normals at 30, 90, 150, 210, 270, 330 degrees
    normal_angles = [math.radians(30 + 60 * j) for j in range(6)]
    normal_cos = np.array([math.cos(a) for a in normal_angles])
    normal_sin = np.array([math.sin(a) for a in normal_angles])
    
    def min_L_for_hexagon(cx, cy, theta=0.0):
        """Find the minimum L needed for a unit hexagon at (cx,cy) with rotation theta to fit in outer hex of side L."""
        # Vertex angles relative to hexagon center
        vertex_angles = np.array([math.radians(60 * k) + theta for k in range(6)])
        vx = cx + np.cos(vertex_angles)
        vy = cy + np.sin(vertex_angles)
        
        # Projection of each vertex onto each normal direction
        projs = vx[:, np.newaxis] * normal_cos[np.newaxis, :] + vy[:, np.newaxis] * normal_sin[np.newaxis, :]
        max_proj = np.max(projs)
        return max_proj / (sqrt3 / 2.0)

    def compute_L(positions):
        """Compute the outer side length for a set of positions (12, 2) with angle 0."""
        max_proj = 0.0
        # Vertices for a unit hexagon at origin, angle 0
        # Flat top: (1,0), (0.5, sqrt3/2), ...
        verts_x = np.array([1.0, 0.5, -0.5, -1.0, -0.5, 0.5])
        verts_y = np.array([0.0, sqrt3/2, sqrt3/2, 0.0, -sqrt3/2, -sqrt3/2])
        
        for i in range(N):
            cx, cy = positions[i]
            vx = cx + verts_x
            vy = cy + verts_y
            projs = vx * normal_cos + vy * normal_sin
            # Max projection for this hexagon
            # We need the global max over all hexagons
            # Actually, we can just take the max of all projections
            max_proj = max(max_proj, np.max(projs))
        return max_proj / (sqrt3 / 2.0)

    def h_minkowski(d_vec):
        """
        Compute the support function of the Minkowski sum of two aligned unit hexagons
        in the direction of d_vec.
        The Minkowski sum is a hexagon with circumradius 2.
        h(theta) = 2 * cos(delta), where delta is the angle to the nearest vertex.
        Vertices are at 0, 60, 120, 180, 240, 300 degrees.
        """
        norm = np.linalg.norm(d_vec)
        if norm < 1e-9:
            return 2.0 # Max support
        u = d_vec / norm
        theta = np.arctan2(u[1], u[0])
        # Angle to nearest vertex (multiples of 60 deg)
        # theta is in [-pi, pi]
        # Nearest vertex angle is round(theta / (pi/3)) * (pi/3)
        # But we can just use cos difference
        # delta = theta - k*pi/3
        # We want min |delta|
        # delta is in [-pi/6, pi/6] relative to a vertex
        # Let's normalize theta to [0, 2pi)
        theta_pos = theta % (2 * math.pi)
        # Find closest multiple of pi/3
        k = round(theta_pos / (math.pi / 3))
        theta_vertex = k * (math.pi / 3)
        delta = abs(theta_pos - theta_vertex)
        # If delta > pi/6, it's closer to the next one?
        # The max delta is pi/6 (30 deg)
        # So h = 2 * cos(delta)
        return 2.0 * math.cos(delta)

    def constraints_func(positions):
        """
        Return list of constraint values.
        Constraint: ||p_i - p_j|| - h_minkowski(p_i - p_j) >= 0
        """
        cons = []
        for i in range(N):
            for j in range(i + 1, N):
                d = positions[i] - positions[j]
                dist = np.linalg.norm(d)
                h = h_minkowski(d)
                # We want dist >= h
                # So dist - h >= 0
                cons.append(dist - h)
        return cons

    # 1. Find best lattice subset (fast)
    # Precompute optimal theta for each center (independent of subset) - we will ignore theta in optimization for speed
    # But we need to know which 12 centers to use.
    # The seed program does this. We can reuse the logic or hardcode a good one.
    # Let's run the fast combinatorial search for the best 12 centers based on L with angle 0.
    from itertools import combinations
    
    best_L_init = float('inf')
    best_subset = None
    
    # We can speed this up by just checking a few promising subsets
    # Or just run the full search as it was fast enough in the seed.
    # Let's assume the seed's search was fast.
    
    # Precompute L for each center with angle 0
    center_Ls = []
    for i in range(19):
        x, y, _, _ = all_centers[i]
        L_h = min_L_for_hexagon(x, y, 0.0)
        center_Ls.append(L_h)
        
    for subset in combinations(range(19), 12):
        if time.time() > deadline - 10.0: # Leave time for optimization
            break
        L = max(center_Ls[i] for i in subset)
        if L < best_L_init:
            best_L_init = L
            best_subset = subset
            
    if best_subset is None:
        best_subset = tuple(range(12))
        
    # Initialize positions from the best lattice subset
    x0 = np.zeros((N, 2))
    for k, idx in enumerate(best_subset):
        x0[k, 0] = all_centers[idx][0]
        x0[k, 1] = all_centers[idx][1]
        
    # 2. Continuous Optimization
    # Variables: flattened x0
    x_flat0 = x0.flatten()
    
    # Objective function
    def objective(x_flat):
        pos = x_flat.reshape(N, 2)
        return compute_L(pos)
        
    # Constraints
    # SLSQP requires constraints in the form {'type': 'ineq', 'fun': ...}
    # fun must return an array of values >= 0
    def constraints(x_flat):
        pos = x_flat.reshape(N, 2)
        vals = constraints_func(pos)
        return vals
        
    # Bounds: Keep positions within a reasonable box
    # The outer hexagon side is likely around 5-6.
    # Centers should be within [-4, 4]
    bounds = [(-5, 5) for _ in range(2 * N)]
    
    try:
        if time.time() < deadline - 5.0:
            res = minimize(objective, x_flat0, method='SLSQP', 
                           bounds=bounds, 
                           constraints={'type': 'ineq', 'fun': constraints},
                           options={'maxiter': 1000, 'ftol': 1e-8})
            if res.success or res.fun < best_L_init:
                best_positions = res.x.reshape(N, 2)
                best_L_opt = res.fun
            else:
                best_positions = x0
                best_L_opt = best_L_init
        else:
            best_positions = x0
            best_L_opt = best_L_init
    except Exception as e:
        # Fallback to lattice
        best_positions = x0
        best_L_opt = best_L_init
        
    # Construct inner hexagons
    inner = []
    for i in range(N):
        x, y = best_positions[i]
        inner.append([x, y, 0.0]) # Angle 0
        
    # Calculate final L
    final_L = compute_L(best_positions)
    
    return inner, (0.0, 0.0), final_L, 0.0
# EVOLVE-BLOCK-END
