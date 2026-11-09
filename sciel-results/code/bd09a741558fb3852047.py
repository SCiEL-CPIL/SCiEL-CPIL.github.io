"""Packing 12 unit hexagons in minimal regular hexagon."""
import math
import time
import numpy as np
from scipy.optimize import minimize

N = 12

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s kill
    
    sqrt3 = math.sqrt(3.0)
    
    # Generate all 19 centers of the side-3 hexagon cluster
    all_centers = []
    for q in range(-2, 3):
        for r in range(-2, 3):
            if abs(q + r) <= 2:
                x = 1.5 * q
                y = sqrt3 * (r + q / 2.0)
                all_centers.append((x, y, q, r))
    
    # Normal directions for outer hexagon (pointing outward from center)
    # For a regular hexagon with side L, the inradius is sqrt(3)/2 * L
    # Normals at 0, 60, 120, 180, 240, 300 degrees (flat-top orientation)
    normal_angles = [math.radians(60 * j) for j in range(6)]
    normal_cos = np.array([math.cos(a) for a in normal_angles])
    normal_sin = np.array([math.sin(a) for a in normal_angles])
    
    def hexagon_vertices(cx, cy, theta=0.0):
        """Return vertices of a unit regular hexagon at (cx,cy) with rotation theta."""
        angles = np.array([math.radians(60 * k) + theta for k in range(6)])
        vx = cx + np.cos(angles)
        vy = cy + np.sin(angles)
        return np.column_stack([vx, vy])
    
    def min_L_for_hexagon(cx, cy, theta=0.0):
        """Find the minimum L needed for a unit hexagon at (cx,cy) with rotation theta to fit in outer hex of side L centered at origin."""
        verts = hexagon_vertices(cx, cy, theta)
        # The outer hexagon has inradius = sqrt(3)/2 * L
        # A point (x,y) is inside if max over normals of (x*cos(a) + y*sin(a)) <= inradius
        projs = verts[:, 0] * normal_cos + verts[:, 1] * normal_sin
        max_proj = np.max(projs)
        return max_proj / (sqrt3 / 2.0)
    
    def compute_L(positions, angles):
        """Compute the outer side length for a set of positions (12,2) with angles (12,)."""
        max_proj = 0.0
        for i in range(N):
            cx, cy = positions[i]
            theta = angles[i]
            verts = hexagon_vertices(cx, cy, theta)
            projs = verts[:, 0] * normal_cos + verts[:, 1] * normal_sin
            max_proj = max(max_proj, np.max(projs))
        return max_proj / (sqrt3 / 2.0)
    
    def min_distance_for_non_overlap(theta1, theta2):
        """
        Compute the minimum distance between centers of two unit hexagons 
        with rotations theta1 and theta2 such that they don't overlap.
        This is the support function of the Minkowski sum.
        For two convex polygons P and Q, they don't overlap iff 
        for all directions d, h_P(-d) + h_Q(d) >= |p_q| where p_q is the vector between centers.
        The minimum distance is max over directions of h_P(d) + h_Q(d).
        """
        # For two regular hexagons, the Minkowski sum is a 12-gon (or hexagon if aligned)
        # The support function h(d) = max over vertices of dot(v, d)
        # We need max over all directions d of h_hex1(d) + h_hex2(-d)
        # = max over d of h_hex1(d) + h_hex2(d)  [by symmetry]
        
        # For a regular hexagon with circumradius 1, the support function in direction theta is:
        # h(theta) = cos(theta - nearest_vertex_angle) where nearest_vertex_angle is multiple of 60 deg
        # More precisely: h(theta) = cos(min_k |theta - k*pi/3|) for the unit hexagon
        
        def support_hex(theta, rotation):
            """Support function of unit hexagon rotated by 'rotation' in direction theta."""
            # The hexagon has vertices at angles rotation + k*60 for k=0..5
            # Support in direction theta is max over vertices of cos(theta - vertex_angle)
            # = cos(theta - rotation - nearest_multiple_of_60)
            # The maximum of cos(a) over a in the set {theta - rotation - k*pi/3} 
            # is cos(min_k |theta - rotation - k*pi/3|)
            
            # Normalize to find the angular distance to nearest vertex
            diff = theta - rotation
            # Find nearest multiple of pi/3
            k = round(diff / (math.pi / 3))
            delta = abs(diff - k * math.pi / 3)
            # delta is in [0, pi/6]
            return math.cos(delta)
        
        # The minimum distance is max over direction theta of h1(theta) + h2(-theta)
        # = max over theta of h1(theta) + h2(theta)  [since h2(-theta) = h2(theta) for centrally symmetric]
        # Wait, for a centrally symmetric body, h(-d) = h(d). So:
        # min_dist = max_theta h1(theta) + h2(theta)
        
        # We can sample theta at critical angles: the vertex directions of both hexagons
        # and the edge midpoints
        best_dist = 0.0
        # Critical directions: vertices of hex1, vertices of hex2, and midpoints
        for k in range(6):
            # Vertex direction of hex1
            theta = theta1 + k * math.pi / 3
            d = support_hex(theta, theta1) + support_hex(theta, theta2)
            best_dist = max(best_dist, d)
            # Vertex direction of hex2
            theta = theta2 + k * math.pi / 3
            d = support_hex(theta, theta1) + support_hex(theta, theta2)
            best_dist = max(best_dist, d)
            # Midpoint between consecutive vertices of hex1
            theta = theta1 + (k + 0.5) * math.pi / 3
            d = support_hex(theta, theta1) + support_hex(theta, theta2)
            best_dist = max(best_dist, d)
            # Midpoint between consecutive vertices of hex2
            theta = theta2 + (k + 0.5) * math.pi / 3
            d = support_hex(theta, theta1) + support_hex(theta, theta2)
            best_dist = max(best_dist, d)
        
        return best_dist
    
    def constraints_func(positions, angles):
        """
        Return constraint values for non-overlap.
        For each pair (i,j): dist(p_i, p_j) - min_dist(theta_i, theta_j) >= 0
        """
        cons = []
        for i in range(N):
            for j in range(i + 1, N):
                d = positions[i] - positions[j]
                dist = np.linalg.norm(d)
                min_d = min_distance_for_non_overlap(angles[i], angles[j])
                cons.append(dist - min_d)
        return cons
    
    # 1. Fast combinatorial search over lattice positions with angle 0
    from itertools import combinations
    
    # Precompute L for each center with angle 0
    center_Ls = []
    for i in range(19):
        x, y, _, _ = all_centers[i]
        L_h = min_L_for_hexagon(x, y, 0.0)
        center_Ls.append(L_h)
    
    best_L_init = float('inf')
    best_subset = None
    
    for subset in combinations(range(19), 12):
        L = max(center_Ls[i] for i in subset)
        if L < best_L_init:
            best_L_init = L
            best_subset = subset
    
    if best_subset is None:
        best_subset = tuple(range(12))
    
    # Initialize positions and angles
    x0 = np.zeros((N, 2))
    angles0 = np.zeros(N)
    for k, idx in enumerate(best_subset):
        x0[k, 0] = all_centers[idx][0]
        x0[k, 1] = all_centers[idx][1]
    
    best_positions = x0.copy()
    best_angles = angles0.copy()
    best_L = compute_L(best_positions, best_angles)
    
    # 2. Continuous optimization with SLSQP
    def objective(x_flat):
        pos = x_flat[:2*N].reshape(N, 2)
        ang = x_flat[2*N:3*N]
        return compute_L(pos, ang)
    
    def constraints(x_flat):
        pos = x_flat[:2*N].reshape(N, 2)
        ang = x_flat[2*N:3*N]
        vals = constraints_func(pos, ang)
        return vals
    
    # Bounds
    bounds = [(-4, 4) for _ in range(2*N)] + [(-math.pi, math.pi) for _ in range(N)]
    
    try:
        if time.time() < deadline - 10.0:
            x_flat0 = np.concatenate([x0.flatten(), angles0])
            res = minimize(objective, x_flat0, method='SLSQP', 
                           bounds=bounds, 
                           constraints={'type': 'ineq', 'fun': constraints},
                           options={'maxiter': 500, 'ftol': 1e-10})
            if res.fun < best_L:
                best_positions = res.x[:2*N].reshape(N, 2)
                best_angles = res.x[2*N:3*N]
                best_L = res.fun
    except Exception:
        pass
    
    # 3. Try a few different initial configurations with random perturbations
    for trial in range(3):
        if time.time() > deadline - 15.0:
            break
        try:
            # Perturb the best solution
            x_flat0 = np.concatenate([best_positions.flatten(), best_angles])
            noise = np.random.normal(0, 0.1, size=x_flat0.shape)
            x_flat0 += noise
            
            res = minimize(objective, x_flat0, method='SLSQP', 
                           bounds=bounds, 
                           constraints={'type': 'ineq', 'fun': constraints},
                           options={'maxiter': 300, 'ftol': 1e-10})
            if res.fun < best_L:
                best_positions = res.x[:2*N].reshape(N, 2)
                best_angles = res.x[2*N:3*N]
                best_L = res.fun
        except Exception:
            pass
    
    # Construct inner hexagons
    inner = []
    for i in range(N):
        x, y = best_positions[i]
        angle_deg = math.degrees(best_angles[i]) % 360
        inner.append([x, y, angle_deg])
    
    # Calculate final L
    final_L = compute_L(best_positions, best_angles)
    
    return inner, (0.0, 0.0), final_L, 0.0
# EVOLVE-BLOCK-END
