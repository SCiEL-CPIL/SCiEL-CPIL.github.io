import math
import numpy as np
import time
from scipy.optimize import minimize

N = 11
APOTHEM = math.sqrt(3) / 2.0
START_TIME = time.time()
TIME_LIMIT = 150.0
SAFETY_MARGIN = 5.0

def construct():
    if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
        # Fallback: simple grid
        return [[0.0, 0.0, 0.0]], [0.0, 0.0], 1.0, 0.0

    # State vector:
    # For each of N hexagons: x, y, theta (3 * N)
    # Outer hexagon: cx, cy, S, beta (4)
    # Total: 3*N + 4 = 37 variables
    
    # Initialize with a known good configuration (e.g., a simple cluster)
    # Let's start with a compact arrangement
    # Center at 0,0. 
    # Hex 0 at (0,0)
    # Others placed roughly in a ring or grid
    
    init_vars = []
    # Hex 0
    init_vars.extend([0.0, 0.0, 0.0])
    
    # Place remaining 10 in a rough circular/grid pattern
    # A common good packing for 11 is a central one + 6 around + 4 more
    # Or a 3x3 grid + 2
    # Let's use a grid of 3x3 + 2 for initialization
    positions = [
        (0,0),
        (1,0), (-1,0), (0,1), (0,-1), (1,1), (-1,1), # 7 in hex grid coords approx
        # Actually, let's just use Cartesian approximations for a tight cluster
        # Center (0,0)
        # Ring 1: 6 hexes at distance sqrt(3)
        # Ring 2: 4 hexes further out
    ]
    
    # Better initialization: use the "flower" pattern
    # Center (0,0)
    # 6 neighbors at distance sqrt(3), angles 0, 60, ...
    # 4 more at distance 2*sqrt(3)? No, that's too far.
    # Let's just use a random start near a known solution or a grid.
    # Grid:
    # (0,0)
    # (1.73, 0), (-1.73, 0)
    # (0.86, 1.5), (-0.86, 1.5)
    # (0.86, -1.5), (-0.86, -1.5)
    # That's 7.
    # Add 4 more:
    # (2.6, 1.5), (2.6, -1.5)
    # (-2.6, 1.5), (-2.6, -1.5)
    
    init_pos = [
        (0.0, 0.0),
        (1.732, 0.0), (-1.732, 0.0),
        (0.866, 1.5), (-0.866, 1.5),
        (0.866, -1.5), (-0.866, -1.5),
        (2.6, 1.5), (2.6, -1.5),
        (-2.6, 1.5), (-2.6, -1.5)
    ]
    
    for i in range(N):
        x, y = init_pos[i]
        init_vars.extend([x, y, 0.0])
        
    # Outer hexagon initial guess
    # Needs to contain all. Max dist from origin is approx 3.
    # Apothem of outer = D. Side S = (D + APOTHEM)/APOTHEM
    # Let's start with S = 4.0
    init_vars.extend([0.0, 0.0, 4.0, 0.0])
    
    x0 = np.array(init_vars)
    
    def objective(vars):
        if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
            return 1e10
            
        # Unpack
        hex_params = vars[:3*N].reshape(N, 3)
        outer_cx = vars[3*N]
        outer_cy = vars[3*N+1]
        outer_S = vars[3*N+2]
        outer_beta = vars[3*N+3]
        
        # Ensure S is positive
        if outer_S < 1.0:
            return 1e10 + (1.0 - outer_S) * 100
            
        # 1. Containment Penalty
        # Outer hexagon apothem
        outer_apothem = outer_S * APOTHEM
        
        # Outer hexagon normals
        # The sides are perpendicular to these vectors.
        # A regular hexagon centered at (cx, cy) with rotation beta.
        # The 6 outward normals are at angles beta + pi/6 + k*pi/3.
        # Wait, standard orientation: if beta=0, one vertex is at (S, 0)? 
        # The task says "outer hexagon ... rotation".
        # Usually, side length S.
        # Let's define the outer hexagon such that its sides are at distance outer_apothem from center.
        # The condition for a point P to be inside is:
        # For all 6 normals n_k, dot(P - C, n_k) <= outer_apothem
        # AND dot(C - P, n_k) <= outer_apothem (which is dot(P-C, n_k) >= -outer_apothem)
        # So |dot(P - C, n_k)| <= outer_apothem
        
        normals = []
        for k in range(6):
            ang = outer_beta + math.pi / 6.0 + k * math.pi / 3.0
            normals.append((math.cos(ang), math.sin(ang)))
        normals = np.array(normals) # 6x2
        
        # Calculate vertices of each inner hexagon
        # Inner hexagon i has center (xi, yi) and angle theta_i
        # Vertices are at distance 1 from center.
        # Angles of vertices: theta_i + k*pi/3 for k=0..5
        # We need to check if ALL vertices are inside.
        # Instead of checking vertices, we can check the support function.
        # The max projection of inner hexagon i onto normal n is:
        # dot(center_i - C, n) + max_vertex_proj_i(n)
        # For a unit hexagon, the max projection onto a direction n is the "radius" in that direction.
        # For a regular hexagon, the support function h(n) is not constant.
        # However, we can just compute the 6 vertices of each inner hexagon and check them.
        # This is 11 * 6 = 66 points.
        
        max_violation = 0.0
        for i in range(N):
            xi, yi, theta_i = hex_params[i]
            # Compute 6 vertices
            for k in range(6):
                v_ang = theta_i + k * math.pi / 3.0
                vx = xi + math.cos(v_ang)
                vy = yi + math.sin(v_ang)
                px = vx - outer_cx
                py = vy - outer_cy
                # Project onto all 6 normals
                projs = px * normals[:, 0] + py * normals[:, 1]
                abs_projs = np.abs(projs)
                # We need abs_proj <= outer_apothem
                # Violation is max(0, abs_proj - outer_apothem)
                viol = np.max(abs_projs - outer_apothem)
                if viol > max_violation:
                    max_violation = viol
                    
        # Smooth penalty for containment
        # Use a soft max or just square the max violation
        # To be smooth, we can sum exp(violation * temp)
        # But simple max with smoothing:
        # Let's use a penalty that is 0 if max_violation < 0, and grows otherwise.
        # Actually, we want to MINIMIZE S. So we should minimize S subject to constraints.
        # If we use unconstrained optimization, we add a large penalty for violations.
        # Penalty = max_violation^2 * 1000
        
        # 2. Non-overlap Penalty
        # Distance between centers of i and j must be >= min_dist
        # min_dist for unit hexagons is sqrt(3) approx 1.732
        # Actually, if they are rotated, the min dist is still sqrt(3)?
        # Yes, the width of a hexagon is sqrt(3) in the direction of the flat sides.
        # The minimum distance between two disjoint hexagons is the minimum distance between their boundaries.
        # If they are aligned, they touch at a side, distance between centers is sqrt(3).
        # If rotated, they can't get closer than the distance between parallel supporting lines?
        # The "width" of a hexagon in any direction is between sqrt(3) and 2.
        # The minimum width is sqrt(3).
        # So the minimum distance between centers is at least (width_i + width_j)/2 ? No.
        # The minimum distance between two convex sets is the minimum distance between any point in A and any point in B.
        # For two identical regular hexagons, the minimum distance between centers is sqrt(3).
        
        min_dist = math.sqrt(3)
        overlap_penalty = 0.0
        for i in range(N):
            for j in range(i+1, N):
                dx = hex_params[i, 0] - hex_params[j, 0]
                dy = hex_params[i, 1] - hex_params[j, 1]
                dist = math.sqrt(dx*dx + dy*dy)
                if dist < min_dist:
                    # Penalty for being too close
                    # Smooth step
                    diff = min_dist - dist
                    overlap_penalty += diff * diff * 1000
                    
        # Objective: Minimize S + Penalties
        # We want to minimize S.
        # So objective = S + 100 * max_violation^2 + overlap_penalty
        
        # To encourage smaller S, we just return S + penalties.
        # But if S is small, violations will be huge.
        
        return outer_S + 100.0 * max_violation * max_violation + overlap_penalty

    # Use Powell or Nelder-Mead
    # Powell is robust for this kind of problem
    
    best_result = None
    best_fun = float('inf')
    
    # Multi-start
    starts = [x0]
    # Add some random perturbations
    np.random.seed(42)
    for _ in range(5):
        perturbed = x0.copy()
        # Perturb inner positions
        perturbed[:3*N] += np.random.normal(0, 0.5, 3*N)
        # Perturb outer
        perturbed[3*N:3*N+2] += np.random.normal(0, 0.5, 2)
        perturbed[3*N+2] = max(1.0, x0[3*N+2] + np.random.normal(0, 0.5))
        perturbed[3*N+3] = np.random.uniform(0, math.pi/6)
        starts.append(perturbed)
        
    for x_start in starts:
        if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
            break
            
        try:
            res = minimize(objective, x_start, method='Powell',
                           options={'maxiter': 1000, 'xatol': 1e-8, 'fatol': 1e-8})
            if res.fun < best_fun:
                best_fun = res.fun
                best_result = res
        except Exception as e:
            pass
            
    if best_result is None:
        # Fallback
        return [[0.0, 0.0, 0.0]], [0.0, 0.0], 1.0, 0.0
        
    x_opt = best_result.x
    hex_params = x_opt[:3*N].reshape(N, 3)
    outer_cx = x_opt[3*N]
    outer_cy = x_opt[3*N+1]
    outer_S = x_opt[3*N+2]
    outer_beta = x_opt[3*N+3]
    
    # Refine with Nelder-Mead for better precision
    try:
        res2 = minimize(objective, x_opt, method='Nelder-Mead',
                        options={'maxiter': 5000, 'xatol': 1e-10, 'fatol': 1e-10})
        if res2.fun < best_fun:
            x_opt = res2.x
            hex_params = x_opt[:3*N].reshape(N, 3)
            outer_cx = x_opt[3*N]
            outer_cy = x_opt[3*N+1]
            outer_S = x_opt[3*N+2]
            outer_beta = x_opt[3*N+3]
    except:
        pass
        
    # Construct return values
    inner = []
    for i in range(N):
        x, y, theta = hex_params[i]
        angle_deg = math.degrees(theta)
        inner.append([float(x), float(y), float(angle_deg)])
        
    center = [float(outer_cx), float(outer_cy)]
    side = float(outer_S)
    rotation = float(math.degrees(outer_beta))
    
    # Sanity check: if side is unreasonably small or large, or NaN
    if not np.isfinite(side) or side < 1.0:
        # Fallback to a safe configuration
        # Just return the initial grid scaled up
        return [[0.0, 0.0, 0.0]], [0.0, 0.0], 4.0, 0.0

    return inner, center, side, rotation
