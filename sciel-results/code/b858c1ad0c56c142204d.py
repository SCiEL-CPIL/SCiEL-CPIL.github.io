import math
import time
import numpy as np
from scipy.optimize import minimize

N = 11

def construct():
    start_time = time.time()
    deadline = 140.0
    safety_margin = 5.0

    d = math.sqrt(3)

    # Unit hexagon vertices (flat-top, centered at origin)
    # Angle 0 is (1, 0)
    verts_rel = np.array([[math.cos(i * math.pi / 3), math.sin(i * math.pi / 3)] for i in range(6)])

    def get_hex_verts(cx, cy, angle):
        c = math.cos(angle)
        s = math.sin(angle)
        rx = verts_rel[:, 0] * c - verts_rel[:, 1] * s
        ry = verts_rel[:, 0] * s + verts_rel[:, 1] * c
        return np.column_stack([rx + cx, ry + cy])

    def compute_score(params):
        """
        params: [x1..x11, y1..y11, theta1..theta11, alpha, cx_out, cy_out]
        Returns: outer_side_length + overlap_penalty
        """
        if time.time() > deadline - safety_margin:
            return 1e10
            
        # Unpack
        xs = params[0:N]
        ys = params[N:2*N]
        thetas = params[2*N:3*N]
        alpha = params[3*N]
        cx_out = params[3*N + 1]
        cy_out = params[3*N + 2]

        # Compute vertices for all inner hexagons
        all_verts = []
        for i in range(N):
            v = get_hex_verts(xs[i], ys[i], thetas[i])
            all_verts.append(v)
        
        # Stack all vertices (66 points)
        all_pts = np.vstack(all_verts)
        
        # Compute extents in 6 directions relative to outer center
        k_angles = alpha + np.arange(6) * math.pi / 3
        normals = np.stack([np.cos(k_angles), np.sin(k_angles)], axis=-1)
        
        # Vectorized dot product
        # all_pts shape: (66, 2)
        # normals shape: (6, 2)
        # dots shape: (6, 66)
        dots = np.sum((all_pts - np.array([cx_out, cy_out]))[np.newaxis, :, :] * normals[:, np.newaxis, :], axis=-1)
        
        extents = np.max(dots, axis=1) - np.min(dots, axis=1)
        max_extent = np.max(extents)
        
        # Side length S
        # For a regular hexagon, distance between opposite sides is S * sqrt(3)
        # So max_extent = S * sqrt(3)
        S = max_extent / math.sqrt(3)
        
        # Overlap Penalty
        # Check pairwise distances between centers
        # If dist < sqrt(3), they overlap (assuming same orientation, which is close to true)
        # For robustness, we use a simple distance penalty
        penalty = 0.0
        for i in range(N):
            for j in range(i + 1, N):
                dx = xs[i] - xs[j]
                dy = ys[i] - ys[j]
                dist = math.sqrt(dx*dx + dy*dy)
                min_dist = d # sqrt(3)
                if dist < min_dist:
                    penalty += (min_dist - dist) ** 2
                    
        return S + penalty * 100.0

    # Initial Configuration: Hexagonal Lattice
    # 1 Center, 6 Ring 1, 4 Ring 2
    # Ring 2 selection: "Alternating 4" for C2 symmetry
    # Angles for Ring 2 corners: 30, 90, 210, 270 degrees
    
    init_params = np.zeros(N * 3 + 3)
    
    # Center
    init_params[0] = 0.0
    init_params[1] = 0.0
    init_params[2] = 0.0
    
    # Ring 1
    for i in range(6):
        angle = i * math.pi / 3
        x = d * math.cos(angle)
        y = d * math.sin(angle)
        idx = i + 1
        init_params[idx*3] = x
        init_params[idx*3+1] = y
        init_params[idx*3+2] = 0.0
        
    # Ring 2 (4 hexagons)
    # Indices 7, 8, 9, 10
    ring2_angles_deg = [30, 90, 210, 270]
    for i, ang_deg in enumerate(ring2_angles_deg):
        angle = math.radians(ang_deg)
        x = 2 * d * math.cos(angle)
        y = 2 * d * math.sin(angle)
        idx = 7 + i
        init_params[idx*3] = x
        init_params[idx*3+1] = y
        init_params[idx*3+2] = 0.0
        
    # Outer hexagon params
    init_params[3*N] = 0.0 # alpha
    init_params[3*N+1] = 0.0 # cx_out
    init_params[3*N+2] = 0.0 # cy_out

    bounds = [(-10, 10), (-10, 10), (-math.pi, math.pi)] * N
    bounds.append((0, math.pi/3)) # alpha
    bounds.append((-10, 10)) # cx_out
    bounds.append((-10, 10)) # cy_out

    best_score = compute_score(init_params)
    best_params = init_params.copy()

    # Try a few different initial rotations for the outer hexagon
    for alpha_init in [0.0, math.pi/6, math.pi/12, math.pi/4]:
        if time.time() > deadline - safety_margin:
            break
            
        p = init_params.copy()
        p[3*N] = alpha_init
        
        # Local optimization
        try:
            res = minimize(compute_score, p, method='L-BFGS-B', bounds=bounds, 
                           options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-8})
            if res.fun < best_score:
                best_score = res.fun
                best_params = res.x.copy()
        except Exception:
            pass

    # If time permits, try a different Ring 2 selection
    if time.time() < deadline - safety_margin:
        # Try "Consecutive 4": 30, 90, 150, 210
        init_params2 = init_params.copy()
        ring2_angles_deg2 = [30, 90, 150, 210]
        for i, ang_deg in enumerate(ring2_angles_deg2):
            angle = math.radians(ang_deg)
            x = 2 * d * math.cos(angle)
            y = 2 * d * math.sin(angle)
            idx = 7 + i
            init_params2[idx*3] = x
            init_params2[idx*3+1] = y
            init_params2[idx*3+2] = 0.0
            
        for alpha_init in [0.0, math.pi/6]:
            if time.time() > deadline - safety_margin:
                break
            p = init_params2.copy()
            p[3*N] = alpha_init
            try:
                res = minimize(compute_score, p, method='L-BFGS-B', bounds=bounds, 
                               options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-8})
                if res.fun < best_score:
                    best_score = res.fun
                    best_params = res.x.copy()
            except Exception:
                pass

    # Extract results
    inner = []
    for i in range(N):
        x = best_params[i*3]
        y = best_params[i*3+1]
        angle_deg = math.degrees(best_params[i*3+2])
        inner.append([x, y, angle_deg])
        
    center = (best_params[3*N+1], best_params[3*N+2])
    side = best_score
    angle = math.degrees(best_params[3*N])
    
    return inner, center, side, angle

