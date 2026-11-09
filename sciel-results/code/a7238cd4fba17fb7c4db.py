import numpy as np
import time
from scipy.optimize import minimize

N = 11
START_TIME = time.time()
TIME_LIMIT = 140.0 # Safety margin for 150s limit

def hex_radius(phi, theta):
    """
    Calculate the distance from the center to the boundary of a unit regular hexagon
    in the direction phi, given the hexagon is rotated by theta.
    """
    apothem = np.sqrt(3) / 2.0
    # Normals of the sides are at theta + pi/6 + k*pi/3
    # We need the normal closest to phi
    # phi is the direction of the ray from center
    # The side that the ray intersects is the one whose normal is closest to phi
    # because the hexagon is convex and symmetric.
    
    # Calculate angles of normals
    normals = theta + np.pi/6 + np.arange(6) * np.pi/3
    
    # Calculate angular difference between phi and each normal
    # We want the smallest absolute difference
    diffs = np.abs((phi - normals + np.pi) % (2 * np.pi) - np.pi)
    
    # Find the minimum difference
    min_diff = np.min(diffs)
    
    # The distance is apothem / cos(min_diff)
    # min_diff is always <= pi/6 (30 degrees) for a hexagon
    return apothem / np.cos(min_diff)

def penalty(params):
    """
    Objective function to minimize.
    params: [x0, y0, th0, ..., x10, y10, th10, L]
    """
    L = params[-1]
    positions = params[:-1].reshape(N, 3)
    
    # 1. Containment Penalty
    # Outer hexagon side L, rotation 0.
    outer_apothem = L * np.sqrt(3) / 2.0
    outer_normals = np.arange(6) * np.pi / 3
    
    pen = 0.0
    
    # Check vertices of inner hexagons
    for i in range(N):
        x_i, y_i, th_i = positions[i]
        # Vertices of unit hexagon are at distance 1 from center
        v_angles = th_i + np.arange(6) * np.pi / 3
        vx = x_i + np.cos(v_angles)
        vy = y_i + np.sin(v_angles)
        
        for j in range(6):
            vx_j = vx[j]
            vy_j = vy[j]
            # Check against outer hex inequalities: n_k . p <= outer_apothem
            for k in range(6):
                nx = np.cos(outer_normals[k])
                ny = np.sin(outer_normals[k])
                val = nx * vx_j + ny * vy_j
                if val > outer_apothem:
                    pen += (val - outer_apothem) ** 2
    
    # 2. Overlap Penalty
    for i in range(N):
        for j in range(i + 1, N):
            x1, y1, th1 = positions[i]
            x2, y2, th2 = positions[j]
            dx = x2 - x1
            dy = y2 - y1
            d = np.sqrt(dx * dx + dy * dy)
            if d < 1e-6:
                pen += 100.0
                continue
            psi = np.arctan2(dy, dx)
            # Required distance to avoid overlap
            r1 = hex_radius(psi, th1)
            r2 = hex_radius(psi + np.pi, th2)
            req = r1 + r2
            if d < req:
                pen += (req - d) ** 2
    
    # We want to minimize L, but penalize violations heavily
    # If penalty is 0, we just minimize L
    return L + 50.0 * pen

def construct():
    best_L = float('inf')
    best_params = None
    
    # Generate initial guesses
    inits = []
    
    # 1. Honeycomb-like
    # Center + 6 neighbors + 4 extra
    pos = []
    pos.append([0.0, 0.0, 0.0])
    for k in range(6):
        angle = k * np.pi / 3
        pos.append([2.0 * np.cos(angle), 2.0 * np.sin(angle), 0.0])
    # Add 4 more in the "gaps" or further out
    for k in range(4):
        angle = k * np.pi / 2 + np.pi / 4 # 45, 135, 225, 315
        pos.append([3.5 * np.cos(angle), 3.5 * np.sin(angle), 0.0])
    params_init = np.array(pos).flatten()
    params_init = np.append(params_init, 10.0) # Initial L
    inits.append(params_init)
    
    # 2. Random
    np.random.seed(42)
    for _ in range(5):
        pos = []
        for _ in range(N):
            r = np.random.uniform(0, 4)
            a = np.random.uniform(0, 2 * np.pi)
            th = np.random.uniform(0, np.pi / 3) # Symmetry
            pos.append([r * np.cos(a), r * np.sin(a), th])
        params_init = np.array(pos).flatten()
        params_init = np.append(params_init, 10.0)
        inits.append(params_init)
        
    # 3. Grid-like
    pos = []
    # 3x4 grid approx
    for i in range(3):
        for j in range(4):
            if i == 1 and j == 3: continue # Skip one to make 11? 3*4=12.
            x = (j - 1.5) * 2.2
            y = (i - 1) * 2.2
            th = 0.0 if i % 2 == 0 else np.pi / 6
            pos.append([x, y, th])
    if len(pos) > 11: pos = pos[:11]
    while len(pos) < 11:
        pos.append([0.0, 0.0, 0.0])
    params_init = np.array(pos).flatten()
    params_init = np.append(params_init, 10.0)
    inits.append(params_init)

    # Optimization
    for init in inits:
        if time.time() - START_TIME > TIME_LIMIT:
            break
            
        try:
            res = minimize(penalty, init, method='Nelder-Mead', 
                           options={'maxiter': 2000, 'xatol': 1e-6, 'fatol': 1e-6})
            
            if res.fun < best_L:
                # Check if penalty is low enough (valid solution)
                # res.fun = L + 50 * pen. If pen is high, L is effectively high.
                # We should check the actual penalty separately to ensure validity
                p = res.x
                L_val = p[-1]
                # Re-evaluate penalty to check validity
                pos = p[:-1].reshape(N, 3)
                outer_apothem = L_val * np.sqrt(3) / 2.0
                outer_normals = np.arange(6) * np.pi / 3
                valid = True
                for i in range(N):
                    x_i, y_i, th_i = pos[i]
                    v_angles = th_i + np.arange(6) * np.pi / 3
                    vx = x_i + np.cos(v_angles)
                    vy = y_i + np.sin(v_angles)
                    for j in range(6):
                        for k in range(6):
                            nx = np.cos(outer_normals[k])
                            ny = np.sin(outer_normals[k])
                            if nx * vx[j] + ny * vy[j] > outer_apothem + 1e-5:
                                valid = False
                                break
                        if not valid: break
                    if not valid: break
                if not valid: continue
                
                # Check overlap
                for i in range(N):
                    for j in range(i+1, N):
                        x1, y1, th1 = pos[i]
                        x2, y2, th2 = pos[j]
                        dx = x2 - x1
                        dy = y2 - y1
                        d = np.sqrt(dx*dx + dy*dy)
                        if d < 1e-6:
                            valid = False
                            break
                        psi = np.arctan2(dy, dx)
                        r1 = hex_radius(psi, th1)
                        r2 = hex_radius(psi + np.pi, th2)
                        if d < r1 + r2 - 1e-5:
                            valid = False
                            break
                    if not valid: break
                
                if valid:
                    best_L = L_val
                    best_params = p
                    
        except Exception:
            pass

    if best_params is None:
        # Fallback: Return a very large L with non-overlapping hexes
        inner = [[4.0 * k, 0.0, 0.0] for k in range(N)]
        return inner, (0.0, 0.0), 40.0, 0.0

    # Extract results
    p = best_params
    L = p[-1]
    positions = p[:-1].reshape(N, 3)
    
    inner = []
    for i in range(N):
        x, y, th_rad = positions[i]
        th_deg = np.degrees(th_rad) % 360.0
        inner.append([x, y, th_deg])
        
    return inner, (0.0, 0.0), L, 0.0

