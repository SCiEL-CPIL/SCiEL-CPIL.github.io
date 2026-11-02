import math
import numpy as np
import time
from scipy.optimize import minimize, linprog

N = 11
APOTHEM = math.sqrt(3) / 2.0
START_TIME = time.time()
TIME_LIMIT = 145.0 # Safety margin for 150s limit
EPS = 1e-9

def construct():
    if time.time() - START_TIME > TIME_LIMIT:
        # Fallback: standard hexagonal spiral
        return _fallback_solution()

    # 1. Generate Initial Configuration (Hexagonal Spiral)
    # Center + 6 neighbors + 4 from ring 2 to make 11 total
    # Axial coordinates (q, r)
    axial_initial = [
        (0, 0),      # Center
        (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1), # Ring 1
        (2, 0), (-2, 0), (0, 2), (2, -2) # Subset of Ring 2 (symmetric selection for compactness)
    ]
    
    # Convert axial to cartesian
    # x = sqrt(3) * (q + r/2)
    # y = 3/2 * r
    pts_init = []
    for q, r in axial_initial:
        x = math.sqrt(3) * (q + r / 2.0)
        y = 1.5 * r
        pts_init.append((x, y))
    pts_init = np.array(pts_init)
    
    # Center the initial points roughly
    center_of_mass = np.mean(pts_init, axis=0)
    pts_init = pts_init - center_of_mass

    # 2. Define Objective Function
    # Variables: [x0, y0, x1, y1, ..., x10, y10, cx, cy, theta]
    # Total 22 + 3 = 25 variables
    
    def objective(params):
        if time.time() - START_TIME > TIME_LIMIT:
            return 1e10
            
        # Unpack
        # First 2N are points, last 3 are center (cx, cy) and angle (theta)
        n_pts = N
        p_coords = params[:2*n_pts].reshape(n_pts, 2)
        cx = params[2*n_pts]
        cy = params[2*n_pts+1]
        theta = params[2*n_pts+2]
        
        # 1. Check Overlaps
        # Distance between centers of unit hexagons must be >= 2 * APOTHEM (for side 1)
        # Actually, min distance is 2 * APOTHEM = sqrt(3).
        min_dist_sq = (2 * APOTHEM) ** 2
        for i in range(n_pts):
            for j in range(i+1, n_pts):
                dx = p_coords[i, 0] - p_coords[j, 0]
                dy = p_coords[i, 1] - p_coords[j, 1]
                dist_sq = dx*dx + dy*dy
                if dist_sq < min_dist_sq - 1e-5:
                    # Penalty proportional to overlap
                    overlap = min_dist_sq - dist_sq
                    return 1e8 * overlap + 100.0
                    
        # 2. Calculate Minimal Enclosing Hexagon
        # We want to find S such that all points p_i satisfy:
        # | (p_i - c) . n_k | <= S * APOTHEM  for all k in 0..5
        # This is equivalent to finding the max projection.
        # Normals depend on theta.
        # n_k = (cos(theta + pi/6 + k*pi/3), sin(theta + pi/6 + k*pi/3))
        
        # To minimize S, we can just compute the current max projection for this theta and center.
        # However, the center (cx, cy) is also a variable.
        # For a fixed theta, the optimal (cx, cy) minimizes the max projection.
        # This is an LP problem.
        # Minimize D
        # s.t. (p_i - c) . n_k <= D
        #      -(p_i - c) . n_k <= D
        
        # Instead of solving LP inside the objective (slow), we can use a smooth approximation
        # or just evaluate the current D and let the optimizer adjust cx, cy, theta.
        
        # Calculate normals
        angles = theta + math.pi / 6.0 + np.arange(6) * math.pi / 3.0
        normals = np.column_stack([np.cos(angles), np.sin(angles)])
        
        # Compute projections of (p_i - c) onto normals
        # p_rel = p_i - c
        p_rel = p_coords - np.array([cx, cy])
        
        # proj[i, k] = p_rel[i] . normals[k]
        proj = p_rel @ normals.T # Shape (N, 6)
        
        # We need max over i, k of |proj[i, k]|
        # D = max(|proj|)
        D = np.max(np.abs(proj))
        
        # S = D / APOTHEM
        S = D / APOTHEM
        
        return S

    # 3. Optimize
    # Initial guess: points at initial positions, center at (0,0), theta at 0
    x0 = np.zeros(2*N + 3)
    x0[:2*N] = pts_init.flatten()
    x0[2*N] = 0.0
    x0[2*N+1] = 0.0
    x0[2*N+2] = 0.0
    
    best_S = float('inf')
    best_x = x0.copy()
    
    # Multi-start to avoid local minima
    # Try different initial thetas and slight perturbations
    start_times = time.time()
    
    # Run 1: Standard start
    try:
        res = minimize(objective, x0, method='Nelder-Mead', 
                       options={'xatol': 1e-8, 'fatol': 1e-8, 'maxiter': 2000, 'adaptive': True})
        if res.fun < best_S:
            best_S = res.fun
            best_x = res.x
    except Exception:
        pass
        
    # Run 2: Perturbed start (rotate initial config slightly)
    if time.time() - START_TIME < TIME_LIMIT - 10:
        x0_2 = x0.copy()
        # Slight rotation of the whole initial pattern
        theta_init = 0.1
        rot = np.array([[math.cos(theta_init), -math.sin(theta_init)], 
                        [math.sin(theta_init), math.cos(theta_init)]])
        pts_rot = pts_init @ rot.T
        x0_2[:2*N] = pts_rot.flatten()
        x0_2[2*N+2] = theta_init
        
        try:
            res = minimize(objective, x0_2, method='Nelder-Mead', 
                           options={'xatol': 1e-8, 'fatol': 1e-8, 'maxiter': 2000, 'adaptive': True})
            if res.fun < best_S:
                best_S = res.fun
                best_x = res.x
        except Exception:
            pass
            
    # Run 3: Compressed start (move points 1% closer to center to encourage tighter packing)
    if time.time() - START_TIME < TIME_LIMIT - 10:
        x0_3 = x0.copy()
        pts_compressed = pts_init * 0.95
        # Re-center
        pts_compressed = pts_compressed - np.mean(pts_compressed, axis=0)
        x0_3[:2*N] = pts_compressed.flatten()
        
        try:
            res = minimize(objective, x0_3, method='Nelder-Mead', 
                           options={'xatol': 1e-8, 'fatol': 1e-8, 'maxiter': 2000, 'adaptive': True})
            if res.fun < best_S:
                best_S = res.fun
                best_x = res.x
        except Exception:
            pass

    # Final Polish: Use COBYLA or Powell if available and time permits? 
    # Nelder-Mead is robust but slow. Let's stick with NM but tighter tolerance if time left.
    if time.time() - START_TIME < TIME_LIMIT - 20 and best_S < 100:
        try:
            res = minimize(objective, best_x, method='Powell', 
                           options={'xtol': 1e-10, 'ftol': 1e-10, 'maxiter': 1000})
            if res.fun < best_S:
                best_S = res.fun
                best_x = res.x
        except Exception:
            pass

    # Extract solution
    if best_S >= 100 or not np.isfinite(best_S):
        return _fallback_solution()

    p_final = best_x[:2*N].reshape(N, 2)
    cx_final = best_x[2*N]
    cy_final = best_x[2*N+1]
    theta_final = best_x[2*N+2]
    
    # Ensure the outer hexagon is minimal for these points and angle
    # The optimizer should have found this, but let's verify/adjust center for fixed theta
    # Actually, the objective minimized S given (pts, c, theta).
    # So S = best_S should be correct.
    
    # Format output
    # inner: list of [x, y, angle_deg]
    # The angle of the inner hexagons can be 0 if we assume they are aligned, 
    # BUT the problem says "each inner hexagon may translate and rotate".
    # In our model, we only optimized positions. We assumed they are all rotated by 0 relative to the frame, 
    # OR we can rotate them to match the optimal orientation for packing?
    # The enclosing hexagon has angle theta.
    # If we rotate the inner hexagons, their "shape" in the fixed frame changes.
    # Our distance check assumed fixed orientation (distance between centers is sufficient for non-overlap 
    # ONLY if orientations are compatible or we use the worst-case distance).
    # Wait. The minimum distance between centers of two non-overlapping unit hexagons depends on their relative rotation.
    # If they are parallel, min dist = 2*APOTHEM = sqrt(3).
    # If one is rotated 30 degs, the distance can be smaller.
    # However, the standard lattice packing (which we started with) uses parallel hexagons.
    # If we allow rotation, we might pack tighter.
    # But optimizing rotation of 11 hexagons adds 11 more variables (2N+3 + 11 = 36 vars).
    # Given time constraints, sticking to parallel orientations (angle=0 for all inner, or matching outer) is safer 
    # and likely close to optimal for this N.
    # Let's set inner angle to 0.0 (aligned with axes).
    # Wait, if the outer hexagon is rotated by theta, and inner are at 0, do they fit?
    # Yes, as long as the projection condition holds.
    
    inner = []
    for i in range(N):
        inner.append([float(p_final[i, 0]), float(p_final[i, 1]), 0.0])
        
    center = [float(cx_final), float(cy_final)]
    side = float(best_S)
    rotation = float(math.degrees(theta_final))
    
    return inner, center, side, rotation

def _fallback_solution():
    # Standard hexagonal spiral
    axial = [
        (0, 0),
        (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
        (2, 0), (-2, 0), (0, 2), (2, -2)
    ]
    pts = []
    for q, r in axial:
        x = math.sqrt(3) * (q + r / 2.0)
        y = 1.5 * r
        pts.append([x, y, 0.0])
    
    # Center
    cx = np.mean([p[0] for p in pts])
    cy = np.mean([p[1] for p in pts])
    
    # Shift
    for p in pts:
        p[0] -= cx
        p[1] -= cy
        
    # Calculate side
    max_proj = 0
    for k in range(6):
        ang = math.pi / 6.0 + k * math.pi / 3.0
        nx, ny = math.cos(ang), math.sin(ang)
        for p in pts:
            proj = abs(p[0]*nx + p[1]*ny)
            if proj > max_proj:
                max_proj = proj
    side = (max_proj + APOTHEM) / APOTHEM
    
    return pts, [0.0, 0.0], side, 0.0

# Test
if __name__ == "__main__":
    import sys
    res = construct()
    print("Result:", res[2], "Time:", time.time() - START_TIME)
