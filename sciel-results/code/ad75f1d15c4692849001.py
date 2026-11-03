import math
import numpy as np
import time
from scipy.optimize import minimize

N = 11
APOTHEM = math.sqrt(3) / 2.0
MIN_DIST = math.sqrt(3)
START_TIME = time.time()
TIME_LIMIT = 150.0
SAFETY_MARGIN = 5.0

def construct():
    if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
        return [[0.0, 0.0, 0.0]], [0.0, 0.0], 1.0, 0.0

    def axial_to_cartesian(axial_list):
        pts = []
        for q, r in axial_list:
            x = math.sqrt(3) * (q + r / 2.0)
            y = 1.5 * r
            pts.append((x, y))
        return np.array(pts)

    # Generate initial candidates
    candidates = []
    
    # Reuse some axial candidates from the previous program as seeds
    axial_seeds = [
        [(0, 0), (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1), (2, 0), (-2, 0), (0, 2), (0, -2)],
        [(0, 0), (1, 0), (2, 0), (0, 1), (1, 1), (2, 1), (0, 2), (1, 2), (2, 2), (-1, 1), (3, 1)],
        [(0, 0), (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1), (2, 1), (1, 2), (-1, 2), (0, 3)],
        [(0, 0), (1, 0), (2, 0), (3, 0), (0, 1), (1, 1), (2, 1), (3, 1), (1, 2), (2, 2), (3, 2)],
        [(0, 0), (1, 0), (2, 0), (0, 1), (1, 1), (0, 2), (1, 2), (0, 3), (1, 3), (0, 4), (1, 4)]
    ]
    
    for ax in axial_seeds:
        if len(ax) == N:
            candidates.append(axial_to_cartesian(ax))
        else:
            # Pad or truncate if necessary, though seeds should be correct
            if len(ax) > N:
                candidates.append(axial_to_cartesian(ax[:N]))
            else:
                # Fill with random points
                pts = axial_to_cartesian(ax)
                while len(pts) < N:
                    pts = np.vstack([pts, np.random.uniform(-3, 3, 2)])
                candidates.append(pts)

    # Add a random start
    candidates.append(np.random.uniform(-3, 3, (N, 2)))
    
    # Add a "flower" start
    flower = np.zeros((N, 2))
    for i in range(1, 7):
        angle = i * math.pi / 3
        flower[i] = [math.sqrt(3) * math.cos(angle), math.sqrt(3) * math.sin(angle)]
    # Add 4 more
    flower[7] = [2 * math.sqrt(3), 0]
    flower[8] = [0, 2 * math.sqrt(3)]
    flower[9] = [-2 * math.sqrt(3), 0]
    flower[10] = [0, -2 * math.sqrt(3)]
    candidates.append(flower)

    best_S = float('inf')
    best_pts = None
    best_theta = 0.0

    def calc_S_and_widths(pts, theta):
        # pts: (N, 2)
        # theta: float
        widths = []
        for k in range(6):
            ang = theta + k * math.pi / 3.0
            nx, ny = math.cos(ang), math.sin(ang)
            projs = pts[:, 0] * nx + pts[:, 1] * ny
            w = np.max(projs) - np.min(projs)
            widths.append(w)
        widths = np.array(widths)
        S = np.max(widths / math.sqrt(3) + 1.0)
        return S, widths

    def objective(params):
        if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
            return float('inf')
        
        theta = params[0]
        # params[1:] are 20 coords for 10 points (p0 is fixed at 0,0)
        pts = np.zeros((N, 2))
        pts[1:, 0] = params[1::2]
        pts[1:, 1] = params[2::2]
        
        S, _ = calc_S_and_widths(pts, theta)
        
        # Penalty for overlaps
        penalty = 0.0
        for i in range(N):
            for j in range(i+1, N):
                d = np.linalg.norm(pts[i] - pts[j])
                if d < MIN_DIST:
                    penalty += (MIN_DIST - d)**2
        return S + 100.0 * penalty

    for i, init_pts in enumerate(candidates):
        if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
            break
            
        # Center the initial points
        init_pts = init_pts - np.mean(init_pts, axis=0)
        
        # Fix p0 to 0,0 for optimization
        init_params = np.zeros(1 + 2*(N-1))
        init_params[0] = 0.0 # Initial theta
        init_params[1::2] = init_pts[1:, 0]
        init_params[2::2] = init_pts[1:, 1]
        
        # Try a few theta values
        for theta_init in [0.0, math.pi/6, math.pi/3]:
            init_params[0] = theta_init
            try:
                res = minimize(objective, init_params, method='Nelder-Mead', options={'maxiter': 2000, 'xatol': 1e-6, 'fatol': 1e-6})
                if res.fun < best_S:
                    # Reconstruct
                    theta = res.x[0]
                    pts = np.zeros((N, 2))
                    pts[1:, 0] = res.x[1::2]
                    pts[1:, 1] = res.x[2::2]
                    
                    # Check validity
                    valid = True
                    for a in range(N):
                        for b in range(a+1, N):
                            if np.linalg.norm(pts[a] - pts[b]) < MIN_DIST - 1e-5:
                                valid = False
                                break
                        if not valid:
                            break
                    if valid:
                        S, _ = calc_S_and_widths(pts, theta)
                        if S < best_S:
                            best_S = S
                            best_pts = pts
                            best_theta = theta
            except Exception:
                pass

    if best_pts is None:
        # Fallback
        best_pts = axial_to_cartesian(axial_seeds[0])
        best_theta = 0.0
        best_S, _ = calc_S_and_widths(best_pts, best_theta)

    # Center the final points
    cx = np.mean(best_pts[:, 0])
    cy = np.mean(best_pts[:, 1])
    best_pts_centered = best_pts - np.array([cx, cy])
    
    # Calculate final S
    S, _ = calc_S_and_widths(best_pts_centered, best_theta)
    
    # Apply safety margin
    SCALE = 1.0 + 1e-9
    pts_scaled = best_pts_centered * SCALE
    
    # Recalculate S with scale
    S_final, _ = calc_S_and_widths(pts_scaled, best_theta)
    
    inner = []
    for x, y in pts_scaled:
        inner.append([float(x), float(y), float(math.degrees(best_theta))])
    
    center = [0.0, 0.0]
    side = float(S_final)
    rotation = float(math.degrees(best_theta))
    
    return inner, center, side, rotation
