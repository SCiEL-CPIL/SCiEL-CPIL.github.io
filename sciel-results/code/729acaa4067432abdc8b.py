import numpy as np
import time
from scipy.optimize import minimize

N = 50
RING_COUNTS = [1, 6, 12, 18, 13]

def construct():
    start_time = time.time()
    deadline = 145.0 # 5s margin

    # Initial guess based on known optimal structure
    # r ~ 0.1258
    # R1 ~ 2*r, R2 ~ 3*r, R3 ~ 4*r, R4 ~ 5*r?
    # Actually, for hex packing, R_k ~ k * r * sqrt(3)?
    # Let's use a simple heuristic start.
    r0 = 0.12
    R0 = [0.0, 2*r0, 3.5*r0, 5*r0, 6.5*r0]
    phi0 = [0.0, 0.0, np.pi/6, 0.0, np.pi/13] # Some offsets
    
    params0 = np.array([r0] + R0[1:] + phi0[1:])
    
    # Bounds
    # r: [0, 0.5]
    # R1..R4: [0, 1]
    # phi: [0, 2pi/N]
    bounds = [(0, 0.5)] # r
    for i in range(1, 5):
        bounds.append((0, 1)) # R
    for i, n in enumerate(RING_COUNTS[1:], 1):
        bounds.append((0, 2*np.pi/n)) # phi
    
    def compute_distances(params):
        r = params[0]
        R = params[1:5]
        phi = params[5:9]
        
        # Generate points
        pts = [np.array([0.0, 0.0])]
        for k in range(1, 5):
            n = RING_COUNTS[k]
            angles = np.linspace(0, 2*np.pi, n, endpoint=False) + phi[k-1]
            ring_pts = np.column_stack([R[k-1]*np.cos(angles), R[k-1]*np.sin(angles)])
            pts.append(ring_pts)
            
        all_pts = np.vstack(pts)
        
        # Compute min distance
        # We only need to check neighbors or use cKDTree? 
        # N=50 is small, O(N^2) is fine.
        min_d = np.inf
        for i in range(N):
            for j in range(i+1, N):
                d = np.linalg.norm(all_pts[i] - all_pts[j])
                if d < min_d:
                    min_d = d
                    if min_d <= 2*r:
                        break
            if min_d <= 2*r:
                break
        
        # Boundary constraint
        max_r = np.max(np.linalg.norm(all_pts, axis=1))
        
        return min_d, max_r

    def objective(params):
        r = params[0]
        R = params[1:5]
        
        # Hard boundary check for speed
        if R[3] + r > 1.0:
            return 1e6
            
        min_d, max_r = compute_distances(params)
        
        if max_r + r > 1.0:
            return 1e6
            
        if min_d < 2*r:
            # Penalty for overlap
            return 1e6 - min_d/2*r + r # Try to push r down or d up?
            # Better: return a large value if invalid.
            # We want to MAXIMIZE r. 
            # If we minimize -r, we need constraints.
            # Let's use a penalty method for simplicity.
            # Score = r - penalty
            # Penalty = max(0, 2r - min_d) + max(0, max_r + r - 1)
            pen = max(0, 2*r - min_d) + max(0, max_r + r - 1)
            return -(r - pen)
            
        return -r

    # Use L-BFGS-B
    best_score = -np.inf
    best_params = None
    
    # Run a few restarts
    for restart in range(10):
        if time.time() > deadline:
            break
            
        if restart == 0:
            p_init = params0
        else:
            p_init = params0 + 0.05 * np.random.randn(len(params0))
            p_init = np.clip(p_init, [b[0] for b in bounds], [b[1] for b in bounds])
            
        try:
            res = minimize(objective, p_init, method='L-BFGS-B', bounds=bounds, options={'maxiter': 500, 'ftol': 1e-8})
            if res.success:
                score = -res.fun
                if score > best_score:
                    best_score = score
                    best_params = res.x
        except Exception:
            pass

    # Fallback to hex lattice if optimization failed or score is low
    if best_params is None or best_score < 0.1:
        # Hex lattice logic from seed
        spacing = 2.0 / (np.sqrt(N) + 1.0)
        while True:
            k = int(1.0 / spacing) + 2
            rows = [((i + 0.5 * (j % 2)) * spacing, j * spacing * np.sqrt(3.0) / 2.0)
                    for i in range(-k, k + 1) for j in range(-k, k + 1)]
            points = np.array(rows)
            radius = np.hypot(points[:, 0], points[:, 1])
            points, radius = points[radius <= 1.0 - spacing / 2.0], radius[radius <= 1.0 - spacing / 2.0]
            if len(points) >= N:
                return points[np.argsort(radius, kind="stable")[:N]]
            spacing *= 0.98

    # Convert best_params to points
    r = best_params[0]
    R = best_params[1:5]
    phi = best_params[5:9]
    
    pts = [np.array([0.0, 0.0])]
    for k in range(1, 5):
        n = RING_COUNTS[k]
        angles = np.linspace(0, 2*np.pi, n, endpoint=False) + phi[k-1]
        ring_pts = np.column_stack([R[k-1]*np.cos(angles), R[k-1]*np.sin(angles)])
        pts.append(ring_pts)
        
    all_pts = np.vstack(pts)
    
    # Final check
    min_d = np.inf
    for i in range(N):
        for j in range(i+1, N):
            d = np.linalg.norm(all_pts[i] - all_pts[j])
            if d < min_d: min_d = d
    max_r = np.max(np.linalg.norm(all_pts, axis=1))
    
    if min_d < 2*r or max_r + r > 1.0:
        # Invalid, fallback
        spacing = 2.0 / (np.sqrt(N) + 1.0)
        while True:
            k = int(1.0 / spacing) + 2
            rows = [((i + 0.5 * (j % 2)) * spacing, j * spacing * np.sqrt(3.0) / 2.0)
                    for i in range(-k, k + 1) for j in range(-k, k + 1)]
            points = np.array(rows)
            radius = np.hypot(points[:, 0], points[:, 1])
            points, radius = points[radius <= 1.0 - spacing / 2.0], radius[radius <= 1.0 - spacing / 2.0]
            if len(points) >= N:
                return points[np.argsort(radius, kind="stable")[:N]]
            spacing *= 0.98

    return all_pts
