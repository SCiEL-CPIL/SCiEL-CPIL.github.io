import time
import math
import numpy as np
from scipy.optimize import differential_evolution, minimize

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s kill
    
    # Objective function for 8 variables representing 4 complex numbers
    # x = [x1, y1, x2, y2, x3, y3, x4, y4]
    # Points are (xk, yk) and (xk, -yk)
    def get_points(x):
        x1, y1, x2, y2, x3, y3, x4, y4 = x
        return np.array([
            [x1, y1],
            [x1, -y1],
            [x2, y2],
            [x2, -y2],
            [x3, y3],
            [x3, -y3],
            [x4, y4],
            [x4, -y4]
        ])

    def compute_score(pts):
        """
        Compute the actual score: Delta / n^n.
        Delta is product over ordered pairs of distances after scaling to diameter 2.
        """
        n = 8
        pts = np.asarray(pts)
        if not np.all(np.isfinite(pts)):
            return 0.0
            
        diffs = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diffs**2, axis=2))
        iu = np.triu_indices(n, k=1)
        d = dists[iu]
        
        # Check for coincident points
        if np.min(d) < 1e-12:
            return 0.0
            
        D = np.max(d)
        if D < 1e-12:
            return 0.0
            
        # Scale distances to diameter 2
        # d'_ij = d_ij * (2 / D)
        # Score = (Prod_{i!=j} d'_ij) / n^n
        # Prod_{i!=j} d'_ij = (Prod_{i<j} d'_ij)^2
        # log(Score) = 2 * sum_{i<j} log(d'_ij) - n*log(n)
        # log(d'_ij) = log(d_ij) + log(2) - log(D)
        # sum_{i<j} log(d'_ij) = sum(log(d_ij)) + 28*log(2) - 28*log(D)
        
        log_d_sum = np.sum(np.log(d))
        log_prod_unordered_scaled = log_d_sum + 28.0 * math.log(2.0) - 28.0 * math.log(D)
        log_score = 2.0 * log_prod_unordered_scaled - 8.0 * math.log(8.0)
        
        return math.exp(log_score)

    def objective(x):
        """
        Minimize negative log score for optimization.
        Returns a large penalty for invalid configurations.
        """
        pts = get_points(x)
        if not np.all(np.isfinite(pts)):
            return 1e100
            
        diffs = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diffs**2, axis=2))
        iu = np.triu_indices(8, k=1)
        d = dists[iu]
        
        if np.min(d) < 1e-9:
            return 1e100
            
        D = np.max(d)
        if D < 1e-9:
            return 1e100
            
        log_d_sum = np.sum(np.log(d))
        log_score = 2.0 * (log_d_sum + 28.0 * math.log(2.0) - 28.0 * math.log(D)) - 8.0 * math.log(8.0)
        
        return -log_score

    bounds = [(-10, 10)] * 8
    
    best_val = np.inf
    best_x = None
    best_score = 0.0

    def update_best(x, val):
        nonlocal best_val, best_x, best_score
        if val < best_val:
            best_val = val
            best_x = x.copy()
            # Calculate actual score for reporting/tracking
            pts = get_points(best_x)
            s = compute_score(pts)
            if s > best_score:
                best_score = s

    # 1. Initial Guess: Regular Octagon (rotated to avoid axis alignment issues in symmetry)
    # Angles: 22.5, 67.5, 112.5, 157.5 degrees
    angles = np.deg2rad([22.5, 67.5, 112.5, 157.5])
    init_x = []
    for a in angles:
        init_x.append(math.cos(a))
        init_x.append(math.sin(a))
    init_x = np.array(init_x)
    
    val = objective(init_x)
    update_best(init_x, val)

    # 2. Differential Evolution to find a good global basin
    # 8 variables is small, so DE is fast.
    try:
        res = differential_evolution(
            objective, 
            bounds, 
            seed=42, 
            maxiter=150, 
            popsize=25,
            tol=1e-8,
            polish=True  # Uses L-BFGS-B to polish
        )
        
        if res.fun < best_val:
            update_best(res.x, res.fun)
            
    except Exception:
        pass

    # 3. Local Refinement with L-BFGS-B from multiple starting points
    # We try the current best, and some random perturbations/structured starts
    
    start_points = [best_x]
    
    # Add random perturbations of the best solution
    np.random.seed(42)
    for _ in range(5):
        if time.time() > deadline - 10.0:
            break
        x_perturb = best_x + 0.5 * np.random.randn(8)
        # Clip to bounds
        x_perturb = np.clip(x_perturb, [b[0] for b in bounds], [b[1] for b in bounds])
        start_points.append(x_perturb)
        
    # Add structured symmetric candidates
    # Example: Two concentric squares
    # Square 1: (r1, 0), (0, r1), (-r1, 0), (0, -r1) -> Symmetric pairs: (r1,0),(r1,0) -- Degenerate!
    # We need y != 0.
    # Square 1 rotated by 45 deg: (r1/sqrt(2), r1/sqrt(2)), etc.
    # Let's use a generic symmetric set: 4 points in upper half plane.
    
    # Candidate: Regular Octagon rotated slightly
    for angle_offset in [0.1, -0.1, 0.05]:
        if time.time() > deadline - 10.0:
            break
        angles_test = np.deg2rad([22.5, 67.5, 112.5, 157.5]) + angle_offset
        x_test = []
        for a in angles_test:
            x_test.append(math.cos(a))
            x_test.append(math.sin(a))
        x_test = np.array(x_test)
        start_points.append(x_test)

    # Run L-BFGS-B on each start point
    for x0 in start_points:
        if time.time() > deadline - 5.0:
            break
            
        try:
            res_local = minimize(
                objective, 
                x0, 
                method='L-BFGS-B', 
                bounds=bounds, 
                options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-8}
            )
            
            if np.all(np.isfinite(res_local.x)):
                if res_local.fun < best_val:
                    update_best(res_local.x, res_local.fun)
                    
        except Exception:
            pass

    # 4. Final High-Precision Polish
    if time.time() < deadline - 2.0 and best_x is not None:
        try:
            res_final = minimize(
                objective, 
                best_x, 
                method='L-BFGS-B', 
                bounds=bounds, 
                options={'maxiter': 1000, 'ftol': 1e-15, 'gtol': 1e-10}
            )
            if np.all(np.isfinite(res_final.x)) and res_final.fun < best_val:
                update_best(res_final.x, res_final.fun)
        except Exception:
            pass

    # Construct final points
    if best_x is None:
        # Fallback
        angles = np.linspace(0, 2*np.pi, 8, endpoint=False)
        pts = np.column_stack([np.cos(angles), np.sin(angles)])
        return pts.tolist()
        
    pts = get_points(best_x)
    
    # Verify distinctness
    diffs = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diffs**2, axis=2))
    iu = np.triu_indices(8, k=1)
    d = dists[iu]
    
    if np.min(d) < 1e-6 or not np.all(np.isfinite(pts)):
        # Fallback to regular octagon if invalid
        angles = np.linspace(0, 2*np.pi, 8, endpoint=False)
        pts = np.column_stack([np.cos(angles), np.sin(angles)])
        
    # Scale to ensure coordinates are within bounds (they should be, but just in case)
    max_coord = np.max(np.abs(pts))
    if max_coord > 1e6:
        pts = pts / max_coord * 1e5

    return pts.tolist()
