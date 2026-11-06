"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import minimize
from math import cos, sin, pi

def _score(points):
    """Compute Delta / n^n for 8 points, where Delta is product of ordered pair distances
    after scaling to diameter 2."""
    n = len(points)
    pts = np.array(points, dtype=float)
    # Compute all pairwise distances
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dists = np.linalg.norm(diff, axis=2)
    # Diameter = max distance
    diam = np.max(dists)
    if diam < 1e-15:
        return 0.0
    # Scale to diameter 2
    scaled_dists = dists / diam * 2.0
    # Product of ordered pair distances (exclude self-distances which are 0)
    mask = ~np.eye(n, dtype=bool)
    ordered_dists = scaled_dists[mask]
    # Delta = product of all ordered pair distances
    # Use log to avoid underflow
    log_delta = np.sum(np.log(ordered_dists))
    log_score = log_delta - n * np.log(n)
    return np.exp(log_score)

def _neg_log_score(points):
    """Negative log-score for optimization. Returns (value, gradient)."""
    pts = np.array(points, dtype=float)
    n = len(pts)
    
    # Pairwise differences and distances
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dists = np.linalg.norm(diff, axis=2)
    
    # Avoid log(0)
    mask = ~np.eye(n, dtype=bool)
    dists_masked = dists[mask]
    if np.min(dists_masked) < 1e-15:
        return 0.0, np.zeros_like(pts)
        
    # Log of product of distances (ordered)
    log_prod = 2.0 * np.sum(np.log(dists_masked))
    
    # Diameter
    D = np.max(dists)
    
    # Objective: log_prod - n(n-1) * log(D)
    val = log_prod - n * (n - 1) * np.log(D)
    
    # Gradient
    # grad log_prod = 2 * sum_{j != i} (p_i - p_j) / d_ij^2
    dists_inv_sq = np.zeros_like(dists)
    np.fill_diagonal(dists, 1.0)
    dists_inv_sq = 1.0 / (dists * dists)
    np.fill_diagonal(dists_inv_sq, 0.0)
    
    grad_log_prod = np.einsum('ij,ijk->ik', dists_inv_sq, diff)
    grad_log_prod *= 2.0
    
    # grad D
    # D is max distance. Assume unique max for gradient (or use soft max if needed, 
    # but for high precision polish, hard max with active set is better if we know the pair).
    # However, for generic gradient, we can use the soft max approach or just 
    # approximate the gradient of max. 
    # Let's use the soft max gradient for robustness in the local optimizer.
    beta = 100.0
    triu_indices = np.triu_indices(n, k=1)
    d_pairs = dists[triu_indices]
    max_d = np.max(d_pairs)
    exp_d = np.exp(beta * (d_pairs - max_d))
    w = exp_d / np.sum(exp_d)
    
    grad_D = np.zeros_like(pts)
    for k in range(len(w)):
        a, b = triu_indices[0][k], triu_indices[1][k]
        d_k = dists[a, b]
        if d_k < 1e-15: continue
        vec = (pts[a] - pts[b]) / d_k
        grad_D[a] += w[k] * vec
        grad_D[b] += w[k] * (-vec)
        
    grad_penalty = -n * (n - 1) * (1.0 / D) * grad_D
    
    total_grad = grad_log_prod + grad_penalty
    
    return -val, -total_grad

def construct():
    start_time = time.time()
    deadline = start_time + 145.0

    best_score = 0.0
    best_points = None

    # Strategy: Search in reduced symmetry spaces
    # The regular octagon has D8 symmetry. The optimal solution might have lower symmetry.
    # We will test several symmetry classes:
    # 1. D8 (Regular Octagon) - baseline
    # 2. D4 (Square with 4-fold symmetry) - 4 unique points, repeated/rotated
    # 3. D2 (Rectangle with 2-fold symmetry) - 4 unique points, reflected
    # 4. C4 (4-fold rotational) - 4 unique points
    # 5. General (no symmetry) - but constrained to be close to symmetric starts

    def evaluate_and_update(points, score_func=None):
        nonlocal best_score, best_points
        # Check distinctness
        diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
        dists = np.linalg.norm(diff, axis=2)
        np.fill_diagonal(dists, np.inf)
        if np.min(dists) < 1e-8:
            return
        
        if score_func is None:
            s = _score(points)
        else:
            s = score_func(points)
            
        if s > best_score:
            best_score = s
            best_points = points.copy()

    # --- Symmetry Class 1: D8 (Regular Octagon) ---
    # Parameters: none (fixed shape)
    octagon = np.array([[cos(2 * pi * k / 8), sin(2 * pi * k / 8)] for k in range(8)])
    evaluate_and_update(octagon)

    # --- Symmetry Class 2: D4 (4-fold rotational symmetry) ---
    # Points: (r1, 0), (r2, theta), (-r1, 0), (-r2, theta), (0, r1), (-theta, r2), (0, -r1), (theta, -r2)
    # Actually, simpler: 4 unique points p0..p3. The set is {p_k, R(p_k), R^2(p_k), R^3(p_k)}? No, that's 4 points.
    # For 8 points with D4 symmetry:
    # 2 orbits of size 4.
    # Orbit 1: (r1, 0), (0, r1), (-r1, 0), (0, -r1) -> Square
    # Orbit 2: (r2, theta), (-theta, r2), (-r2, -theta), (theta, -r2) -> Square rotated by theta
    # Parameters: r1, r2, theta
    # Let's optimize these 3 parameters.
    
    def gen_d4(params):
        r1, r2, theta = params
        # Ensure positive radii
        r1 = abs(r1) + 1e-6
        r2 = abs(r2) + 1e-6
        
        pts = []
        # Orbit 1
        pts.append([r1, 0])
        pts.append([0, r1])
        pts.append([-r1, 0])
        pts.append([0, -r1])
        # Orbit 2
        pts.append([r2 * cos(theta), r2 * sin(theta)])
        pts.append([-r2 * sin(theta), r2 * cos(theta)])
        pts.append([-r2 * cos(theta), -r2 * sin(theta)])
        pts.append([r2 * sin(theta), -r2 * cos(theta)])
        return np.array(pts)

    # Grid search for D4
    r1_vals = np.linspace(0.2, 1.5, 20)
    r2_vals = np.linspace(0.2, 1.5, 20)
    theta_vals = np.linspace(0.1, pi/4, 10)
    
    # We need to scale to diameter 2 in _score, so absolute values don't matter for shape, 
    # but relative values do.
    # Let's fix r1=1 and vary r2 and theta to reduce dimensionality, or just let _score handle scaling.
    # Since _score scales, we can just search r1, r2, theta in a reasonable range.
    
    # To save time, we'll do a coarse grid and then local optimization on the best few.
    candidates_d4 = []
    for r1 in r1_vals[::2]: # Coarser for speed
        for r2 in r2_vals[::2]:
            for theta in theta_vals:
                pts = gen_d4([r1, r2, theta])
                s = _score(pts)
                candidates_d4.append((s, pts))
                
    candidates_d4.sort(key=lambda x: -x[0])
    
    # Refine top 5 D4 candidates
    for s, pts in candidates_d4[:5]:
        if time.time() > deadline - 10: break
        x0 = pts.flatten()
        try:
            # We can't directly optimize the 16 coords with D4 constraint easily in standard minimize
            # unless we use a wrapper that enforces symmetry.
            # Let's define a function that takes 3 params and returns the 16 coords,
            # and use minimize on the 3 params.
            def obj_d4(params):
                p = gen_d4(params)
                # Check distinctness
                diff = p[:, np.newaxis, :] - p[np.newaxis, :, :]
                dists = np.linalg.norm(diff, axis=2)
                np.fill_diagonal(dists, np.inf)
                if np.min(dists) < 1e-8:
                    return 1e10
                return -_score(p)
            
            # Initial guess from grid
            # Extract r1, r2, theta from pts is hard, so we just use the grid point that generated it.
            # Better: store the params.
            # Let's re-run the grid storing params.
            pass
        except:
            pass
            
    # Let's do a proper local optimization on the 3 parameters for D4
    def obj_d4_params(params):
        p = gen_d4(params)
        diff = p[:, np.newaxis, :] - p[np.newaxis, :, :]
        dists = np.linalg.norm(diff, axis=2)
        np.fill_diagonal(dists, np.inf)
        if np.min(dists) < 1e-8:
            return 1e10
        return -_score(p)
        
    # Grid search again to get params
    best_d4_params = None
    best_d4_score = -1e10
    for r1 in r1_vals:
        for r2 in r2_vals:
            for theta in theta_vals:
                s = -obj_d4_params([r1, r2, theta])
                if s > best_d4_score:
                    best_d4_score = s
                    best_d4_params = [r1, r2, theta]
                    
    if best_d4_params:
        # Local optimization on params
        try:
            res = minimize(obj_d4_params, best_d4_params, method='Nelder-Mead', options={'xatol': 1e-12, 'fatol': 1e-12, 'maxiter': 10000})
            if res.fun < -best_d4_score:
                p_opt = gen_d4(res.x)
                evaluate_and_update(p_opt)
        except:
            pass

    # --- Symmetry Class 3: D2 (2-fold rotational + reflections) ---
    # 4 unique points: (x1, y1), (-x1, -y1), (x2, -y1), (-x2, y1) ? No, D2 is rectangle symmetry.
    # Points: (x1, y1), (-x1, y1), (-x1, -y1), (x1, -y1) -> 4 points.
    # Need 8 points. 2 orbits of 4?
    # Orbit 1: (x1, y1), (-x1, y1), (-x1, -y1), (x1, -y1)
    # Orbit 2: (x2, y2), (-x2, y2), (-x2, -y2), (x2, -y2)
    # Parameters: x1, y1, x2, y2
    
    def gen_d2(params):
        x1, y1, x2, y2 = params
        pts = [
            [x1, y1], [-x1, y1], [-x1, -y1], [x1, -y1],
            [x2, y2], [-x2, y2], [-x2, -y2], [x2, -y2]
        ]
        return np.array(pts)

    def obj_d2_params(params):
        p = gen_d2(params)
        diff = p[:, np.newaxis, :] - p[np.newaxis, :, :]
        dists = np.linalg.norm(diff, axis=2)
        np.fill_diagonal(dists, np.inf)
        if np.min(dists) < 1e-8:
            return 1e10
        return -_score(p)

    # Grid search for D2
    best_d2_params = None
    best_d2_score = -1e10
    # Range: points should be within ~1.5 of origin for diameter ~2-3
    x1_vals = np.linspace(-1.5, 1.5, 15)
    y1_vals = np.linspace(-1.5, 1.5, 15)
    x2_vals = np.linspace(-1.5, 1.5, 15)
    y2_vals = np.linspace(-1.5, 1.5, 15)
    
    # Coarse grid
    for x1 in x1_vals[::3]:
        for y1 in y1_vals[::3]:
            for x2 in x2_vals[::3]:
                for y2 in y2_vals[::3]:
                    s = -obj_d2_params([x1, y1, x2, y2])
                    if s > best_d2_score:
                        best_d2_score = s
                        best_d2_params = [x1, y1, x2, y2]
                        
    if best_d2_params:
        try:
            res = minimize(obj_d2_params, best_d2_params, method='Nelder-Mead', options={'xatol': 1e-12, 'fatol': 1e-12, 'maxiter': 10000})
            if res.fun < -best_d2_score:
                p_opt = gen_d2(res.x)
                evaluate_and_update(p_opt)
        except:
            pass

    # --- Symmetry Class 4: General C4 (4-fold rotational) ---
    # 2 orbits of 4.
    # Orbit 1: (r1, theta1), (r1, theta1+pi/2), ...
    # Orbit 2: (r2, theta2), (r2, theta2+pi/2), ...
    # Parameters: r1, theta1, r2, theta2
    
    def gen_c4(params):
        r1, t1, r2, t2 = params
        r1 = abs(r1) + 1e-6
        r2 = abs(r2) + 1e-6
        pts = []
        for k in range(4):
            pts.append([r1 * cos(t1 + k*pi/2), r1 * sin(t1 + k*pi/2)])
            pts.append([r2 * cos(t2 + k*pi/2), r2 * sin(t2 + k*pi/2)])
        return np.array(pts)

    def obj_c4_params(params):
        p = gen_c4(params)
        diff = p[:, np.newaxis, :] - p[np.newaxis, :, :]
        dists = np.linalg.norm(diff, axis=2)
        np.fill_diagonal(dists, np.inf)
        if np.min(dists) < 1e-8:
            return 1e10
        return -_score(p)

    # Grid search for C4
    best_c4_params = None
    best_c4_score = -1e10
    r1_vals = np.linspace(0.2, 1.5, 15)
    t1_vals = np.linspace(0, pi/4, 5)
    r2_vals = np.linspace(0.2, 1.5, 15)
    t2_vals = np.linspace(0, pi/4, 5)
    
    for r1 in r1_vals[::2]:
        for t1 in t1_vals:
            for r2 in r2_vals[::2]:
                for t2 in t2_vals:
                    s = -obj_c4_params([r1, t1, r2, t2])
                    if s > best_c4_score:
                        best_c4_score = s
                        best_c4_params = [r1, t1, r2, t2]
                        
    if best_c4_params:
        try:
            res = minimize(obj_c4_params, best_c4_params, method='Nelder-Mead', options={'xatol': 1e-12, 'fatol': 1e-12, 'maxiter': 10000})
            if res.fun < -best_c4_score:
                p_opt = gen_c4(res.x)
                evaluate_and_update(p_opt)
        except:
            pass

    # --- Final Polish on the best found point (breaking symmetry if it helps) ---
    if best_points is not None and time.time() < deadline - 5:
        # Use the full 16-dim optimizer on the best symmetric point
        # This might break symmetry to find a slightly better non-symmetric local optimum
        x0 = best_points.flatten()
        try:
            res = minimize(_neg_log_score, x0, method='BFGS', options={'gtol': 1e-12, 'maxiter': 1000})
            if res.success:
                p_opt = res.x.reshape(8, 2)
                evaluate_and_update(p_opt)
        except:
            pass
        try:
            res = minimize(_neg_log_score, x0, method='L-BFGS-B', options={'gtol': 1e-12, 'maxiter': 1000})
            if res.success:
                p_opt = res.x.reshape(8, 2)
                evaluate_and_update(p_opt)
        except:
            pass

    # Fallback: Regular Octagon
    if best_points is None:
        best_points = octagon

    return [[float(p[0]), float(p[1])] for p in best_points]
# EVOLVE-BLOCK-END
