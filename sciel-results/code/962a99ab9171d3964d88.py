"""Program to maximize product of distances for 8 planar points at fixed diameter."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from math import cos, sin, pi, log, sqrt, exp
from scipy.optimize import minimize, differential_evolution

def _score(points):
    """Compute Delta / n^n where Delta is product of ordered-pair distances after scaling to diameter 2."""
    n = len(points)
    pts = np.array(points, dtype=np.float64)
    if n < 2:
        return 0.0
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    diam = np.max(dists)
    if diam == 0:
        return 0.0
    scale = 2.0 / diam
    dists_scaled = dists * scale
    mask = ~np.eye(n, dtype=bool)
    log_prod = np.sum(np.log(dists_scaled[mask]))
    score = np.exp(log_prod) / (n ** n)
    return score

def _neg_log_score(points):
    """Negative log of the score for minimization."""
    n = len(points)
    pts = np.array(points, dtype=np.float64)
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    diam = np.max(dists)
    if diam == 0:
        return 1e10
    scale = 2.0 / diam
    dists_scaled = dists * scale
    mask = ~np.eye(n, dtype=bool)
    log_prod = np.sum(np.log(dists_scaled[mask]))
    return n * log(n) - log_prod

def _sum_sq_dists(points):
    """Sum of squared distances between all pairs."""
    pts = np.array(points)
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    sq_dists = np.sum(diff**2, axis=2)
    return np.sum(sq_dists)

def _two_squares_model(r1, r2, theta1, theta2, y_off):
    """
    Generate 8 points: 
    Square 1: 4 points at angles theta1, theta1+pi/2, theta1+pi, theta1+3pi/2 with radius r1, shifted by (0, y_off)
    Square 2: 4 points at angles theta2, theta2+pi/2, theta2+pi, theta2+3pi/2 with radius r2, shifted by (0, -y_off)
    """
    angles1 = np.array([theta1, theta1 + pi/2, theta1 + pi, theta1 + 3*pi/2])
    angles2 = np.array([theta2, theta2 + pi/2, theta2 + pi, theta2 + 3*pi/2])
    
    pts1 = np.column_stack([r1 * cos(angles1), r1 * sin(angles1) + y_off])
    pts2 = np.column_stack([r2 * cos(angles2), r2 * sin(angles2) - y_off])
    
    return np.vstack([pts1, pts2])

def _five_param_neglog(params):
    """Negative log score for 5-parameter model."""
    r1, r2, theta1, theta2, y_off = params
    if r1 < 0.01 or r2 < 0.01:
        return 1e10
    pts = _two_squares_model(r1, r2, theta1, theta2, y_off)
    return _neg_log_score(pts)

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s kill
    n = 8
    best_score = 0.0
    best_pts = None
    
    # Regular octagon as baseline
    oct_pts = np.array([[cos(2 * pi * k / 8), sin(2 * pi * k / 8)] for k in range(8)])
    best_score = _score(oct_pts)
    best_pts = oct_pts.copy()
    
    rng = np.random.RandomState(42)
    
    # === PHASE 1: Global Optimization on 5-parameter model ===
    # This model captures the most likely symmetric structures.
    if time.time() < deadline - 60:
        bounds = [
            (0.1, 3.0),   # r1
            (0.1, 3.0),   # r2
            (0, pi/2),    # theta1
            (0, pi/2),    # theta2
            (-1.0, 1.0)   # y_offset
        ]
        
        try:
            de_result = differential_evolution(
                _five_param_neglog, bounds,
                seed=42, maxiter=300, popsize=30,
                tol=1e-12, polish=True,
                workers=1
            )
            if de_result.fun < 1e6:
                pts = _two_squares_model(*de_result.x)
                if pts is not None:
                    sc = _score(pts)
                    if sc > best_score:
                        best_score = sc
                        best_pts = pts.copy()
        except Exception:
            pass

    # === PHASE 2: Local Optimization in 16D with Scale Fix ===
    # We optimize the full coordinates to allow breaking the square symmetry.
    # To handle scale invariance, we constrain the sum of squared distances.
    
    def objective_16d(x):
        pts = x.reshape(n, 2)
        return _neg_log_score(pts)

    # Constraints for 16D optimization:
    # 1. Centroid at origin (2 constraints)
    # 2. Sum of squared distances fixed to target (1 constraint)
    
    def constraint_centroid(x):
        pts = x.reshape(n, 2)
        return np.sum(pts, axis=0) # Should be [0,0]

    # We will define the target sum of sq dists based on the current best point
    # This function will be used inside the loop
    
    def make_constraints(target_ssd):
        def constraint_ssd(x):
            pts = x.reshape(n, 2)
            s = _sum_sq_dists(pts)
            return s - target_ssd
        return constraint_ssd

    def run_local_opt(initial_pts, target_ssd, max_iter=500):
        # Initial point
        x0 = initial_pts.flatten()
        
        # Constraints
        cons = [
            {'type': 'eq', 'fun': constraint_centroid},
            {'type': 'eq', 'fun': make_constraints(target_ssd)}
        ]
        
        # Bounds to keep coordinates reasonable
        bounds = [(-10, 10)] * (2 * n)
        
        try:
            result = minimize(
                objective_16d, x0, method='SLSQP',
                bounds=bounds, constraints=cons,
                options={'maxiter': max_iter, 'ftol': 1e-14}
            )
            if result.success or result.fun < 1e6:
                opt_pts = result.x.reshape(n, 2)
                # Check if it's valid (distinct points)
                if np.all(np.linalg.norm(opt_pts - opt_pts[0], axis=1) > 1e-6):
                    return opt_pts
        except Exception:
            pass
        return None

    # Iterate local optimization from the best 5-param point and its perturbations
    if time.time() < deadline - 10:
        # Start from the best found so far
        current_best = best_pts
        current_ssd = _sum_sq_dists(current_best)
        
        # Try to improve
        for attempt in range(20):
            if time.time() > deadline - 5:
                break
            
            # Perturb the current best
            noise_scale = 0.1 / (attempt + 1)
            noise = rng.randn(n, 2) * noise_scale
            perturbed = current_best + noise
            
            # Re-center to help the optimizer
            perturbed = perturbed - np.mean(perturbed, axis=0)
            
            # Update target SSD based on the perturbed point to maintain scale roughly
            # Or keep the original target? Keeping original target forces it to return to same scale.
            # Let's use the SSD of the perturbed point to allow slight scale changes during search,
            # but the objective is scale invariant, so the constraint just anchors it.
            target_ssd = _sum_sq_dists(perturbed)
            
            opt_pts = run_local_opt(perturbed, target_ssd, max_iter=200)
            
            if opt_pts is not None:
                sc = _score(opt_pts)
                if sc > best_score:
                    best_score = sc
                    best_pts = opt_pts
                    current_best = opt_pts
                    current_ssd = _sum_sq_dists(current_best)
                    
    # === PHASE 3: Fine Polish ===
    # Do a few more rounds of very small perturbations and optimization
    if time.time() < deadline - 2:
        for trial in range(10):
            if time.time() > deadline - 1:
                break
            noise = rng.randn(n, 2) * 0.01
            perturbed = best_pts + noise
            perturbed = perturbed - np.mean(perturbed, axis=0)
            target_ssd = _sum_sq_dists(perturbed)
            
            opt_pts = run_local_opt(perturbed, target_ssd, max_iter=100)
            
            if opt_pts is not None:
                sc = _score(opt_pts)
                if sc > best_score:
                    best_score = sc
                    best_pts = opt_pts

    return best_pts.tolist()
# EVOLVE-BLOCK-END
