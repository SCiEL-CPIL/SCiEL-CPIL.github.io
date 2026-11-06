import time
import math
import numpy as np
from scipy.optimize import minimize, differential_evolution

def construct():
    n = 8
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin

    def compute_score(pts):
        """Compute the score for a set of points."""
        pts = np.asarray(pts)
        if not np.all(np.isfinite(pts)):
            return 0.0
        diffs = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diffs**2, axis=2))
        iu = np.triu_indices(n, k=1)
        d = dists[iu]
        if np.min(d) < 1e-12:
            return 0.0
        D = np.max(d)
        if D < 1e-12:
            return 0.0
        # Scale to diameter 2
        scaled_d = d * (2.0 / D)
        prod_unordered = np.prod(scaled_d)
        Delta = prod_unordered ** 2  # Product over ordered pairs
        return Delta / (n ** n)

    def objective_general(x):
        """
        Objective for general optimization.
        Minimizes -sum(log(d_ij)) + penalty for diameter > 2.
        """
        pts = x.reshape(n, 2)
        if not np.all(np.isfinite(pts)):
            return 1e10
        
        diffs = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diffs**2, axis=2))
        iu = np.triu_indices(n, k=1)
        d = dists[iu]
        
        if np.min(d) < 1e-10:
            return 1e10
            
        D = np.max(d)
        if D < 1e-10:
            return 1e10
            
        # Maximize sum(log(d)) -> Minimize -sum(log(d))
        log_sum = np.sum(np.log(d))
        
        # Penalty for exceeding diameter 2
        penalty = 0.0
        if D > 2.0:
            penalty = 100.0 * (D - 2.0)**2
            
        return -log_sum + penalty

    best_score = 0.0
    best_pts = None

    def try_candidate(pts):
        nonlocal best_score, best_pts
        pts = np.asarray(pts)
        if not np.all(np.isfinite(pts)):
            return 0.0
        s = compute_score(pts)
        if s > best_score:
            best_score = s
            best_pts = pts.copy()
        return s

    # 1. Regular Octagon
    angles_reg = np.linspace(0, 2 * np.pi, n, endpoint=False)
    reg_pts = np.column_stack([np.cos(angles_reg), np.sin(angles_reg)])
    try_candidate(reg_pts)

    # 2. Differential Evolution for Global Search
    # DE explores the 16D coordinate space.
    # We define a fitness function that returns the negative score (since DE minimizes).
    # To handle the diameter constraint in DE, we can use the log-product objective
    # which is scale-invariant up to a constant, or just penalize diameter > 2.
    # Actually, the score is invariant to scaling.
    # So we can just maximize the product of distances directly, or minimize -log(product).
    # Let's use -log(product) + penalty for diameter.
    # But DE works best with bounded variables.
    # Let's bound coordinates to [-1.5, 1.5].
    
    def de_objective(x):
        pts = x.reshape(n, 2)
        if not np.all(np.isfinite(pts)):
            return 1e10
        diffs = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diffs**2, axis=2))
        iu = np.triu_indices(n, k=1)
        d = dists[iu]
        if np.min(d) < 1e-10:
            return 1e10
        D = np.max(d)
        if D < 1e-10:
            return 1e10
        log_sum = np.sum(np.log(d))
        penalty = 0.0
        if D > 2.0:
            penalty = 100.0 * (D - 2.0)**2
        return -log_sum + penalty

    bounds = [(-1.5, 1.5)] * (2 * n)
    
    if time.time() < deadline - 30.0:
        try:
            # Run DE with a moderate number of iterations
            # popsize=15, maxiter=100 should be fast enough
            res_de = differential_evolution(
                de_objective,
                bounds,
                maxiter=100,
                popsize=15,
                tol=1e-8,
                seed=42,
                polish=True  # Use L-BFGS-B to polish the result
            )
            if res_de.success or res_de.fun < 1e9:
                de_pts = res_de.x.reshape(n, 2)
                try_candidate(de_pts)
        except Exception:
            pass

    # 3. Local Refinement from DE result and other starts
    # Use L-BFGS-B to refine the DE result and other promising starts
    
    # Get the current best points as a start
    if best_pts is not None:
        start_pts = best_pts
    else:
        start_pts = reg_pts

    # Also try refining the regular octagon
    starts = [start_pts.flatten(), reg_pts.flatten()]
    
    # Add a few random perturbations of the best found so far
    np.random.seed(42)
    for _ in range(5):
        perturbed = best_pts + 0.1 * np.random.randn(n, 2)
        starts.append(perturbed.flatten())

    for init in starts:
        if time.time() > deadline - 5.0:
            break
        try:
            res = minimize(
                objective_general,
                init,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 200, 'ftol': 1e-14}
            )
            if np.all(np.isfinite(res.x)):
                refined_pts = res.x.reshape(n, 2)
                try_candidate(refined_pts)
        except Exception:
            pass

    # Fallback
    if best_pts is None or not np.all(np.isfinite(best_pts)):
        best_pts = reg_pts

    # Final validation
    best_pts = np.asarray(best_pts)
    diffs = best_pts[:, np.newaxis, :] - best_pts[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diffs**2, axis=2))
    iu = np.triu_indices(n, k=1)
    d = dists[iu]
    if np.min(d) < 1e-6 or not np.all(np.isfinite(best_pts)):
        best_pts = reg_pts

    # Scale to ensure coords are small (optional, but good for stability)
    max_coord = np.max(np.abs(best_pts))
    if max_coord > 1e6:
        best_pts = best_pts / max_coord * 1e5

    return best_pts.tolist()
