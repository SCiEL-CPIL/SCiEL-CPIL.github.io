import numpy as np
from scipy.optimize import differential_evolution, minimize
import time

N = 50

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # 10s safety margin

    # Ring structure: 1 center, 6, 12, 18, 13
    # This sums to 50.
    
    def generate_centers(params):
        R1, th1, R2, th2, R3, th3, R4, th4 = params
        centers = [(0.0, 0.0)]
        
        # Ring 1: 6
        angles1 = th1 + np.arange(6) * 2 * np.pi / 6
        centers.extend([(R1 * np.cos(a), R1 * np.sin(a)) for a in angles1])
        
        # Ring 2: 12
        angles2 = th2 + np.arange(12) * 2 * np.pi / 12
        centers.extend([(R2 * np.cos(a), R2 * np.sin(a)) for a in angles2])
        
        # Ring 3: 18
        angles3 = th3 + np.arange(18) * 2 * np.pi / 18
        centers.extend([(R3 * np.cos(a), R3 * np.sin(a)) for a in angles3])
        
        # Ring 4: 13
        angles4 = th4 + np.arange(13) * 2 * np.pi / 13
        centers.extend([(R4 * np.cos(a), R4 * np.sin(a)) for a in angles4])
        
        return np.array(centers)

    def objective(params):
        centers = generate_centers(params)
        min_dist = np.inf
        # Vectorized calculation for speed?
        # 50 points, 1225 pairs. Double loop is fine.
        for i in range(N):
            dist_boundary = 1.0 - np.linalg.norm(centers[i])
            if dist_boundary < min_dist:
                min_dist = dist_boundary
            
            # Use numpy for pairwise distances from i
            diffs = centers[i+1:] - centers[i]
            dists = np.linalg.norm(diffs, axis=1)
            min_pair_dist = np.min(dists) if len(dists) > 0 else np.inf
            if min_pair_dist < min_dist:
                min_dist = min_pair_dist
                
        return -min_dist / 2.0

    bounds = [
        (0.1, 0.95), (0, 2*np.pi),
        (0.1, 0.95), (0, 2*np.pi),
        (0.1, 0.95), (0, 2*np.pi),
        (0.1, 0.95), (0, 2*np.pi)
    ]
    
    best_params = None
    best_score = -np.inf

    # Run DE multiple times with different seeds to explore landscape
    seeds = [42, 123, 456, 789]
    for seed in seeds:
        if time.time() > deadline - 20:
            break
        try:
            res = differential_evolution(
                objective, 
                bounds, 
                maxiter=500, 
                popsize=15, 
                tol=1e-9,
                seed=seed,
                polish=False
            )
            if -res.fun > best_score:
                best_score = -res.fun
                best_params = res.x
        except Exception:
            continue
            
    if best_params is None:
        # Fallback
        best_params = np.array([0.25, 0, 0.45, 0, 0.65, 0, 0.85, 0])
        
    # Polish with Nelder-Mead
    if time.time() < deadline - 5:
        try:
            res_nm = minimize(
                objective, 
                best_params, 
                method='Nelder-Mead',
                options={'maxiter': 5000, 'xatol': 1e-12, 'fatol': 1e-12}
            )
            if -res_nm.fun > best_score:
                best_params = res_nm.x
                best_score = -res_nm.fun
        except Exception:
            pass
            
    centers = generate_centers(best_params)
    
    # Final validation: ensure all points are inside
    # If any point is slightly outside due to numerical error, project it?
    # Or just return as is, the evaluator will handle it?
    # "A centre on or outside the container ... makes the candidate invalid."
    # So we MUST ensure validity.
    # If the optimizer found a solution where a center is outside, the score is invalid (0).
    # The objective function returns a negative radius if a center is outside.
    # DE minimizes this. It will avoid outside points if possible.
    # But to be safe, we can check and scale down if needed?
    # Scaling down reduces r, which reduces score.
    # Better to ensure the optimizer stays in valid region.
    # The bounds on R_i (max 0.95) and r ~ 0.12 means max radius ~ 0.95+0.12 = 1.07.
    # So it's possible to go outside.
    # We can add a penalty to the objective if any point is outside.
    
    # Let's refine objective to return a large positive value (bad) if invalid.
    # But DE minimizes. So we want to maximize r.
    # If invalid, r is effectively negative or undefined.
    # Let's just check the final result.
    
    max_rad = np.max(np.linalg.norm(centers, axis=1))
    if max_rad > 1.0:
        # Scale down to fit inside
        scale = 1.0 / max_rad
        centers = centers * scale
        # This reduces r, but ensures validity.
        
    return centers
