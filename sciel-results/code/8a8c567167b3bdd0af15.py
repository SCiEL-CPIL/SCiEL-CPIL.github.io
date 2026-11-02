import numpy as np
from scipy.optimize import minimize
import time

N = 11

def construct():
    start_time = time.time()
    deadline = start_time + 110.0 # Safety margin
    
    # Objective function
    def objective(params):
        S = params[0]
        theta = params[1]
        pts = params[2:].reshape(N, 2)
        
        # Center points
        pts = pts - np.mean(pts, axis=0)
        
        # Calculate boundary penalty
        # Normals for hexagon rotated by theta
        # Base normals for pointy-top (vertex at 0 deg) are at 0, 60, ...
        # If we rotate the hexagon by theta, normals rotate by theta.
        angles = np.deg2rad(theta) + np.deg2rad(np.arange(0, 360, 60))
        normals = np.column_stack([np.cos(angles), np.sin(angles)])
        
        R = S * np.sqrt(3) / 2
        
        # Projections
        proj = pts @ normals.T # Shape (N, 6)
        # Max projection
        max_proj = np.max(proj, axis=1) # Wait, we need max over k for each point?
        # No, the condition is n_k . p <= R for ALL k.
        # So violation is max(0, max_k (n_k . p - R))
        # Actually, if any n_k . p > R, it's outside.
        # So for each point, violation_i = max(0, max_k (proj[i,k] - R))
        violations = np.max(proj - R, axis=1)
        violations = np.maximum(violations, 0)
        boundary_penalty = np.sum(violations**2)
        
        # Calculate overlap penalty
        # Pairwise distances
        diffs = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dists = np.linalg.norm(diffs, axis=2)
        # Mask diagonal
        np.fill_diagonal(dists, np.inf)
        # We need min distance >= 2
        # Violation is max(0, 2 - d)
        viol_dists = np.maximum(0, 2 - dists)
        # Sum of squares of violations for i < j
        # To avoid double counting, take upper triangle
        iu = np.triu_indices(N, k=1)
        overlap_penalty = np.sum(viol_dists[iu]**2)
        
        return S + boundary_penalty + overlap_penalty

    # Initial guesses
    best_score = np.inf
    best_params = None
    
    # Run multiple restarts
    for i in range(100):
        if time.time() > deadline:
            break
            
        # Random initialization
        # S guess around 3.0
        S_init = 2.5 + np.random.rand() * 1.0
        theta_init = np.random.rand() * 30.0 # 0 to 30 deg
        # Points in a random hexagon
        pts_init = np.random.rand(N, 2) * 4.0 - 2.0
        
        x0 = np.concatenate(([S_init, theta_init], pts_init.flatten()))
        
        # Bounds
        # S: 2.0 to 5.0
        # theta: 0 to 30
        # pts: -5 to 5
        bounds = [(2.0, 5.0), (0, 30)] + [(-5, 5)] * (2 * N)
        
        try:
            res = minimize(objective, x0, method='L-BFGS-B', bounds=bounds, options={'maxiter': 1000, 'ftol': 1e-10})
            if res.success:
                # Check validity
                S_val = res.x[0]
                theta_val = res.x[1]
                pts_val = res.x[2:].reshape(N, 2)
                pts_val = pts_val - np.mean(pts_val, axis=0)
                
                # Recalculate penalties to ensure validity
                # (Reuse logic or just check objective value components)
                # Simpler: check if objective is close to S_val
                # If penalties are 0, objective == S_val
                if abs(res.fun - S_val) < 1e-5:
                    if S_val < best_score:
                        best_score = S_val
                        best_params = res.x
        except Exception:
            pass

    if best_params is None:
        # Fallback: Return a valid but large solution
        # Use the initial line solution but smaller?
        # Just return the seed-like solution to be safe
        inner = [[4.0 * k, 0.0, 0.0] for k in range(-(N // 2), N - N // 2)]
        return inner, (0.0, 0.0), 40.0, 0.0

    # Extract best
    S_best = best_params[0]
    theta_best = best_params[1]
    pts_best = best_params[2:].reshape(N, 2)
    pts_best = pts_best - np.mean(pts_best, axis=0)
    
    # Format inner list
    inner = [[float(x), float(y), 0.0] for x, y in pts_best]
    
    return inner, (0.0, 0.0), float(S_best), float(theta_best)
