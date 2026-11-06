"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import math
from scipy.optimize import minimize

N = 50

# EVOLVE-BLOCK-START

def construct():
    # We want to maximize r = min(min_{i<j} dist_ij / 2, min_i (1 - |c_i|))
    # This is equivalent to minimizing -r.
    # Since r is a min of mins, it's non-smooth.
    # We can use a differentiable approximation or simply use a robust optimizer with finite differences.
    # L-BFGS-B with finite differences works well for this.
    
    def objective(x):
        # x is shape (N, 2)
        pts = x.reshape(N, 2)
        
        # Check validity: if any point is outside, return a large penalty
        # Actually, we want to maximize the radius.
        # Let's compute the actual score r.
        # If any point is outside the unit circle, the score is invalid (or effectively 0/negative).
        # The problem states: "A centre on or outside the container... make the candidate invalid."
        # So we must ensure |c_i| < 1.
        
        norms = np.linalg.norm(pts, axis=1)
        if np.any(norms >= 1.0):
            return 1e6  # Large penalty
        
        # Pairwise distances
        # To avoid memory blowup for large N, we can compute in chunks, but N=50 is small.
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dist_sq = np.sum(diff**2, axis=2)
        np.fill_diagonal(dist_sq, np.inf)
        
        min_dist_sq = np.min(dist_sq)
        min_dist = np.sqrt(min_dist_sq)
        
        # The radius of the circles is half the minimum distance between centers
        r_pair = min_dist / 2.0
        
        # The distance from the farthest center to the boundary
        max_norm = np.max(norms)
        r_boundary = 1.0 - max_norm
        
        # The common radius is the minimum of these two
        r = min(r_pair, r_boundary)
        
        # We want to maximize r, so we minimize -r
        return -r

    def gradient(x):
        # We could implement the gradient analytically, but finite differences are robust enough for N=50.
        # However, for speed, let's try to use the analytic gradient if possible, or just let L-BFGS-B use finite diff.
        # scipy.optimize.minimize with method='L-BFGS-B' can use finite differences if jac is None.
        # Given the non-smooth nature (min function), finite differences might be noisy but effective with enough iterations.
        # Let's stick to finite differences for simplicity and robustness against the 'min' non-differentiability.
        return None

    best_score = -1e9
    best_pts = None
    
    # Multi-start strategy
    np.random.seed(42)
    
    for run in range(10):
        # Initialize with a structured pattern + noise
        # Ring initialization
        if run == 0:
            # Try a simple grid or ring
            # Let's use a hexagonal-ish packing initialization
            # Or just random but projected
            pts = np.random.rand(N, 2) * 1.6 - 0.8
            # Project to be inside
            norms = np.linalg.norm(pts, axis=1)
            mask = norms > 0.9
            if np.any(mask):
                pts[mask] = pts[mask] / norms[mask, None] * 0.9
        else:
            # Perturb the previous best or use random
            if best_pts is not None and run < 5:
                # Small perturbation of best
                pts = best_pts + np.random.randn(N, 2) * 0.05
                norms = np.linalg.norm(pts, axis=1)
                mask = norms > 0.95
                if np.any(mask):
                    pts[mask] = pts[mask] / norms[mask, None] * 0.95
            else:
                pts = np.random.rand(N, 2) * 1.6 - 0.8
                norms = np.linalg.norm(pts, axis=1)
                mask = norms > 0.9
                if np.any(mask):
                    pts[mask] = pts[mask] / norms[mask, None] * 0.9
        
        x0 = pts.flatten()
        
        # Optimization
        # Maximize r -> Minimize -r
        # Constraints: |c_i| <= 1.
        # L-BFGS-B handles bounds well.
        bounds = [(-1.0, 1.0)] * (2 * N)
        
        try:
            res = minimize(
                objective, 
                x0, 
                method='L-BFGS-B', 
                jac=None, # Use finite differences
                bounds=bounds,
                options={'maxiter': 2000, 'ftol': 1e-12, 'gtol': 1e-8, 'maxls': 50}
            )
            
            # Extract result
            res_pts = res.x.reshape(N, 2)
            
            # Check if valid
            norms = np.linalg.norm(res_pts, axis=1)
            if np.any(norms >= 1.0):
                continue
                
            # Calculate score
            score = -res.fun
            
            if score > best_score:
                best_score = score
                best_pts = res_pts.copy()
                
        except Exception as e:
            continue

    if best_pts is None:
        # Fallback to a valid random configuration
        pts = np.random.rand(N, 2) * 0.5
        best_pts = pts
        best_score = -objective(pts.flatten())

    return best_pts

# EVOLVE-BLOCK-END
