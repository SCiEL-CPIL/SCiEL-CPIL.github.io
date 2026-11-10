from scipy.optimize import minimize
# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial.distance import pdist
import time
from scipy.optimize import minimize_scalar

def min_max_dist_dim2_16() -> np.ndarray:
    """
    Creates 16 points in 2 dimensions in order to maximize the ratio of minimum to maximum distance.
    
    Strategy: 
    1. Start with a regular 4x4 grid.
    2. Apply a radial distortion (warp) to the grid points to optimize the distance ratio.
    3. Use a 1D optimization to find the optimal distortion parameter.
    4. Refine with a few local steps if time permits, but the core is the deterministic construction.
    """
    start_time = time.time()
    deadline = start_time + 580.0  # Safety margin of 50 seconds

    def compute_score(points):
        if points.shape != (16, 2):
            return -np.inf
        d = pdist(points)
        if d.size == 0:
            return 0.0
        d_min = np.min(d)
        d_max = np.max(d)
        if d_max == 0:
            return 0.0
        ratio = d_min / d_max
        return ratio * ratio

    # Generate the base 4x4 grid
    # Centered at origin
    base_coords = np.array([[-1.5, -0.5, 0.5, 1.5] * 4,
                            [-1.5, -1.5, -1.5, -1.5,
                             -0.5, -0.5, -0.5, -0.5,
                              0.5,  0.5,  0.5,  0.5,
                              1.5,  1.5,  1.5,  1.5]]).T
    
    # Normalize scale
    base_coords = base_coords / np.max(np.linalg.norm(base_coords, axis=1))

    def generate_warped_grid(alpha):
        """
        Apply radial distortion to the grid.
        r' = r / (1 + alpha * r^2) is a common Möbius-like warp.
        Alternatively, we can use a power law: r' = r^beta.
        Let's try a combination: r' = r / (1 + alpha * r^2)
        """
        r = np.linalg.norm(base_coords, axis=1)
        # Avoid division by zero (though center is 0,0 which is fine for r=0)
        # For r=0, r' = 0.
        # For r > 0, apply warp.
        r_new = np.zeros_like(r)
        mask = r > 1e-10
        r_new[mask] = r[mask] / (1 + alpha * r[mask]**2)
        
        # Scale back to unit radius max to maintain gauge invariance
        max_r_new = np.max(r_new)
        if max_r_new < 1e-10:
            return base_coords.copy()
        
        scale_factor = 1.0 / max_r_new
        angles = np.arctan2(base_coords[:, 1], base_coords[:, 0])
        warped = np.column_stack([r_new * scale_factor * np.cos(angles), 
                                  r_new * scale_factor * np.sin(angles)])
        return warped

    def objective_1d(alpha):
        """
        Objective for 1D optimization: minimize -score(warped_grid(alpha))
        """
        points = generate_warped_grid(alpha)
        score = compute_score(points)
        if not np.isfinite(score):
            return 1e10
        return -score

    # Phase 1: 1D Optimization of the Warp Parameter
    # We expect alpha to be in a reasonable range, e.g., [-1, 1]
    # Negative alpha expands the center, positive alpha compresses it.
    
    best_points = None
    best_score = -np.inf
    
    # Initial evaluation
    initial_points = generate_warped_grid(0.0)
    initial_score = compute_score(initial_points)
    best_points = initial_points
    best_score = initial_score

    # Scan a few points to find a good bracket
    alphas = np.linspace(-1.0, 1.0, 21)
    scores = []
    for a in alphas:
        s = compute_score(generate_warped_grid(a))
        scores.append(s)
    
    best_idx = np.argmax(scores)
    best_alpha_scan = alphas[best_idx]
    best_score_scan = scores[best_idx]
    
    if best_score_scan > best_score:
        best_score = best_score_scan
        best_points = generate_warped_grid(best_alpha_scan)

    # Local 1D optimization around the best scan point
    if time.time() < deadline - 100:
        try:
            res = minimize_scalar(objective_1d, bounds=(best_alpha_scan - 0.5, best_alpha_scan + 0.5), 
                                  method='bounded', options={'xatol': 1e-8})
            if res.success:
                opt_alpha = res.x
                opt_points = generate_warped_grid(opt_alpha)
                opt_score = compute_score(opt_points)
                if opt_score > best_score:
                    best_score = opt_score
                    best_points = opt_points
        except Exception:
            pass

    # Phase 2: Symmetry-Breaking Local Refinement
    # The radial warp preserves the 4-fold symmetry. The optimal solution for 16 points 
    # might break this symmetry slightly. We can apply a small random perturbation 
    # and run a very fast local optimizer to see if it improves.
    
    if time.time() < deadline - 50:
        # Try a few random perturbations of the best symmetric solution
        np.random.seed(42)
        for i in range(5):
            if time.time() > deadline - 20:
                break
            
            noise = np.random.normal(0, 0.01, best_points.shape)
            perturbed = best_points + noise
            
            # Quick Nelder-Mead polish
            def obj_flat(x):
                pts = x.reshape(16, 2)
                return -compute_score(pts)
            
            try:
                res = minimize(obj_flat, perturbed.flatten(), method='Nelder-Mead', 
                               options={'maxiter': 100, 'fatol': 1e-8, 'xatol': 1e-8, 'disp': False})
                if res.success:
                    final_pts = res.x.reshape(16, 2)
                    final_score = compute_score(final_pts)
                    if final_score > best_score:
                        best_score = final_score
                        best_points = final_pts
            except Exception:
                pass

    # Final check
    if best_points is None:
        best_points = base_coords

    return best_points

# EVOLVE-BLOCK-END
