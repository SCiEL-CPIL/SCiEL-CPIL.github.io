import numpy as np
from scipy.optimize import minimize, differential_evolution
import time

N = 50

# EVOLVE-BLOCK-START
def construct():
    deadline = time.time() + 140.0  # safety margin before 150s kill
    
    # Ring structure for N=50: 1 center + 6 + 12 + 12 + 19 = 50
    ring_sizes = [6, 12, 12, 19]
    n_rings = len(ring_sizes)
    
    # Parameters: radial distances (n_rings) + angular offsets (n_rings) + center offset (2)
    # Total: 4 + 4 + 2 = 10 parameters
    n_params = n_rings * 2 + 2
    
    def build_points(params):
        """Build 50 centers from parametric representation."""
        radii = params[:n_rings]
        offsets = params[n_rings:2*n_rings]
        cx, cy = params[2*n_rings], params[2*n_rings+1]
        
        centers = np.array([[cx, cy]])
        
        for i, (r, off) in enumerate(zip(radii, offsets)):
            n = ring_sizes[i]
            # Ensure positive radius
            r = max(r, 0.01)
            angles = np.linspace(0, 2*np.pi, n, endpoint=False) + off
            xs = cx + r * np.cos(angles)
            ys = cy + r * np.sin(angles)
            centers = np.vstack([centers, np.column_stack([xs, ys])])
        
        return centers
    
    def compute_score(centers):
        """Compute the packing radius (minimum of inter-circle and boundary constraints)."""
        # Check all centers are finite
        if not np.all(np.isfinite(centers)):
            return -1.0
        
        # Distance to boundary
        dists_to_origin = np.linalg.norm(centers, axis=1)
        boundary_margin = 1.0 - np.max(dists_to_origin)
        
        # Minimum inter-center distance
        min_dist = np.inf
        for i in range(N):
            d = np.linalg.norm(centers - centers[i], axis=1)
            d[i] = np.inf  # exclude self
            min_dist = min(min_dist, np.min(d))
        
        return min(min_dist / 2.0, boundary_margin)
    
    def objective(params):
        """Negative score for minimization."""
        if time.time() > deadline:
            return 1e6
        centers = build_points(params)
        score = compute_score(centers)
        if score < 0:
            return 1e6 - score
        return -score
    
    # Initial guess based on known good packing
    # Radii roughly: 0.25, 0.5, 0.75, 0.95
    init_radii = np.array([0.25, 0.5, 0.75, 0.95])
    init_offsets = np.array([0.0, np.pi/12, 0.0, np.pi/12])
    init_center = np.array([0.0, 0.0])
    init_params = np.concatenate([init_radii, init_offsets, init_center])
    
    best_params = init_params.copy()
    best_score = compute_score(build_points(init_params))
    
    # Phase 1: Differential Evolution for global exploration
    if time.time() < deadline - 60:
        try:
            bounds = []
            # Radii: [0.1, 1.0]
            for i in range(n_rings):
                bounds.append((0.1, 1.0))
            # Offsets: [0, 2*pi]
            for i in range(n_rings):
                bounds.append((0, 2*np.pi))
            # Center: [-0.3, 0.3]
            bounds.append((-0.3, 0.3))
            bounds.append((-0.3, 0.3))
            
            de_result = differential_evolution(
                objective, bounds,
                seed=42,
                maxiter=100,
                tol=1e-8,
                popsize=15,
                polish=False,
                init='sobol'
            )
            
            if de_result.fun < -best_score:
                best_params = de_result.x
                best_score = -de_result.fun
        
        except Exception:
            pass
    
    # Phase 2: L-BFGS-B polish from best DE result
    if time.time() < deadline - 20:
        try:
            res = minimize(
                objective, best_params,
                method='L-BFGS-B',
                options={'maxiter': 500, 'ftol': 1e-15, 'gtol': 1e-12}
            )
            
            if res.fun < -best_score:
                best_params = res.x
                best_score = -res.fun
        
        except Exception:
            pass
    
    # Phase 3: Additional local refinements with different starting points
    if time.time() < deadline - 10:
        # Try a few perturbations of the best solution
        rng = np.random.RandomState(42)
        for attempt in range(5):
            if time.time() > deadline - 5:
                break
            
            perturbed = best_params + rng.randn(n_params) * 0.01
            try:
                res = minimize(
                    objective, perturbed,
                    method='Nelder-Mead',
                    options={'maxiter': 200, 'xatol': 1e-12, 'fatol': 1e-12}
                )
                
                if res.fun < -best_score:
                    best_params = res.x
                    best_score = -res.fun
                    
            except Exception:
                pass
    
    # Final polish
    if time.time() < deadline - 2:
        try:
            res = minimize(
                objective, best_params,
                method='L-BFGS-B',
                options={'maxiter': 100, 'ftol': 1e-16, 'gtol': 1e-14}
            )
            
            if res.fun < -best_score:
                best_params = res.x
                best_score = -res.fun
        
        except Exception:
            pass
    
    final_centers = build_points(best_params)
    
    # Validate: ensure all centers are inside unit circle
    dists = np.linalg.norm(final_centers, axis=1)
    if np.max(dists) >= 1.0:
        # Scale down to fit
        scale = 0.999 / np.max(dists)
        final_centers = final_centers * scale
    
    # Validate: ensure no coincident centers
    for i in range(N):
        for j in range(i+1, N):
            d = np.linalg.norm(final_centers[i] - final_centers[j])
            if d < 1e-10:
                # Nudge slightly
                final_centers[j] += 1e-8
    
    return final_centers
# EVOLVE-BLOCK-END
