import numpy as np
import time
from scipy.optimize import minimize

N = 21

def _voronoi_radii(centers, W, H):
    """
    Compute the maximal radius for each circle given its center and the rectangle boundaries.
    For each circle i, r_i = min(
        x_i, W - x_i, y_i, H - y_i,
        min_{j != i} (dist(i,j) - r_j)  -- but this is circular.
    )
    Actually, the standard "Voronoi" radius for a set of points in a box is:
    r_i = min(
        x_i, W - x_i, y_i, H - y_i,
        0.5 * min_{j != i} dist(i,j)
    )
    This ensures that if we set all radii to these values, they are disjoint.
    Proof: dist(i,j) >= r_i + r_j?
    r_i <= 0.5 dist(i,j) and r_j <= 0.5 dist(i,j) => r_i + r_j <= dist(i,j).
    Yes, this is a valid packing.
    """
    radii = np.zeros(N)
    for i in range(N):
        # Boundary constraints
        r = min(centers[i, 0], W - centers[i, 0], centers[i, 1], H - centers[i, 1])
        # Neighbor constraints
        # dist(i, j) for all j != i
        # We can vectorize this:
        dx = centers[:, 0] - centers[i, 0]
        dy = centers[:, 1] - centers[i, 1]
        dists = np.sqrt(dx**2 + dy**2)
        dists[i] = np.inf
        min_dist = np.min(dists)
        r = min(r, 0.5 * min_dist)
        radii[i] = r
    return radii

def _objective_and_grad(x_flat, W, H):
    """
    Objective to MINIMIZE: -sum(radii)
    We use finite differences or analytic gradients?
    Nelder-Mead doesn't use gradients, so we just need the objective.
    """
    centers = x_flat.reshape(-1, 2)
    radii = _voronoi_radii(centers, W, H)
    return -np.sum(radii)

def construct():
    start_time = time.time()
    deadline = 145.0  # Safety margin
    
    best_value = 0.0
    best_result = None
    
    # Search over aspect ratios
    # W + H = 2. W in [0.5, 1.5]
    # Symmetric, so W in [1.0, 1.5] is enough, but let's cover [0.8, 1.2] finely.
    aspect_ratios = np.linspace(0.85, 1.15, 13)
    
    # Random restarts per aspect ratio
    n_restarts = 5
    
    # Nelder-Mead parameters
    # We have 2N = 42 variables.
    # This might be slow. Let's see.
    # If it's too slow, we reduce restarts or use a simpler method.
    # Nelder-Mead on 42 dims is heavy.
    # Maybe use COBYLA? Or just a few restarts.
    # Let's try to be efficient.
    
    # Alternative: Use a coarse grid + local optimization.
    # Or use a simpler local search on the Voronoi radii.
    # Actually, the Voronoi objective is non-smooth (min function).
    # Nelder-Mead handles non-smoothness well.
    
    # Let's estimate time.
    # Each evaluation: N^2 distance calc. 21^2 = 441 ops. Very fast.
    # Nelder-Mead iterations: typically 10 * D = 420 iterations?
    # 420 * 441 ~ 180k ops. Negligible.
    # But Python overhead for _voronoi_radii loop might add up.
    # Let's vectorize _voronoi_radii more.
    
    def _voronoi_radii_fast(centers, W, H):
        radii = np.zeros(N)
        # Boundary
        r_bound = np.minimum(np.minimum(centers[:, 0], W - centers[:, 0]), 
                             np.minimum(centers[:, 1], H - centers[:, 1]))
        
        # Neighbor
        # dist matrix
        dx = centers[:, 0][:, None] - centers[:, 0][None, :]
        dy = centers[:, 1][:, None] - centers[:, 1][None, :]
        dists = np.sqrt(dx**2 + dy**2)
        np.fill_diagonal(dists, np.inf)
        min_dists = np.min(dists, axis=1)
        r_neigh = 0.5 * min_dists
        
        radii = np.minimum(r_bound, r_neigh)
        return radii

    def _objective(x_flat, W, H):
        centers = x_flat.reshape(-1, 2)
        # Clip centers to be inside box? 
        # If a center goes outside, r_bound becomes negative.
        # We should penalize or clip.
        # Clipping centers to [0, W]x[0, H] makes the function continuous?
        # If we clip, the gradient is 0 outside, which is fine for NM.
        # But if we clip, the "true" radius is 0 if center is outside?
        # No, the circle must be inside. So center must be in [r, W-r].
        # In the Voronoi formulation, if center is at 0.1 and W=1, r_bound=0.1.
        # If center is at -0.1, r_bound = -0.1. Radius cannot be negative.
        # So we should max(0, r_bound).
        # But if r is 0, it contributes nothing.
        # To guide the optimizer, we should probably penalize going out of bounds
        # or just clip the centers to a valid range [0, W]x[0, H].
        # If we clip, the objective is flat outside the box.
        # Let's just evaluate on the clipped centers for the radius calculation,
        # but this might create plateaus.
        # Better: Let the centers be free, but calculate radius as:
        # r_i = max(0, min(x_i, W-x_i, y_i, H-y_i, 0.5*min_dist))
        
        radii = _voronoi_radii_fast(centers, W, H)
        radii = np.maximum(0, radii)
        return -np.sum(radii)

    # We will run Nelder-Mead for each (aspect, seed)
    # To save time, we will limit the number of iterations or function calls.
    # maxiter=200, maxfev=1000
    
    for W in aspect_ratios:
        H = 2.0 - W
        
        for seed in range(n_restarts):
            if time.time() > deadline:
                break
            
            np.random.seed(seed + int(W*100))
            
            # Initial guess: Random points in the box
            # Or grid points? Random is better for escaping local minima.
            # But random might start with overlaps.
            # Voronoi radii handles overlaps by shrinking radii.
            # So random start is valid.
            
            x0 = np.random.rand(N, 2)
            x0[:, 0] *= W
            x0[:, 1] *= H
            
            # Try a few different initializations?
            # 1. Random
            # 2. Grid
            # 3. Jittered Grid
            
            # Let's stick to random for now, maybe add grid if time permits.
            
            try:
                res = minimize(
                    _objective, 
                    x0.flatten(), 
                    args=(W, H),
                    method='Nelder-Mead',
                    options={'maxiter': 500, 'maxfev': 2000, 'xatol': 1e-6, 'fatol': 1e-8}
                )
                
                val = -res.fun
                if val > best_value:
                    best_value = val
                    centers = res.x.reshape(-1, 2)
                    radii = _voronoi_radii_fast(centers, W, H)
                    radii = np.maximum(0, radii)
                    
                    # Verify feasibility just in case (floating point)
                    # If any radius is negative, set to 0.
                    # If circles overlap due to FP error, shrink them slightly.
                    # The Voronoi construction guarantees no overlap mathematically.
                    # r_i + r_j <= 0.5 d_ij + 0.5 d_ij = d_ij.
                    # So it should be safe.
                    
                    result = np.zeros((N, 3))
                    result[:, 0] = centers[:, 0]
                    result[:, 1] = centers[:, 1]
                    result[:, 2] = radii
                    
                    best_result = result
                    
            except Exception:
                pass

    if best_result is None:
        # Fallback
        W = 1.0
        H = 1.0
        cols = 5
        rows = 5
        x = np.zeros(N)
        y = np.zeros(N)
        r = np.zeros(N)
        for i in range(N):
            c = i % cols
            r_idx = i // cols
            x[i] = (c + 0.5) / cols
            y[i] = (r_idx + 0.5) / rows
            r[i] = 0.09
        best_result = np.column_stack((x, y, r))
        best_value = np.sum(r)
        
    return best_result
