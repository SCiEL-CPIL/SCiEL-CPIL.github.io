import numpy as np
import time
from scipy.optimize import minimize

N = 200

def construct():
    start_time = time.time()
    timeout = 110  # 10s buffer

    # 1. Initialization: Hexagonal Grid
    # Generate a hex grid and pick the N points closest to the center
    s = 0.075  # spacing
    points = []
    max_range = 5
    for j in range(-max_range, max_range + 1):
        for i in range(-max_range, max_range + 1):
            x = i * s + (j % 2) * s / 2
            y = j * s * np.sqrt(3) / 2
            if -0.5 <= x <= 0.5 and -0.5 <= y <= 0.5:
                points.append([x, y])

    points = np.array(points)

    if len(points) > N:
        dists = np.sum(points**2, axis=1)
        indices = np.argsort(dists)[:N]
        c = points[indices]
    elif len(points) < N:
        # Fallback to square grid
        m = int(np.ceil(np.sqrt(N)))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y) for y in ticks for x in ticks])
        c = grid[:N]
    else:
        c = points

    c = c.copy()

    # 2. Optimization
    def objective(c_flat):
        if time.time() - start_time > timeout:
            raise TimeoutError("Optimization timed out")
            
        c = c_flat.reshape(N, 2)
        
        # Pairwise repulsion
        c_sq = np.sum(c**2, axis=1)
        dist_sq = c_sq[:, None] + c_sq[None, :] - 2 * c @ c.T
        np.fill_diagonal(dist_sq, 0)
        
        lam = 40.0
        pair_potential = np.sum(np.exp(-lam * dist_sq))
        
        # Boundary repulsion
        d_x = 0.5 - np.abs(c[:, 0])
        d_y = 0.5 - np.abs(c[:, 1])
        min_d = np.minimum(d_x, d_y)
        
        # Penalize being close to boundary
        boundary_potential = np.sum(np.exp(-lam * min_d**2))
        
        return pair_potential + boundary_potential

    bounds = [(-0.5, 0.5)] * (2 * N)
    
    c_opt = c
    try:
        res = minimize(
            objective, 
            c.flatten(), 
            method='L-BFGS-B', 
            bounds=bounds,
            options={'maxiter': 300, 'ftol': 1e-9, 'gtol': 1e-9}
        )
        
        if res.success or res.nit > 0:
            c_candidate = res.x.reshape(N, 2)
            
            # Validate
            def calculate_radius(c_arr):
                c_sq = np.sum(c_arr**2, axis=1)
                dist_sq = c_sq[:, None] + c_sq[None, :] - 2 * c_arr @ c_arr.T
                np.fill_diagonal(dist_sq, 0)
                min_pair_dist = np.sqrt(np.min(dist_sq))
                
                d_x = 0.5 - np.abs(c_arr[:, 0])
                d_y = 0.5 - np.abs(c_arr[:, 1])
                min_boundary_dist = np.min(np.minimum(d_x, d_y))
                
                return min(min_pair_dist, min_boundary_dist) / 2.0

            r_candidate = calculate_radius(c_candidate)
            r_init = calculate_radius(c)
            
            if r_candidate > r_init:
                c_opt = c_candidate
                
    except TimeoutError:
        pass
    except Exception:
        pass

    return c_opt
# EVOLVE-BLOCK-END
