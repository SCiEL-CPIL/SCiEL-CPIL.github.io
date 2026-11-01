"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 32

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time

def construct():
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 140  # 140s out of 150s limit

    # Objective function: Maximize sum of radii.
    # Radii are determined by the minimum distance to boundary and other circles.
    # We optimize centers (x, y) to maximize this sum.
    # r_i = min(x_i, 1-x_i, y_i, 1-y_i, min_{j!=i} dist(i,j)/2)
    
    def objective(params):
        x = params[0::2]
        y = params[1::2]
        
        # Calculate radii based on clearance
        # Distance to boundaries
        dist_boundary = np.minimum(np.minimum(x, 1 - x), np.minimum(y, 1 - y))
        
        # Distance to other circles (pairwise)
        # We need min_{j != i} dist(i, j) / 2
        # For efficiency, we can compute pairwise distances.
        # N=32, so 32x32 matrix is small.
        dx = x[:, np.newaxis] - x[np.newaxis, :]
        dy = y[:, np.newaxis] - y[np.newaxis, :]
        dist_matrix = np.sqrt(dx**2 + dy**2)
        
        # Set diagonal to inf so min ignores self
        np.fill_diagonal(dist_matrix, np.inf)
        
        # Min distance to any other circle
        min_dist_to_circle = np.min(dist_matrix, axis=1)
        
        # Radius is min of boundary clearance and half min distance to circle
        radii = np.minimum(dist_boundary, min_dist_to_circle / 2.0)
        
        # Ensure radii are non-negative
        radii = np.maximum(radii, 0)
        
        return -np.sum(radii)

    # Initial guess: Grid
    side = int(np.ceil(np.sqrt(N)))
    x0 = np.zeros(N)
    y0 = np.zeros(N)
    idx = 0
    for i in range(side):
        for j in range(side):
            if idx >= N:
                break
            x0[idx] = (i + 0.5) / side
            y0[idx] = (j + 0.5) / side
            idx += 1
    if idx < N:
        # Fill remaining
        for k in range(idx, N):
            x0[k] = 0.5
            y0[k] = 0.5

    params0 = np.zeros(2 * N)
    params0[0::2] = x0
    params0[1::2] = y0

    # Bounds for centers: [0, 1]
    bounds = [(0.0, 1.0)] * (2 * N)

    best_params = params0.copy()
    best_obj = objective(params0)

    # Use Powell's method which is robust for non-smooth objectives (like min)
    # and doesn't require gradients.
    try:
        res = minimize(
            objective,
            params0,
            method='Powell',
            bounds=bounds,
            options={'maxiter': 5000, 'xtol': 1e-8, 'ftol': 1e-8}
        )
        
        if res.fun < best_obj:
            best_params = res.x
            best_obj = res.fun
            
    except Exception as e:
        pass

    # If time remains, try a few random restarts with L-BFGS-B using a smooth approximation
    # or just more Powell iterations.
    if time.time() < deadline - 20:
        np.random.seed(42)
        for restart in range(5):
            if time.time() > deadline - 10:
                break
                
            # Random initial guess
            params_rand = np.random.rand(2 * N)
            # Keep centers away from boundaries initially to avoid tiny radii
            params_rand = 0.1 + 0.8 * params_rand
            
            try:
                res = minimize(
                    objective,
                    params_rand,
                    method='Powell',
                    bounds=bounds,
                    options={'maxiter': 3000, 'xtol': 1e-8, 'ftol': 1e-8}
                )
                
                if res.fun < best_obj:
                    best_params = res.x
                    best_obj = res.fun
                    
            except Exception as e:
                pass

    # Construct output array
    x = best_params[0::2]
    y = best_params[1::2]
    
    # Recalculate radii for the final best positions
    dist_boundary = np.minimum(np.minimum(x, 1 - x), np.minimum(y, 1 - y))
    
    dx = x[:, np.newaxis] - x[np.newaxis, :]
    dy = y[:, np.newaxis] - y[np.newaxis, :]
    dist_matrix = np.sqrt(dx**2 + dy**2)
    np.fill_diagonal(dist_matrix, np.inf)
    min_dist_to_circle = np.min(dist_matrix, axis=1)
    
    radii = np.minimum(dist_boundary, min_dist_to_circle / 2.0)
    radii = np.maximum(radii, 0)

    # Final safety: ensure no overlaps and boundary violations due to numerical precision
    # The construction above guarantees disjoint interiors if we use strict < for distance
    # but floating point might cause dist == sum_r.
    # We shrink radii slightly to ensure strict disjointness.
    radii *= 0.9999

    result = np.column_stack((x, y, radii))
    return result

# EVOLVE-BLOCK-END
