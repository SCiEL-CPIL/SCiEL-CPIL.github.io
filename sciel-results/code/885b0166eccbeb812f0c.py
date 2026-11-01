"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 32

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time

def construct():
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 135  # 135s out of 150s limit, safety margin

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

    # Initial guess: Hexagonal packing pattern (better than grid for dense packing)
    def hex_init(n):
        """Generate hexagonal close-packed positions for n circles in unit square."""
        # Use a hexagonal lattice, then trim to fit in [0,1]^2
        # Hex spacing: if radius ~ r, horizontal spacing = 2r, vertical = 2*r*sqrt(3)/2 = r*sqrt(3)
        # For n=32, try to fit ~6x6 hex grid with some adjustment
        pts = []
        # Try different scales to find good packing
        best_pts = None
        best_sum = -np.inf
        
        for scale in np.arange(0.08, 0.20, 0.01):
            # Generate hex lattice points
            dx = scale * 2.0  # horizontal spacing
            dy = scale * np.sqrt(3)  # vertical spacing
            offset = dx / 2.0
            
            # Generate points in a range larger than [0,1]
            n_rows = int(1.0 / dy) + 3
            n_cols = int(1.0 / dx) + 3
            
            candidates = []
            for row in range(n_rows):
                y_pos = row * dy
                x_offset = offset if row % 2 == 1 else 0.0
                for col in range(n_cols):
                    x_pos = col * dx + x_offset
                    if 0 <= x_pos <= 1 and 0 <= y_pos <= 1:
                        candidates.append((x_pos, y_pos))
            
            if len(candidates) < n:
                continue
            
            # If we have more than n, we need to select n that maximize min distance
            # For simplicity, just take first n (they're in a reasonable order)
            # Better: use a simple greedy selection
            selected = []
            remaining = list(candidates)
            
            for _ in range(min(n, len(remaining))):
                if not remaining:
                    break
                # Pick the point that maximizes minimum distance to already selected
                best_idx = 0
                best_min_dist = -1
                for i, pt in enumerate(remaining):
                    if not selected:
                        min_d = 1.0
                    else:
                        min_d = min(np.sqrt((pt[0]-s[0])**2 + (pt[1]-s[1])**2) for s in selected)
                    if min_d > best_min_dist:
                        best_min_dist = min_d
                        best_idx = i
                selected.append(remaining.pop(best_idx))
            
            if len(selected) == n:
                # Compute sum of radii for this configuration
                x_arr = np.array([p[0] for p in selected])
                y_arr = np.array([p[1] for p in selected])
                dist_b = np.minimum(np.minimum(x_arr, 1-x_arr), np.minimum(y_arr, 1-y_arr))
                dx_m = x_arr[:, None] - x_arr[None, :]
                dy_m = y_arr[:, None] - y_arr[None, :]
                dm = np.sqrt(dx_m**2 + dy_m**2)
                np.fill_diagonal(dm, np.inf)
                min_dc = np.min(dm, axis=1)
                r = np.minimum(dist_b, min_dc / 2.0)
                s = np.sum(np.maximum(r, 0))
                if s > best_sum:
                    best_sum = s
                    best_pts = selected
        
        if best_pts is None:
            # Fallback to grid
            side = int(np.ceil(np.sqrt(n)))
            best_pts = []
            idx = 0
            for i in range(side):
                for j in range(side):
                    if idx >= n:
                        break
                    best_pts.append(((i + 0.5) / side, (j + 0.5) / side))
                    idx += 1
            while len(best_pts) < n:
                best_pts.append((0.5, 0.5))
        
        return best_pts

    # Generate initial hexagonal configuration
    init_points = hex_init(N)
    x0 = np.array([p[0] for p in init_points])
    y0 = np.array([p[1] for p in init_points])

    params0 = np.zeros(2 * N)
    params0[0::2] = x0
    params0[1::2] = y0

    # Bounds for centers: [0, 1]
    bounds = [(0.0, 1.0)] * (2 * N)

    best_params = params0.copy()
    best_obj = objective(params0)

    # Phase 1: Optimize from hexagonal init with Powell
    try:
        res = minimize(
            objective,
            params0,
            method='Powell',
            bounds=bounds,
            options={'maxiter': 8000, 'xtol': 1e-10, 'ftol': 1e-10}
        )
        
        if res.fun < best_obj:
            best_params = res.x
            best_obj = res.fun
            
    except Exception as e:
        pass

    # Phase 2: Random restarts with diverse initializations
    np.random.seed(42)
    n_restarts = 8
    
    for restart in range(n_restarts):
        if time.time() > deadline - 15:
            break
            
        # Different initialization strategies
        if restart % 3 == 0:
            # Random uniform
            params_rand = np.random.rand(2 * N)
            params_rand = 0.05 + 0.9 * params_rand
        elif restart % 3 == 1:
            # Perturbed hex init
            params_rand = params0.copy()
            params_rand += np.random.normal(0, 0.05, size=2 * N)
            params_rand = np.clip(params_rand, 0.0, 1.0)
        else:
            # Cluster-based: group circles into clusters
            n_clusters = 4
            cluster_centers = np.random.rand(n_clusters, 2)
            params_rand = np.zeros(2 * N)
            for i in range(N):
                c = i % n_clusters
                params_rand[2*i] = cluster_centers[c, 0] + np.random.normal(0, 0.1)
                params_rand[2*i+1] = cluster_centers[c, 1] + np.random.normal(0, 0.1)
            params_rand = np.clip(params_rand, 0.0, 1.0)
        
        try:
            res = minimize(
                objective,
                params_rand,
                method='Powell',
                bounds=bounds,
                options={'maxiter': 4000, 'xtol': 1e-10, 'ftol': 1e-10}
            )
            
            if res.fun < best_obj:
                best_params = res.x
                best_obj = res.fun
                
        except Exception as e:
            pass

    # Phase 3: Final polish - tighter optimization on the best solution
    if time.time() < deadline - 10:
        try:
            res = minimize(
                objective,
                best_params,
                method='Powell',
                bounds=bounds,
                options={'maxiter': 10000, 'xtol': 1e-12, 'ftol': 1e-12}
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
