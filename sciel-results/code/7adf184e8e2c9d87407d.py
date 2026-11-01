import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial import Voronoi, QhullError

N = 32

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin
    
    best_score = 0.0
    best_solution = None
    
    # Penalty weight
    lam = 1000.0
    
    def unpack(v):
        x = v[0:N]
        y = v[N:2*N]
        r = v[2*N:3*N]
        return x, y, r

    def clearance_objective(v):
        """
        Voronoi/clearance-based objective: for each circle, the maximum 
        radius it can have is the minimum distance to boundaries and to 
        other circles' boundaries. We maximize sum of these clearances.
        This is a smooth surrogate that allows global adjustment.
        """
        x, y, r = unpack(v)
        
        # For each circle i, compute minimum clearance to:
        # 1. Boundaries: min(x_i, 1-x_i, y_i, 1-y_i)
        # 2. Other circles: min_j (dist(i,j) - r_j) for j != i
        # The max radius for circle i is min of these clearances.
        # We want to maximize sum of min(clearances), but this is nonsmooth.
        # Instead, we maximize sum of radii while penalizing violations.
        
        # Actually, let's use a different approach: 
        # Maximize sum of radii with smooth penalties.
        # But use the clearance insight for initialization.
        
        sum_r = np.sum(r)
        
        # Boundary penalties
        pen_x = np.sum(np.maximum(0, r - x)**2)
        pen_1mx = np.sum(np.maximum(0, r - (1 - x))**2)
        pen_y = np.sum(np.maximum(0, r - y)**2)
        pen_1my = np.sum(np.maximum(0, r - (1 - y))**2)
        pen_bound = pen_x + pen_1mx + pen_y + pen_1my
        
        # Overlap penalties
        dx = x[:, np.newaxis] - x[np.newaxis, :]
        dy = y[:, np.newaxis] - y[np.newaxis, :]
        dist_sq = dx**2 + dy**2
        
        r_sum = r[:, np.newaxis] + r[np.newaxis, :]
        r_sum_sq = r_sum**2
        
        mask = np.triu(np.ones((N, N), dtype=bool), k=1)
        
        violations = np.maximum(0, r_sum_sq[mask] - dist_sq[mask])
        pen_overlap = np.sum(violations**2)
        
        total_penalty = pen_bound + pen_overlap
        
        return -sum_r + lam * total_penalty

    def is_valid(v):
        x, y, r = unpack(v)
        # Check boundaries
        if np.any(x < r - 1e-9) or np.any(1 - x < r - 1e-9):
            return False
        if np.any(y < r - 1e-9) or np.any(1 - y < r - 1e-9):
            return False
        if np.any(r < -1e-9):
            return False
            
        # Check overlaps
        dx = x[:, np.newaxis] - x[np.newaxis, :]
        dy = y[:, np.newaxis] - y[np.newaxis, :]
        dist = np.sqrt(dx**2 + dy**2)
        r_sum = r[:, np.newaxis] + r[np.newaxis, :]
        
        # dist >= r_sum for i != j
        # Subtract large number on diagonal to ignore
        dist[np.diag_indices(N)] = np.inf
        if np.any(dist < r_sum - 1e-9):
            return False
        return True

    def get_voronoi_radii(x, y):
        """
        Given circle centers, compute the maximum radius each circle can have
        based on Voronoi cell clearance.
        """
        points = np.column_stack((x, y))
        
        # Add boundary points to capture boundary constraints
        # Use a grid of boundary points
        n_boundary = 20
        boundary_points = []
        # Left boundary
        boundary_points.extend([(0, i/(n_boundary-1)) for i in range(n_boundary)])
        # Right boundary
        boundary_points.extend([(1, i/(n_boundary-1)) for i in range(n_boundary)])
        # Bottom boundary
        boundary_points.extend([(i/(n_boundary-1), 0) for i in range(n_boundary)])
        # Top boundary
        boundary_points.extend([(i/(n_boundary-1), 1) for i in range(n_boundary)])
        
        all_points = np.vstack([points, np.array(boundary_points)])
        
        try:
            vor = Voronoi(all_points)
        except QhullError:
            # Fallback: use grid-based clearance
            r = np.zeros(N)
            for i in range(N):
                # Distance to boundaries
                d_bound = min(x[i], 1-x[i], y[i], 1-y[i])
                # Distance to other circles
                d_circles = np.inf
                for j in range(N):
                    if i != j:
                        d = np.sqrt((x[i]-x[j])**2 + (y[i]-y[j])**2)
                        d_circles = min(d_circles, d)
                r[i] = min(d_bound, d_circles/2)
            return r
        
        # For each original point, find the nearest vertex in its Voronoi cell
        # The distance to the nearest vertex is the maximum inscribed circle radius
        radii = np.zeros(N)
        for i in range(N):
            region_idx = vor.point_region[i]
            region = vor.regions[region_idx]
            if len(region) == 0 or -1 in region:
                # Unbounded region - use boundary distance
                radii[i] = min(x[i], 1-x[i], y[i], 1-y[i])
            else:
                vertices = vor.vertices[region]
                dists = np.sqrt((vertices[:, 0] - x[i])**2 + (vertices[:, 1] - y[i])**2)
                radii[i] = np.min(dists)
        
        return radii

    def get_initial_guesses():
        guesses = []
        
        # Random
        for _ in range(5):
            x = np.random.rand(N)
            y = np.random.rand(N)
            r = np.random.rand(N) * 0.1 + 0.01
            v = np.concatenate([x, y, r])
            guesses.append(v)
            
        # Grid
        side = int(np.ceil(np.sqrt(N)))
        xs = (np.arange(side) + 0.5) / side
        ys = (np.arange(side) + 0.5) / side
        grid_x, grid_y = np.meshgrid(xs, ys)
        grid_x = grid_x.flatten()[:N]
        grid_y = grid_y.flatten()[:N]
        
        # Use Voronoi to get better initial radii
        v_grid_r = get_voronoi_radii(grid_x, grid_y)
        v_grid = np.concatenate([grid_x, grid_y, v_grid_r])
        guesses.append(v_grid)
        
        # Perturbed Grid
        v_pert = v_grid + np.random.randn(3*N) * 0.01
        guesses.append(v_pert)
        
        # Hexagonal packing initialization
        # For 32 circles, use a hexagonal-like arrangement
        hex_guesses = []
        for offset in [0.0, 0.05, 0.1]:
            x_hex = []
            y_hex = []
            for row in range(6):
                for col in range(6):
                    if len(x_hex) >= N:
                        break
                    xx = 0.5 + col * 0.16 + (row % 2) * 0.08 + offset
                    yy = 0.5 + row * 0.16 + offset
                    if 0.05 <= xx <= 0.95 and 0.05 <= yy <= 0.95:
                        x_hex.append(xx)
                        y_hex.append(yy)
            if len(x_hex) >= N:
                x_hex = np.array(x_hex[:N])
                y_hex = np.array(y_hex[:N])
                r_hex = get_voronoi_radii(x_hex, y_hex)
                v_hex = np.concatenate([x_hex, y_hex, r_hex])
                guesses.append(v_hex)
        
        return guesses

    guesses = get_initial_guesses()
    
    bounds = []
    for i in range(N):
        bounds.append((0, 1)) # x
    for i in range(N):
        bounds.append((0, 1)) # y
    for i in range(N):
        bounds.append((0, 0.5)) # r

    # Phase 1: Coarse optimization
    for v0 in guesses:
        if time.time() > deadline - 30:  # Leave 30s for polish
            break
            
        try:
            res = minimize(clearance_objective, v0, method='L-BFGS-B', bounds=bounds, 
                          options={'maxiter': 1000, 'ftol': 1e-8, 'gtol': 1e-6})
            
            if time.time() > deadline - 30:
                break
                
            if res.success:
                v_opt = res.x
                v_opt = np.clip(v_opt, [b[0] for b in bounds], [b[1] for b in bounds])
                
                if is_valid(v_opt):
                    score = np.sum(v_opt[2*N:3*N])
                    if score > best_score:
                        best_score = score
                        best_solution = v_opt
        except Exception:
            pass

    # Phase 2: High-precision polish on best solution
    if best_solution is not None and time.time() < deadline - 5:
        try:
            res_polish = minimize(clearance_objective, best_solution, method='L-BFGS-B', 
                                  bounds=bounds, options={'maxiter': 2000, 'ftol': 1e-12, 'gtol': 1e-10})
            
            if time.time() < deadline - 2:
                if res_polish.success:
                    v_polished = res_polish.x
                    v_polished = np.clip(v_polished, [b[0] for b in bounds], [b[1] for b in bounds])
                    
                    if is_valid(v_polished):
                        score_polished = np.sum(v_polished[2*N:3*N])
                        if score_polished > best_score:
                            best_score = score_polished
                            best_solution = v_polished
        except Exception:
            pass

    # If no valid solution found, return a trivial valid one
    if best_solution is None:
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3) for i in range(side) for j in range(side)]
        return np.array(cells[:N])

    x, y, r = unpack(best_solution)
    result = np.column_stack((x, y, r))
    return result
