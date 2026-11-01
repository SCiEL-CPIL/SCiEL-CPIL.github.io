import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial import Voronoi

N = 32

def construct():
    start_time = time.time()
    deadline = start_time + 130  # safety margin
    
    best_score = 0.0
    best_solution = None
    
    # Smooth penalty parameters
    lam = 500.0
    beta = 50.0  # smoothness parameter for soft-max

    def unpack(v):
        x = v[0:N]
        y = v[N:2*N]
        r = v[2*N:3*N]
        return x, y, r

    def soft_max(u, b):
        """Smooth approximation of max(0, u) using log-sum-exp."""
        # soft_max(u) ≈ (1/b) * log(1 + exp(b*u)) for u near 0
        # For large positive u, ≈ u; for large negative u, ≈ 0
        # Use stable log-sum-exp
        u_clipped = np.clip(u, -50/b, 50/b)
        return (1.0 / b) * np.log1p(np.exp(b * u_clipped))

    def objective(v):
        x, y, r = unpack(v)
        
        # Sum of radii
        sum_r = np.sum(r)
        
        # Smooth boundary penalties
        # Violations: r - x < 0 means x < r (violation)
        # We want to penalize max(0, r - x)
        viol_x = r - x
        viol_1mx = r - (1 - x)
        viol_y = r - y
        viol_1my = r - (1 - y)
        
        pen_x = np.sum(soft_max(viol_x, beta)**2)
        pen_1mx = np.sum(soft_max(viol_1mx, beta)**2)
        pen_y = np.sum(soft_max(viol_y, beta)**2)
        pen_1my = np.sum(soft_max(viol_1my, beta)**2)
        pen_bound = pen_x + pen_1mx + pen_y + pen_1my
        
        # Smooth overlap penalties
        dx = x[:, np.newaxis] - x[np.newaxis, :]
        dy = y[:, np.newaxis] - y[np.newaxis, :]
        dist_sq = dx**2 + dy**2
        
        r_sum = r[:, np.newaxis] + r[np.newaxis, :]
        r_sum_sq = r_sum**2
        
        mask = np.triu(np.ones((N, N), dtype=bool), k=1)
        
        # Violation: r_sum_sq - dist_sq > 0 means circles overlap
        viol_overlap = r_sum_sq[mask] - dist_sq[mask]
        pen_overlap = np.sum(soft_max(viol_overlap, beta)**2)
        
        total_penalty = pen_bound + pen_overlap
        
        return -sum_r + lam * total_penalty

    def is_valid(v, tol=1e-8):
        x, y, r = unpack(v)
        # Check boundaries
        if np.any(x < r - tol) or np.any(1 - x < r - tol):
            return False
        if np.any(y < r - tol) or np.any(1 - y < r - tol):
            return False
        if np.any(r < -tol):
            return False
            
        # Check overlaps
        dx = x[:, np.newaxis] - x[np.newaxis, :]
        dy = y[:, np.newaxis] - y[np.newaxis, :]
        dist = np.sqrt(dx**2 + dy**2)
        r_sum = r[:, np.newaxis] + r[np.newaxis, :]
        
        dist[np.diag_indices(N)] = np.inf
        if np.any(dist < r_sum - tol):
            return False
        return True

    def voronoi_init(seed=42):
        """Generate initial guess using Voronoi cells."""
        rng = np.random.RandomState(seed)
        # Start with a perturbed grid
        side = int(np.ceil(np.sqrt(N)))
        xs = (np.arange(side) + 0.5) / side
        ys = (np.arange(side) + 0.5) / side
        grid_x, grid_y = np.meshgrid(xs, ys)
        x = grid_x.flatten()[:N] + rng.randn(N) * 0.02
        y = grid_y.flatten()[:N] + rng.randn(N) * 0.02
        x = np.clip(x, 0.05, 0.95)
        y = np.clip(y, 0.05, 0.95)
        
        # Compute Voronoi inradii
        points = np.column_stack([x, y])
        # Add boundary points for proper Voronoi
        boundary_pts = np.array([
            [0, 0], [1, 0], [0, 1], [1, 1],
            [0.5, 0], [0.5, 1], [0, 0.5], [1, 0.5],
            [0.25, 0], [0.75, 0], [0.25, 1], [0.75, 1],
            [0, 0.25], [0, 0.75], [1, 0.25], [1, 0.75],
            [0.5, 0.5]
        ])
        all_pts = np.vstack([points, boundary_pts])
        
        try:
            vor = Voronoi(all_pts)
            # For each original point, find its Voronoi region
            # The inradius is the minimum distance to any edge of the cell
            radii = np.zeros(N)
            for i in range(N):
                # Find the Voronoi region for point i
                region_idx = vor.point_region[i]
                region = vor.regions[region_idx]
                if len(region) == 0:
                    radii[i] = 0.05
                    continue
                
                # Get vertices of the region
                verts = vor.vertices[region]
                if len(verts) == 0:
                    radii[i] = 0.05
                    continue
                
                # Compute inradius: minimum distance from center to any edge
                # For a polygon, inradius = min distance to edges
                min_dist = np.inf
                n_verts = len(verts)
                for j in range(n_verts):
                    v1 = verts[j]
                    v2 = verts[(j + 1) % n_verts]
                    # Distance from point i to line segment v1-v2
                    p = points[i]
                    d = v2 - v1
                    if np.dot(d, d) == 0:
                        continue
                    t = np.dot(p - v1, d) / np.dot(d, d)
                    t = np.clip(t, 0, 1)
                    closest = v1 + t * d
                    dist = np.linalg.norm(p - closest)
                    min_dist = min(min_dist, dist)
                
                # Also check distance to unit square boundary
                min_dist = min(min_dist, x[i], 1 - x[i], y[i], 1 - y[i])
                radii[i] = max(0.01, min_dist * 0.95)
        except Exception:
            radii = np.full(N, 0.08)
        
        r = np.clip(radii, 0.01, 0.4)
        return np.concatenate([x, y, r])

    def get_initial_guesses():
        guesses = []
        
        # Voronoi-based initializations with different seeds
        for seed in [42, 123, 456, 789]:
            if time.time() > deadline - 5:
                break
            guesses.append(voronoi_init(seed))
        
        # Grid
        side = int(np.ceil(np.sqrt(N)))
        xs = (np.arange(side) + 0.5) / side
        ys = (np.arange(side) + 0.5) / side
        grid_x, grid_y = np.meshgrid(xs, ys)
        grid_x = grid_x.flatten()[:N]
        grid_y = grid_y.flatten()[:N]
        grid_r = np.full(N, 0.5 / side * 0.85)
        v_grid = np.concatenate([grid_x, grid_y, grid_r])
        guesses.append(v_grid)
        
        # Random
        for _ in range(3):
            if time.time() > deadline - 5:
                break
            x = np.random.rand(N) * 0.8 + 0.1
            y = np.random.rand(N) * 0.8 + 0.1
            r = np.random.rand(N) * 0.08 + 0.02
            v = np.concatenate([x, y, r])
            guesses.append(v)
        
        return guesses

    guesses = get_initial_guesses()
    
    bounds = []
    for i in range(N):
        bounds.append((0, 1))
    for i in range(N):
        bounds.append((0, 1))
    for i in range(N):
        bounds.append((0, 0.5))
    
    lower = np.array([b[0] for b in bounds])
    upper = np.array([b[1] for b in bounds])

    for v0 in guesses:
        if time.time() > deadline - 10:
            break
            
        try:
            res = minimize(objective, v0, method='L-BFGS-B', bounds=bounds, 
                          options={'maxiter': 3000, 'ftol': 1e-12, 'gtol': 1e-10, 'maxfun': 50000})
            
            if time.time() > deadline - 5:
                break
                
            v_opt = np.clip(res.x, lower, upper)
            
            # Polish pass with tighter tolerances
            if time.time() < deadline - 5:
                res2 = minimize(objective, v_opt, method='L-BFGS-B', bounds=bounds,
                               options={'maxiter': 2000, 'ftol': 1e-14, 'gtol': 1e-12, 'maxfun': 30000})
                v_opt = np.clip(res2.x, lower, upper)
            
            if is_valid(v_opt):
                score = np.sum(v_opt[2*N:3*N])
                if score > best_score:
                    best_score = score
                    best_solution = v_opt.copy()
        except Exception:
            pass

    # Final polish on best solution
    if best_solution is not None and time.time() < deadline - 3:
        try:
            res_final = minimize(objective, best_solution, method='L-BFGS-B', bounds=bounds,
                                options={'maxiter': 5000, 'ftol': 1e-15, 'gtol': 1e-13, 'maxfun': 100000})
            v_final = np.clip(res_final.x, lower, upper)
            if is_valid(v_final):
                score_final = np.sum(v_final[2*N:3*N])
                if score_final > best_score:
                    best_score = score_final
                    best_solution = v_final
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
