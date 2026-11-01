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

    def soft_min(values, beta=50.0):
        """Soft minimum via log-sum-exp trick."""
        min_val = np.min(values)
        return min_val - np.log(np.sum(np.exp(-beta * (values - min_val)))) / beta

    def clearance_objective(v):
        x, y, r = unpack(v)
        
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
        if np.any(x < r - 1e-9) or np.any(1 - x < r - 1e-9):
            return False
        if np.any(y < r - 1e-9) or np.any(1 - y < r - 1e-9):
            return False
        if np.any(r < -1e-9):
            return False
            
        dx = x[:, np.newaxis] - x[np.newaxis, :]
        dy = y[:, np.newaxis] - y[np.newaxis, :]
        dist = np.sqrt(dx**2 + dy**2)
        r_sum = r[:, np.newaxis] + r[np.newaxis, :]
        
        dist[np.diag_indices(N)] = np.inf
        if np.any(dist < r_sum - 1e-9):
            return False
        return True

    def physics_relaxation(x, y, r, n_iters=500, dt=0.001):
        """Physics-based push-apart relaxation to resolve overlaps."""
        x = x.copy()
        y = y.copy()
        r = r.copy()
        
        for it in range(n_iters):
            if time.time() > deadline - 35:
                break
            # Compute forces
            fx = np.zeros(N)
            fy = np.zeros(N)
            
            for i in range(N):
                # Boundary forces
                if x[i] < r[i]:
                    fx[i] += (r[i] - x[i]) * 10
                if 1 - x[i] < r[i]:
                    fx[i] -= (r[i] - (1 - x[i])) * 10
                if y[i] < r[i]:
                    fy[i] += (r[i] - y[i]) * 10
                if 1 - y[i] < r[i]:
                    fy[i] -= (r[i] - (1 - y[i])) * 10
                
                # Overlap forces
                for j in range(N):
                    if i == j:
                        continue
                    dx = x[i] - x[j]
                    dy = y[i] - y[j]
                    d = np.sqrt(dx**2 + dy**2) + 1e-10
                    overlap = r[i] + r[j] - d
                    if overlap > 0:
                        force = overlap * 5.0
                        fx[i] += (dx / d) * force
                        fy[i] += (dy / d) * force
            
            # Update positions
            x += fx * dt
            y += fy * dt
            
            # Clamp to boundaries
            x = np.clip(x, r, 1 - r)
            y = np.clip(y, r, 1 - r)
        
        return x, y, r

    def get_voronoi_radii(x, y):
        points = np.column_stack((x, y))
        
        n_boundary = 20
        boundary_points = []
        boundary_points.extend([(0, i/(n_boundary-1)) for i in range(n_boundary)])
        boundary_points.extend([(1, i/(n_boundary-1)) for i in range(n_boundary)])
        boundary_points.extend([(i/(n_boundary-1), 0) for i in range(n_boundary)])
        boundary_points.extend([(i/(n_boundary-1), 1) for i in range(n_boundary)])
        
        all_points = np.vstack([points, np.array(boundary_points)])
        
        try:
            vor = Voronoi(all_points)
        except QhullError:
            r = np.zeros(N)
            for i in range(N):
                d_bound = min(x[i], 1-x[i], y[i], 1-y[i])
                d_circles = np.inf
                for j in range(N):
                    if i != j:
                        d = np.sqrt((x[i]-x[j])**2 + (y[i]-y[j])**2)
                        d_circles = min(d_circles, d)
                r[i] = min(d_bound, d_circles/2)
            return r
        
        radii = np.zeros(N)
        for i in range(N):
            region_idx = vor.point_region[i]
            region = vor.regions[region_idx]
            if len(region) == 0 or -1 in region:
                radii[i] = min(x[i], 1-x[i], y[i], 1-y[i])
            else:
                vertices = vor.vertices[region]
                dists = np.sqrt((vertices[:, 0] - x[i])**2 + (vertices[:, 1] - y[i])**2)
                radii[i] = np.min(dists)
        
        return radii

    def get_initial_guesses():
        guesses = []
        
        # Hexagonal packing - primary initialization
        x_hex = []
        y_hex = []
        for row in range(6):
            for col in range(6):
                if len(x_hex) >= N:
                    break
                xx = 0.5 + col * 0.16 + (row % 2) * 0.08
                yy = 0.5 + row * 0.16
                if 0.05 <= xx <= 0.95 and 0.05 <= yy <= 0.95:
                    x_hex.append(xx)
                    y_hex.append(yy)
        
        if len(x_hex) >= N:
            x_hex = np.array(x_hex[:N])
            y_hex = np.array(y_hex[:N])
            r_hex = get_voronoi_radii(x_hex, y_hex)
            v_hex = np.concatenate([x_hex, y_hex, r_hex])
            guesses.append(v_hex)
        
        # Grid initialization
        side = int(np.ceil(np.sqrt(N)))
        xs = (np.arange(side) + 0.5) / side
        ys = (np.arange(side) + 0.5) / side
        grid_x, grid_y = np.meshgrid(xs, ys)
        grid_x = grid_x.flatten()[:N]
        grid_y = grid_y.flatten()[:N]
        
        v_grid_r = get_voronoi_radii(grid_x, grid_y)
        v_grid = np.concatenate([grid_x, grid_y, v_grid_r])
        guesses.append(v_grid)
        
        # Perturbed grid
        v_pert = v_grid + np.random.randn(3*N) * 0.01
        guesses.append(v_pert)
        
        # Random initializations with physics relaxation
        for _ in range(8):
            x = np.random.rand(N)
            y = np.random.rand(N)
            r = np.random.rand(N) * 0.12 + 0.02
            v = np.concatenate([x, y, r])
            
            # Apply physics relaxation
            x_rel, y_rel, r_rel = physics_relaxation(x, y, r, n_iters=200)
            v_rel = np.concatenate([x_rel, y_rel, r_rel])
            guesses.append(v_rel)
            
            guesses.append(v)
        
        # Additional hexagonal variants with offsets
        for offset in [0.02, 0.04, 0.06]:
            x_hex2 = []
            y_hex2 = []
            for row in range(6):
                for col in range(6):
                    if len(x_hex2) >= N:
                        break
                    xx = 0.5 + col * 0.16 + (row % 2) * 0.08 + offset
                    yy = 0.5 + row * 0.16 + offset
                    if 0.05 <= xx <= 0.95 and 0.05 <= yy <= 0.95:
                        x_hex2.append(xx)
                        y_hex2.append(yy)
            if len(x_hex2) >= N:
                x_hex2 = np.array(x_hex2[:N])
                y_hex2 = np.array(y_hex2[:N])
                r_hex2 = get_voronoi_radii(x_hex2, y_hex2)
                v_hex2 = np.concatenate([x_hex2, y_hex2, r_hex2])
                guesses.append(v_hex2)
        
        return guesses

    guesses = get_initial_guesses()
    
    bounds = []
    for i in range(N):
        bounds.append((0, 1))
    for i in range(N):
        bounds.append((0, 1))
    for i in range(N):
        bounds.append((0, 0.5))

    # Phase 1: Coarse optimization with multiple starts
    for v0 in guesses:
        if time.time() > deadline - 40:
            break
            
        try:
            res = minimize(clearance_objective, v0, method='L-BFGS-B', bounds=bounds, 
                          options={'maxiter': 800, 'ftol': 1e-8, 'gtol': 1e-6})
            
            if time.time() > deadline - 40:
                break
                
            if res.success:
                v_opt = res.x
                v_opt = np.clip(v_opt, [b[0] for b in bounds], [b[1] for b in bounds])
                
                if is_valid(v_opt):
                    score = np.sum(v_opt[2*N:3*N])
                    if score > best_score:
                        best_score = score
                        best_solution = v_opt.copy()
        except Exception:
            pass

    # Phase 2: High-precision polish on best solution with higher penalty
    if best_solution is not None and time.time() < deadline - 10:
        lam_polish = 5000.0
        
        def polish_objective(v):
            x, y, r = unpack(v)
            sum_r = np.sum(r)
            
            pen_x = np.sum(np.maximum(0, r - x)**2)
            pen_1mx = np.sum(np.maximum(0, r - (1 - x))**2)
            pen_y = np.sum(np.maximum(0, r - y)**2)
            pen_1my = np.sum(np.maximum(0, r - (1 - y))**2)
            pen_bound = pen_x + pen_1mx + pen_y + pen_1my
            
            dx = x[:, np.newaxis] - x[np.newaxis, :]
            dy = y[:, np.newaxis] - y[np.newaxis, :]
            dist_sq = dx**2 + dy**2
            
            r_sum = r[:, np.newaxis] + r[np.newaxis, :]
            r_sum_sq = r_sum**2
            
            mask = np.triu(np.ones((N, N), dtype=bool), k=1)
            
            violations = np.maximum(0, r_sum_sq[mask] - dist_sq[mask])
            pen_overlap = np.sum(violations**2)
            
            total_penalty = pen_bound + pen_overlap
            
            return -sum_r + lam_polish * total_penalty

        try:
            res_polish = minimize(polish_objective, best_solution, method='L-BFGS-B', 
                                  bounds=bounds, options={'maxiter': 3000, 'ftol': 1e-12, 'gtol': 1e-10})
            
            if time.time() < deadline - 3:
                if res_polish.success:
                    v_polished = res_polish.x
                    v_polished = np.clip(v_polished, [b[0] for b in bounds], [b[1] for b in bounds])
                    
                    if is_valid(v_polished):
                        score_polished = np.sum(v_polished[2*N:3*N])
                        if score_polished > best_score:
                            best_score = score_polished
                            best_solution = v_polished.copy()
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
