import numpy as np
import time
from scipy.optimize import minimize

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

    def objective(v):
        x, y, r = unpack(v)
        
        # Sum of radii
        sum_r = np.sum(r)
        
        # Boundary penalties
        # x >= r, 1-x >= r, y >= r, 1-y >= r
        # Violation is max(0, r - x) etc.
        pen_x = np.sum(np.maximum(0, r - x)**2)
        pen_1mx = np.sum(np.maximum(0, r - (1 - x))**2)
        pen_y = np.sum(np.maximum(0, r - y)**2)
        pen_1my = np.sum(np.maximum(0, r - (1 - y))**2)
        pen_bound = pen_x + pen_1mx + pen_y + pen_1my
        
        # Overlap penalties
        # dist(i,j) >= r(i) + r(j)
        # dist^2 >= (r(i)+r(j))^2
        # Using direct distance is safer for gradient, but squared is faster.
        # Let's use direct distance for penalty to be smooth at 0? 
        # max(0, r_i + r_j - dist) is not differentiable at dist = r_i+r_j.
        # Using squared penalty: max(0, (r_i+r_j)^2 - dist^2) is differentiable?
        # No, max(0, u) is not differentiable at u=0.
        # But L-BFGS-B handles non-smoothness reasonably well with finite differences or quasi-Newton steps.
        # To be smoother, we can use a soft-plus or just accept the kink.
        # Let's use the squared overlap amount.
        
        # Compute pairwise distances
        # X: (N, 1), Y: (N, 1)
        dx = x[:, np.newaxis] - x[np.newaxis, :]
        dy = y[:, np.newaxis] - y[np.newaxis, :]
        dist_sq = dx**2 + dy**2
        
        # Sum of radii pairs
        r_sum = r[:, np.newaxis] + r[np.newaxis, :]
        r_sum_sq = r_sum**2
        
        # Violation: r_sum_sq - dist_sq > 0
        # We only care about i < j
        # Create a mask for i < j
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

    # Initial guesses
    # 1. Random
    # 2. Grid
    # 3. Hex-like
    
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
        grid_r = np.full(N, 0.5 / side * 0.9)
        v_grid = np.concatenate([grid_x, grid_y, grid_r])
        guesses.append(v_grid)
        
        # Perturbed Grid
        v_pert = v_grid + np.random.randn(3*N) * 0.01
        guesses.append(v_pert)
        
        return guesses

    guesses = get_initial_guesses()
    
    bounds = []
    for i in range(N):
        bounds.append((0, 1)) # x
    for i in range(N):
        bounds.append((0, 1)) # y
    for i in range(N):
        bounds.append((0, 0.5)) # r

    for v0 in guesses:
        if time.time() > deadline:
            break
            
        try:
            res = minimize(objective, v0, method='L-BFGS-B', bounds=bounds, options={'maxiter': 2000, 'ftol': 1e-10, 'gtol': 1e-8})
            
            if time.time() > deadline:
                break
                
            if res.success:
                v_opt = res.x
                # Clamp to bounds just in case
                v_opt = np.clip(v_opt, [b[0] for b in bounds], [b[1] for b in bounds])
                
                if is_valid(v_opt):
                    score = np.sum(v_opt[2*N:3*N])
                    if score > best_score:
                        best_score = score
                        best_solution = v_opt
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
