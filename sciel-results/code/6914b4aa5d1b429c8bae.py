import numpy as np
import time
import random
from scipy.optimize import minimize

N = 21

def construct():
    start_time = time.time()
    deadline = start_time + 140.0 # 10s safety margin

    best_score = -np.inf
    best_result = None

    def get_score(x):
        """
        x: array of shape (3*N + 1)
        x[0:N]: xs
        x[N:2N]: ys
        x[2N:3N]: rs
        x[3N]: W
        H = 2 - W
        Returns sum of radii if valid, else -inf (or penalized value for optimization)
        """
        xs = x[:N]
        ys = x[N:2*N]
        rs = x[2*N:3*N]
        W = x[3*N]
        H = 2.0 - W
        
        if W <= 0 or H <= 0:
            return -np.inf
            
        # Check bounds
        for i in range(N):
            if xs[i] < rs[i] or xs[i] > W - rs[i] or ys[i] < rs[i] or ys[i] > H - rs[i]:
                return -np.inf
                
        # Check overlaps
        for i in range(N):
            for j in range(i+1, N):
                dist_sq = (xs[i]-xs[j])**2 + (ys[i]-ys[j])**2
                rad_sum_sq = (rs[i]+rs[j])**2
                if dist_sq < rad_sum_sq - 1e-9:
                    return -np.inf
                    
        return np.sum(rs)

    def objective_and_grad(x):
        """
        Objective: -sum(rs) + Penalty
        Penalty is designed to be smooth.
        """
        xs = x[:N]
        ys = x[N:2*N]
        rs = x[2*N:3*N]
        W = x[3*N]
        H = 2.0 - W
        
        if W <= 0.1 or H <= 0.1:
            return 1e10, np.zeros_like(x)

        obj = -np.sum(rs)
        grad = np.zeros_like(x)
        grad[2*N:3*N] = -1.0 # Gradient of -sum(rs)
        
        penalty_weight = 1000.0
        
        # 1. Boundary Penalties
        # For each circle i, we want x_i >= r_i, x_i <= W-r_i, etc.
        # Penalty: 0.5 * max(0, r_i - x_i)^2 + 0.5 * max(0, r_i - (W-x_i))^2 ...
        for i in range(N):
            # Left wall: x_i >= r_i -> violation = max(0, r_i - x_i)
            v = rs[i] - xs[i]
            if v > 0:
                obj += 0.5 * penalty_weight * v * v
                grad[i] += penalty_weight * v
                grad[2*N+i] += penalty_weight * v # d(r_i - x_i)/dr_i = 1
                
            # Right wall: x_i <= W - r_i -> violation = max(0, r_i - (W - x_i)) = max(0, r_i + x_i - W)
            v = rs[i] + xs[i] - W
            if v > 0:
                obj += 0.5 * penalty_weight * v * v
                grad[i] += penalty_weight * v
                grad[2*N+i] += penalty_weight * v
                grad[3*N] -= penalty_weight * v # d(r_i + x_i - W)/dW = -1
                
            # Bottom wall: y_i >= r_i
            v = rs[i] - ys[i]
            if v > 0:
                obj += 0.5 * penalty_weight * v * v
                grad[N+i] += penalty_weight * v
                grad[2*N+i] += penalty_weight * v
                
            # Top wall: y_i <= H - r_i = (2-W) - r_i -> violation = max(0, r_i + y_i - (2-W)) = max(0, r_i + y_i + W - 2)
            v = rs[i] + ys[i] + W - 2.0
            if v > 0:
                obj += 0.5 * penalty_weight * v * v
                grad[N+i] += penalty_weight * v
                grad[2*N+i] += penalty_weight * v
                grad[3*N] += penalty_weight * v # d(r_i + y_i + W - 2)/dW = 1

        # 2. Overlap Penalties
        for i in range(N):
            for j in range(i+1, N):
                dx = xs[i] - xs[j]
                dy = ys[i] - ys[j]
                d_sq = dx*dx + dy*dy
                r_sum = rs[i] + rs[j]
                
                if d_sq < 1e-12:
                    # Handle singularity
                    continue
                    
                d = np.sqrt(d_sq)
                v = r_sum - d
                if v > 0:
                    # Penalty: 0.5 * W_p * v^2
                    obj += 0.5 * penalty_weight * v * v
                    
                    # Gradients
                    # dv/dx_i = -dx/d
                    # dv/dx_j = dx/d
                    # dv/dr_i = 1
                    # dv/dr_j = 1
                    
                    factor = penalty_weight * v / d
                    grad[i] += -factor * dx
                    grad[j] += factor * dx
                    grad[N+i] += -factor * dy
                    grad[N+j] += factor * dy
                    grad[2*N+i] += penalty_weight * v
                    grad[2*N+j] += penalty_weight * v

        return obj, grad

    def grid_init(W, H, cols=None, rows=None):
        if cols is None:
            cols = int(np.ceil(np.sqrt(N)))
        if rows is None:
            rows = int(np.ceil(N / cols))
        
        # Ensure we have enough cells
        if cols * rows < N:
            rows = int(np.ceil(N / cols))
            
        dx = W / cols
        dy = H / rows
        
        xs = np.zeros(N)
        ys = np.zeros(N)
        rs = np.zeros(N)
        
        k = 0
        for i in range(cols):
            for j in range(rows):
                if k >= N:
                    break
                xs[k] = (i + 0.5) * dx
                ys[k] = (j + 0.5) * dy
                # Initial radius slightly smaller than half the cell size
                rs[k] = 0.4 * min(dx, dy)
                k += 1
        if k < N:
             # Fill remaining with small circles if grid wasn't big enough (shouldn't happen with ceil)
             for m in range(k, N):
                 xs[m] = W/2
                 ys[m] = H/2
                 rs[m] = 0.01
        
        return np.concatenate([xs, ys, rs, [W]])

    def random_init(W, H):
        xs = np.random.uniform(0.1, W-0.1, N)
        ys = np.random.uniform(0.1, H-0.1, N)
        rs = np.random.uniform(0.05, 0.15, N)
        return np.concatenate([xs, ys, rs, [W]])

    def optimize(x0, maxiter=500):
        try:
            res = minimize(
                objective_and_grad, 
                x0, 
                method='L-BFGS-B', 
                jac=True,
                bounds=[(1e-5, 1.9)] * (3*N) + [(0.5, 1.5)], # W bounds
                options={'maxiter': maxiter, 'ftol': 1e-12, 'gtol': 1e-8, 'maxls': 50}
            )
            return res.x
        except Exception as e:
            return x0

    def relax_radii(x, n_iter=50):
        """
        Fix positions, maximize radii iteratively.
        """
        x = x.copy()
        xs = x[:N].copy()
        ys = x[N:2*N].copy()
        rs = x[2*N:3*N].copy()
        W = x[3*N]
        H = 2.0 - W
        
        for _ in range(n_iter):
            improved = False
            for i in range(N):
                # Max radius allowed by walls
                r_wall = min(xs[i], W - xs[i], ys[i], H - ys[i])
                r_max = r_wall
                # Max radius allowed by neighbors
                for j in range(N):
                    if i == j:
                        continue
                    d = np.hypot(xs[i] - xs[j], ys[i] - ys[j])
                    r_neighbor = d - rs[j]
                    if r_neighbor < r_max:
                        r_max = r_neighbor
                r_max = max(r_max, 1e-8)
                if r_max > rs[i] + 1e-9:
                    rs[i] = r_max
                    improved = True
            if not improved:
                break
        x[2*N:3*N] = rs
        return x

    # Main Search Loop
    attempts = 0
    max_attempts = 50
    
    # Aspect ratios to try
    aspect_ratios = [1.0, 1.2, 0.83, 1.5, 0.67, 1.3, 0.77, 1.1, 0.9, 1.4]
    
    while time.time() < deadline and attempts < max_attempts:
        attempts += 1
        
        # Select W
        if attempts <= len(aspect_ratios):
            W = aspect_ratios[attempts - 1]
        else:
            W = np.random.uniform(0.6, 1.4)
        
        H = 2.0 - W
        if H < 0.5:
            W = 1.5
            H = 0.5
            
        # Initialization
        if attempts % 3 == 0:
            x0 = random_init(W, H)
        else:
            # Grid init with varying structures
            if attempts % 2 == 0:
                x0 = grid_init(W, H, cols=5, rows=5)
            else:
                x0 = grid_init(W, H, cols=4, rows=6)
        
        # Optimization
        x_opt = optimize(x0, maxiter=300)
        
        # Post-process: Relax radii
        x_relax = relax_radii(x_opt, n_iter=30)
        
        # Check validity and score
        score = get_score(x_relax)
        
        if score > best_score:
            best_score = score
            best_result = x_relax.copy()
            
        # If we have time, do a second pass with refined initial guess from previous best
        if best_result is not None and time.time() < deadline - 20 and attempts > 5:
             # Perturb best result and optimize again
             x_perturb = best_result.copy()
             noise = 0.01
             x_perturb[:N] += np.random.normal(0, noise, N)
             x_perturb[N:2*N] += np.random.normal(0, noise, N)
             # Ensure W is still valid
             x_perturb[3*N] = np.clip(x_perturb[3*N], 0.5, 1.5)
             
             x_opt2 = optimize(x_perturb, maxiter=100)
             x_relax2 = relax_radii(x_opt2, n_iter=20)
             score2 = get_score(x_relax2)
             if score2 > best_score:
                 best_score = score2
                 best_result = x_relax2.copy()

    # Final Polish
    if best_result is not None and time.time() < deadline - 5:
        best_result = relax_radii(best_result, n_iter=50)
        s = get_score(best_result)
        if s > best_score:
            best_score = s

    if best_result is None:
        # Fallback: Simple grid
        side = 5
        W = 1.0
        H = 1.0
        cells = []
        k=0
        for i in range(side):
            for j in range(side):
                if k >= N: break
                cx = (i + 0.5) * (W/side)
                cy = (j + 0.5) * (H/side)
                r = 0.4 * (W/side)
                cells.append([cx, cy, r])
                k+=1
        return np.array(cells)

    xs = best_result[:N]
    ys = best_result[N:2*N]
    rs = best_result[2*N:3*N]
    result = np.column_stack((xs, ys, rs))
    return result
