"""Circle packing in perimeter-4 rectangle: 21 circles, maximize sum of radii."""
import numpy as np
from scipy.optimize import minimize
import time

N = 21
TIME_LIMIT = 110.0  # safety margin before 120s kill

def construct():
    start_time = time.time()
    
    def elapsed():
        return time.time() - start_time
    
    # Variables: x[0:N], y[0:N], r[0:N], W (width), H (height)
    # Constraints: 2*(W + H) = 4 => H = 2 - W
    # So we optimize over x, y, r, W
    # Circle i: (x_i - r_i >= 0), (x_i + r_i <= W), (y_i - r_i >= 0), (y_i + r_i <= H)
    # Pairwise: (x_i - x_j)^2 + (y_i - y_j)^2 >= (r_i + r_j)^2
    
    def unpack(params):
        x = params[0:N]
        y = params[N:2*N]
        r = params[2*N:3*N]
        W = params[3*N]
        H = 2.0 - W
        return x, y, r, W, H
    
    def objective(params):
        x, y, r, W, H = unpack(params)
        # Barrier for non-overlap: penalize pairs that overlap
        # Use soft penalty: -sum over pairs of max(0, (r_i+r_j) - dist_ij)^2
        penalty = 0.0
        for i in range(N):
            for j in range(i+1, N):
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                dist = np.sqrt(dx*dx + dy*dy)
                min_dist = r[i] + r[j]
                if dist < min_dist:
                    penalty += (min_dist - dist)**2
        # Barrier for boundaries
        for i in range(N):
            if x[i] - r[i] < 0: penalty += (r[i] - x[i])**2
            if x[i] + r[i] > W: penalty += (x[i] + r[i] - W)**2
            if y[i] - r[i] < 0: penalty += (r[i] - y[i])**2
            if y[i] + r[i] > H: penalty += (y[i] + r[i] - H)**2
        return -np.sum(r) + 10.0 * penalty
    
    def gradient(params):
        x, y, r, W, H = unpack(params)
        grad = np.zeros_like(params)
        
        # Objective gradient: -sum(r) => grad_r = -1
        grad[2*N:3*N] = -1.0
        
        # Penalty gradient
        pen_weight = 10.0
        
        # Pairwise penalties
        for i in range(N):
            for j in range(i+1, N):
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                dist = np.sqrt(dx*dx + dy*dy)
                min_dist = r[i] + r[j]
                if dist < min_dist and dist > 1e-12:
                    diff = min_dist - dist
                    # d/dx_i of (min_dist - dist)^2 = 2*(min_dist-dist) * d(dist)/dx_i * (-1)
                    # d(dist)/dx_i = dx/dist
                    factor = 2.0 * diff / dist
                    grad[0+i] += pen_weight * factor * dx
                    grad[N+i] += pen_weight * factor * dy
                    grad[0+j] -= pen_weight * factor * dx
                    grad[N+j] -= pen_weight * factor * dy
                    grad[2*N+i] += pen_weight * 2.0 * diff
                    grad[2*N+j] += pen_weight * 2.0 * diff
        
        # Boundary penalties
        for i in range(N):
            if x[i] - r[i] < 0:
                diff = r[i] - x[i]
                grad[0+i] += pen_weight * 2.0 * diff * (-1)
                grad[2*N+i] += pen_weight * 2.0 * diff
            if x[i] + r[i] > W:
                diff = x[i] + r[i] - W
                grad[0+i] += pen_weight * 2.0 * diff
                grad[2*N+i] += pen_weight * 2.0 * diff
                grad[3*N] += pen_weight * 2.0 * diff * (-1)
            if y[i] - r[i] < 0:
                diff = r[i] - y[i]
                grad[N+i] += pen_weight * 2.0 * diff * (-1)
                grad[2*N+i] += pen_weight * 2.0 * diff
            if y[i] + r[i] > H:
                diff = y[i] + r[i] - H
                grad[N+i] += pen_weight * 2.0 * diff
                grad[2*N+i] += pen_weight * 2.0 * diff
                # H = 2 - W, so dH/dW = -1
                grad[3*N] += pen_weight * 2.0 * diff * 1.0  # since H decreases with W
        
        return grad
    
    # Initial guess: grid-based
    def initial_guess(W_init):
        H_init = 2.0 - W_init
        # Use a grid layout
        n_cols = int(np.ceil(np.sqrt(N * W_init / H_init)))
        n_rows = int(np.ceil(N / n_cols))
        x = np.zeros(N)
        y = np.zeros(N)
        r = np.zeros(N)
        for idx in range(N):
            col = idx % n_cols
            row = idx // n_cols
            x[idx] = (col + 0.5) * W_init / n_cols
            y[idx] = (row + 0.5) * H_init / n_rows
            r[idx] = 0.5 * min(W_init / n_cols, H_init / n_rows) - 0.01
        params = np.concatenate([x, y, r, [W_init]])
        return params
    
    # Try multiple initial aspect ratios
    best_params = None
    best_val = -np.inf
    
    for W_init in [0.8, 1.0, 1.2, 1.4, 1.6]:
        if elapsed() > TIME_LIMIT - 10:
            break
        params0 = initial_guess(W_init)
        res = minimize(objective, params0, jac=gradient, method='L-BFGS-B',
                       options={'maxiter': 500, 'ftol': 1e-10, 'gtol': 1e-8})
        val = -res.fun
        # Check validity
        x, y, r, W, H = unpack(res.x)
        if W <= 0 or H <= 0:
            continue
        valid = True
        for i in range(N):
            if x[i] - r[i] < -1e-6 or x[i] + r[i] > W + 1e-6 or y[i] - r[i] < -1e-6 or y[i] + r[i] > H + 1e-6:
                valid = False
                break
            if r[i] <= 0:
                valid = False
                break
        if valid:
            for i in range(N):
                for j in range(i+1, N):
                    dx = x[i] - x[j]
                    dy = y[i] - y[j]
                    dist = np.sqrt(dx*dx + dy*dy)
                    if dist < r[i] + r[j] - 1e-6:
                        valid = False
                        break
                if not valid:
                    break
        if valid and val > best_val:
            best_val = val
            best_params = res.x.copy()
    
    if best_params is None:
        # Fallback to seed
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3) for i in range(side) for j in range(side)]
        return np.array(cells[:N])
    
    x, y, r, W, H = unpack(best_params)
    
    # Polish: local refinement with tighter tolerances
    if elapsed() < TIME_LIMIT - 5:
        for _ in range(3):
            if elapsed() > TIME_LIMIT - 5:
                break
            res2 = minimize(objective, best_params, jac=gradient, method='L-BFGS-B',
                           options={'maxiter': 1000, 'ftol': 1e-12, 'gtol': 1e-10})
            val2 = -res2.fun
            x2, y2, r2, W2, H2 = unpack(res2.x)
            if W2 <= 0 or H2 <= 0:
                break
            valid2 = True
            for i in range(N):
                if x2[i] - r2[i] < -1e-8 or x2[i] + r2[i] > W2 + 1e-8 or y2[i] - r2[i] < -1e-8 or y2[i] + r2[i] > H2 + 1e-8:
                    valid2 = False
                    break
                if r2[i] <= 0:
                    valid2 = False
                    break
            if valid2:
                for i in range(N):
                    for j in range(i+1, N):
                        dx = x2[i] - x2[j]
                        dy = y2[i] - y2[j]
                        dist = np.sqrt(dx*dx + dy*dy)
                        if dist < r2[i] + r2[j] - 1e-8:
                            valid2 = False
                            break
                    if not valid2:
                        break
            if valid2 and val2 > best_val:
                best_val = val2
                best_params = res2.x.copy()
                x, y, r, W, H = x2, y2, r2, W2, H2
    
    # Ensure strict non-overlap by slightly shrinking radii if needed
    for i in range(N):
        for j in range(i+1, N):
            dx = x[i] - x[j]
            dy = y[i] - y[j]
            dist = np.sqrt(dx*dx + dy*dy)
            if dist < r[i] + r[j]:
                # Shrink both
                shrink = (r[i] + r[j] - dist) / 2 + 1e-9
                r[i] -= shrink
                r[j] -= shrink
        # Clamp to boundaries
        r[i] = min(r[i], x[i], W - x[i], y[i], H - y[i])
        r[i] = max(r[i], 1e-6)
    
    result = np.column_stack([x, y, r])
    return result
