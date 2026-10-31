import numpy as np
import time
from scipy.optimize import minimize

N = 26

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Leave 10s buffer

    def unpack(z):
        x = z[0::3]
        y = z[1::3]
        r = z[2::3]
        return x, y, r

    def objective(z):
        _, _, r = unpack(z)
        return -np.sum(r)

    def make_constraints():
        def boundary_and_radius(z):
            x, y, r = unpack(z)
            c = np.zeros(5 * N)
            c[0::5] = x - r
            c[1::5] = 1 - x - r
            c[2::5] = y - r
            c[3::5] = 1 - y - r
            c[4::5] = r - 1e-6
            return c

        def non_overlap(z):
            x, y, r = unpack(z)
            c = []
            for i in range(N):
                for j in range(i + 1, N):
                    dist_sq = (x[i] - x[j])**2 + (y[i] - y[j])**2
                    rad_sum_sq = (r[i] + r[j])**2
                    c.append(dist_sq - rad_sum_sq)
            return np.array(c)

        return [
            {'type': 'ineq', 'fun': boundary_and_radius},
            {'type': 'ineq', 'fun': non_overlap}
        ]

    constraints = make_constraints()
    bounds = []
    for i in range(N):
        bounds.append((0.0, 1.0))
        bounds.append((0.0, 1.0))
        bounds.append((1e-6, 0.5))

    best_val = -np.inf
    best_z = None

    def try_optimize(z0):
        nonlocal best_val, best_z
        if time.time() > deadline:
            return
        try:
            res = minimize(objective, z0, method='SLSQP', bounds=bounds, 
                           constraints=constraints, options={'maxiter': 300, 'ftol': 1e-9})
            
            if res.success or res.status == 0:
                x, y, r = unpack(res.x)
                # Quick validity check
                if np.any(x < r - 1e-7) or np.any(x + r > 1 + 1e-7) or np.any(y < r - 1e-7) or np.any(y + r > 1 + 1e-7):
                    return
                
                # Check overlaps
                for i in range(N):
                    for j in range(i + 1, N):
                        if (x[i] - x[j])**2 + (y[i] - y[j])**2 < (r[i] + r[j])**2 - 1e-7:
                            return
                
                val = np.sum(r)
                if val > best_val:
                    best_val = val
                    best_z = res.x.copy()
        except Exception:
            pass

    # 1. Structured Starts
    # 5x6 Grid, skip 4 corners
    cols, rows = 5, 6
    x_grid = (np.arange(cols) + 0.5) / cols
    y_grid = (np.arange(rows) + 0.5) / rows
    r_base = 0.5 / max(cols, rows) * 0.9
    
    indices_all = np.arange(cols * rows)
    # Corners: 0, cols-1, (rows-1)*cols, (rows-1)*cols + cols-1
    corners = [0, cols-1, (rows-1)*cols, (rows-1)*cols + cols-1]
    indices_struct = [i for i in indices_all if i not in corners]
    # We need 26. We have 30-4=26. Perfect.
    
    x_init = np.array([x_grid[i % cols] for i in indices_struct])
    y_init = np.array([y_grid[i // cols] for i in indices_struct])
    r_init = np.full(N, r_base)
    
    z0 = np.zeros(3 * N)
    z0[0::3] = x_init
    z0[1::3] = y_init
    z0[2::3] = r_init
    try_optimize(z0)
    
    # 6x5 Grid, skip 4 corners
    cols, rows = 6, 5
    x_grid = (np.arange(cols) + 0.5) / cols
    y_grid = (np.arange(rows) + 0.5) / rows
    r_base = 0.5 / max(cols, rows) * 0.9
    
    indices_all = np.arange(cols * rows)
    corners = [0, cols-1, (rows-1)*cols, (rows-1)*cols + cols-1]
    indices_struct = [i for i in indices_all if i not in corners]
    
    x_init = np.array([x_grid[i % cols] for i in indices_struct])
    y_init = np.array([y_grid[i // cols] for i in indices_struct])
    r_init = np.full(N, r_base)
    
    z0 = np.zeros(3 * N)
    z0[0::3] = x_init
    z0[1::3] = y_init
    z0[2::3] = r_init
    try_optimize(z0)

    # 2. Random Perturbations of Grids
    num_random_restarts = 50
    for k in range(num_random_restarts):
        if time.time() > deadline:
            break
        
        if k % 2 == 0:
            cols, rows = 5, 6
        else:
            cols, rows = 6, 5
            
        x_grid = (np.arange(cols) + 0.5) / cols
        y_grid = (np.arange(rows) + 0.5) / rows
        r_base = 0.5 / max(cols, rows) * 0.9
        
        indices = np.random.choice(cols * rows, N, replace=False)
        
        x_init = np.array([x_grid[i % cols] for i in indices])
        y_init = np.array([y_grid[i // cols] for i in indices])
        r_init = np.full(N, r_base)
        
        # Perturb
        noise_scale = 0.05 * r_base
        x_init += np.random.uniform(-noise_scale, noise_scale, N)
        y_init += np.random.uniform(-noise_scale, noise_scale, N)
        
        z0 = np.zeros(3 * N)
        z0[0::3] = x_init
        z0[1::3] = y_init
        z0[2::3] = r_init
        try_optimize(z0)

    if best_z is None:
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3) for i in range(side) for j in range(side)]
        return np.array(cells[:N])
    
    x, y, r = unpack(best_z)
    return np.column_stack((x, y, r))
