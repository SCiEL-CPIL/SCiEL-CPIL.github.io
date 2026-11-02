import numpy as np
import time
from scipy.optimize import minimize

N = 21

def construct():
    start_time = time.time()
    deadline = start_time + 110.0
    
    best_val = -np.inf
    best_sol = None
    
    def softplus(x, alpha=100.0):
        return np.log1p(np.exp(alpha * x)) / alpha

    def objective(theta):
        w = theta[0]
        h = 2.0 - w
        
        circles = theta[1:].reshape(N, 3)
        x = circles[:, 0]
        y = circles[:, 1]
        r = circles[:, 2]
        
        # Objective: Maximize sum(r) -> Minimize -sum(r)
        val = -np.sum(r)
        
        # Boundary penalties
        # Violations:
        # x < r  => r - x > 0
        # x > w-r => x - (w-r) > 0
        # y < r  => r - y > 0
        # y > h-r => y - (h-r) > 0
        
        v_x_min = r - x
        v_x_max = x - (w - r)
        v_y_min = r - y
        v_y_max = y - (h - r)
        
        alpha_b = 50.0
        pen_x_min = softplus(v_x_min, alpha_b)
        pen_x_max = softplus(v_x_max, alpha_b)
        pen_y_min = softplus(v_y_min, alpha_b)
        pen_y_max = softplus(v_y_max, alpha_b)
        
        boundary_penalty = np.sum(pen_x_min**2) + np.sum(pen_x_max**2) + \
                           np.sum(pen_y_min**2) + np.sum(pen_y_max**2)
        
        # Overlap penalties
        dx = x[:, None] - x[None, :]
        dy = y[:, None] - y[None, :]
        d = np.sqrt(dx**2 + dy**2)
        
        r_sum = r[:, None] + r[None, :]
        
        v = r_sum - d
        
        mask = np.triu(np.ones((N, N), dtype=bool), k=1)
        v_pairs = v[mask]
        
        alpha_o = 50.0
        pen_pairs = softplus(v_pairs, alpha_o)
        overlap_penalty = np.sum(pen_pairs**2)
        
        # Weights: 
        # If penalty is too low, circles overlap. 
        # If too high, radii stay small.
        # Start with 10.0.
        total_val = val + 10.0 * boundary_penalty + 10.0 * overlap_penalty
        
        return total_val

    bounds = [(0.5, 1.5)]
    for _ in range(N):
        bounds.append((0.0, 1.5)) # x
        bounds.append((0.0, 1.5)) # y
        bounds.append((0.01, 0.4)) # r

    np.random.seed(42)
    n_restarts = 150
    
    for i in range(n_restarts):
        if time.time() > deadline:
            break
            
        if i % 3 == 0:
            # Grid Init
            side = 5
            x_g, y_g, r_g = [], [], []
            count = 0
            for i_g in range(5):
                for j_g in range(5):
                    if count >= N: break
                    cx = (i_g + 0.5) / 5.0
                    cy = (j_g + 0.5) / 5.0
                    x_g.append(cx)
                    y_g.append(cy)
                    r_g.append(0.09)
                    count += 1
            x_g = np.array(x_g) + np.random.normal(0, 0.01, N)
            y_g = np.array(y_g) + np.random.normal(0, 0.01, N)
            r_g = np.array(r_g) + np.random.normal(0, 0.005, N)
            w0 = 1.0
            theta0 = np.concatenate(([w0], x_g, y_g, r_g))
        elif i % 3 == 1:
            # Random Init
            w0 = np.random.uniform(0.9, 1.1)
            h0 = 2.0 - w0
            x0 = np.random.uniform(0.1, w0 - 0.1, N)
            y0 = np.random.uniform(0.1, h0 - 0.1, N)
            r0 = np.random.uniform(0.05, 0.12, N)
            theta0 = np.concatenate(([w0], x0, y0, r0))
        else:
            # Hex-like Init
            # Approximate hex packing in square
            w0 = 1.0
            h0 = 1.0
            x0, y0, r0 = [], [], []
            r_init = 0.09
            # 5 rows
            for row in range(5):
                y_pos = (row + 0.5) / 5.0
                offset = 0.0 if row % 2 == 0 else 0.1
                for col in range(5):
                    if len(x0) >= N: break
                    x_pos = (col + 0.5) / 5.0 + offset
                    # Wrap around or clip? Clip.
                    if x_pos > 1.0: x_pos = x_pos - 0.2
                    x0.append(x_pos)
                    y0.append(y_pos)
                    r0.append(r_init)
            x0 = np.array(x0) + np.random.normal(0, 0.01, N)
            y0 = np.array(y0) + np.random.normal(0, 0.01, N)
            r0 = np.array(r0) + np.random.normal(0, 0.005, N)
            theta0 = np.concatenate(([w0], x0, y0, r0))
            
        # Clip initial values to bounds to help optimizer
        theta0 = np.clip(theta0, [b[0] for b in bounds], [b[1] for b in bounds])
        
        try:
            res = minimize(
                objective, 
                theta0, 
                method='L-BFGS-B', 
                bounds=bounds,
                options={'maxiter': 500, 'ftol': 1e-10, 'gtol': 1e-6}
            )
        except Exception:
            continue
        
        if res.fun < np.inf:
            w_opt = res.x[0]
            h_opt = 2.0 - w_opt
            circles_opt = res.x[1:].reshape(N, 3)
            
            x_o = circles_opt[:, 0]
            y_o = circles_opt[:, 1]
            r_o = circles_opt[:, 2]
            
            # Strict validation
            valid = True
            if np.any(x_o < r_o - 1e-5) or np.any(x_o > w_opt - r_o + 1e-5) or \
               np.any(y_o < r_o - 1e-5) or np.any(y_o > h_opt - r_o + 1e-5):
                valid = False
            
            if valid:
                for i in range(N):
                    for j in range(i+1, N):
                        dist = np.sqrt((x_o[i]-x_o[j])**2 + (y_o[i]-y_o[j])**2)
                        if dist < r_o[i] + r_o[j] - 1e-5:
                            valid = False
                            break
                    if not valid: break
            
            if valid:
                sum_r = np.sum(r_o)
                if sum_r > best_val:
                    best_val = sum_r
                    best_sol = res.x.copy()
                    
    if best_sol is None:
        # Fallback
        side = 5
        cells = []
        for i in range(side):
            for j in range(side):
                if len(cells) >= N: break
                cells.append([(i+0.5)/side, (j+0.5)/side, 0.09])
        return np.array(cells)
    
    circles_best = best_sol[1:].reshape(N, 3)
    return circles_best
