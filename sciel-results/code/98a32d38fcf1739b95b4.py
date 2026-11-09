import numpy as np
import time
from scipy.optimize import minimize
from scipy.optimize import differential_evolution

N = 21

def construct():
    start_time = time.time()
    deadline = 115  # Safety margin from 150s limit
    
    best_score = 0
    best_solution = None
    
    # Variables: [x1, y1, r1, ..., x21, y21, r21]
    # Total 63 variables
    
    def cost_and_validity(params):
        x = params[0::3]
        y = params[1::3]
        r = params[2::3]
        
        sum_r = np.sum(r)
        
        # Boundary Violations
        viol_x_low = np.maximum(0, r - x)
        viol_x_high = np.maximum(0, r - (1 - x))
        viol_y_low = np.maximum(0, r - y)
        viol_y_high = np.maximum(0, r - (1 - y))
        
        p_boundary = np.sum(viol_x_low**2) + np.sum(viol_x_high**2) + \
                     np.sum(viol_y_low**2) + np.sum(viol_y_high**2)
        
        # Overlap Violations - vectorized
        dx = x[:, None] - x[None, :]
        dy = y[:, None] - y[None, :]
        dist = np.sqrt(dx**2 + dy**2)
        min_dist = r[:, None] + r[None, :]
        # Only consider upper triangle
        iu = np.triu_indices(N, k=1)
        viol = np.maximum(0, min_dist[iu] - dist[iu])
        p_overlap = np.sum(viol**2)
        
        penalty = p_boundary + p_overlap
        valid = penalty < 1e-8
        
        obj_val = -sum_r + 100.0 * penalty
        return obj_val, sum_r, valid

    def objective(params):
        val, s, v = cost_and_validity(params)
        return val

    def get_solution_from_params(params, shrink_factor=1.0):
        """Extract and validate solution from parameters"""
        x = params[0::3]
        y = params[1::3]
        r = params[2::3] * shrink_factor
        
        # Check validity
        params_check = np.array([x[i], y[i], r[i]] for i in range(N)).flatten()
        _, sum_r, valid = cost_and_validity(params_check)
        
        if valid:
            return np.column_stack((x, y, r)), sum_r
        return None, 0

    def try_improve(params0, max_iter=300, timeout=15):
        """Try to improve a starting configuration"""
        nonlocal best_score, best_solution, start_time, deadline
        
        # Nelder-Mead
        try:
            res = minimize(objective, params0, method='Nelder-Mead', 
                           options={'maxiter': max_iter, 'xatol': 1e-7, 'fatol': 1e-7, 'adaptive': True})
            
            # Check time
            if time.time() > deadline - 5:
                return
            
            # Try to extract valid solution
            sol, score = get_solution_from_params(res.x)
            if sol is not None and score > best_score:
                best_score = score
                best_solution = sol
            
            # Also try with slight shrinkage for robustness
            for shrink in [0.999, 0.998, 0.995]:
                sol, score = get_solution_from_params(res.x, shrink)
                if sol is not None and score > best_score:
                    best_score = score
                    best_solution = sol
                    
        except Exception:
            pass

    # Generate diverse initial configurations
    def make_hex_init(n_rows, n_cols, offset_x=0, offset_y=0):
        """Create hexagonal packing initialization"""
        x_list, y_list, r_list = [], [], []
        
        if n_rows * n_cols < N:
            # Add extra circles
            extra = N - n_rows * n_cols
            for i in range(extra):
                x_list.append(0.5 + offset_x)
                y_list.append(0.5 + offset_y)
                r_list.append(0.03)
            return np.array([x, y, r] for x, y, r in zip(x_list, y_list, r_list)).flatten()
        
        for i in range(n_rows):
            for j in range(n_cols):
                if len(x_list) >= N:
                    break
                x = (j + 0.5) / n_cols + (0.5 / n_cols if i % 2 == 1 else 0) * 0.3 + offset_x
                y = (i + 0.5) / n_rows + offset_y
                r = 0.5 / max(n_cols, n_rows) * 0.85
                x_list.append(x)
                y_list.append(y)
                r_list.append(r)
        
        return np.array([x, y, r] for x, y, r in zip(x_list[:N], y_list[:N], r_list[:N])).flatten()

    def make_grid_init(rows, cols):
        """Create grid initialization"""
        x_list, y_list, r_list = [], [], []
        for i in range(rows):
            for j in range(cols):
                if len(x_list) >= N:
                    break
                x = (j + 0.5) / cols
                y = (i + 0.5) / rows
                r = min(0.5/cols, 0.5/rows) * 0.9
                x_list.append(x)
                y_list.append(y)
                r_list.append(r)
        return np.array([x, y, r] for x, y, r in zip(x_list[:N], y_list[:N], r_list[:N])).flatten()

    def make_random_init():
        """Create random initialization with reasonable spacing"""
        x = np.random.uniform(0.1, 0.9, N)
        y = np.random.uniform(0.1, 0.9, N)
        r = np.random.uniform(0.03, 0.07, N)
        return np.array([xi, yi, ri] for xi, yi, ri in zip(x, y, r)).flatten()

    # Try multiple initialization strategies
    initializations = [
        make_grid_init(5, 5),
        make_grid_init(4, 6),
        make_grid_init(6, 4),
        make_grid_init(3, 7),
        make_grid_init(7, 3),
        make_hex_init(4, 5),
        make_hex_init(5, 4),
        make_hex_init(3, 7),
        make_hex_init(7, 3),
        make_random_init(),
        make_random_init(),
        make_random_init(),
    ]

    # Shuffle and try each
    np.random.shuffle(initializations)

    for idx, params0 in enumerate(initializations):
        if time.time() > deadline - 10:
            break
            
        # First pass: Nelder-Mead
        try_improve(params0, max_iter=400, timeout=12)
        
        if time.time() > deadline - 15:
            break

    # If we have a good solution, try to refine with more iterations
    if best_solution is not None and time.time() < deadline - 20:
        # Reconstruct params from best solution
        x, y, r = best_solution[:, 0], best_solution[:, 1], best_solution[:, 2]
        params = np.array([xi, yi, ri] for xi, yi, ri in zip(x, y, r)).flatten()
        
        # Try more aggressive refinement
        try_improve(params, max_iter=800, timeout=25)

    # If no valid solution found, return a simple grid
    if best_solution is None:
        # Simple 5x5 grid with 21 circles
        side = 5
        x_list, y_list, r_list = [], [], []
        for i in range(side):
            for j in range(side):
                if len(x_list) >= N:
                    break
                x_list.append((j + 0.5) / side)
                y_list.append((i + 0.5) / side)
                r_list.append(0.5 / side - 0.01)
        
        best_solution = np.column_stack((x_list, y_list, r_list))
    
    return best_solution
