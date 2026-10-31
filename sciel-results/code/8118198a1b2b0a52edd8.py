"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 21

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize

def construct():
    """Pack 21 circles in a perimeter-4 rectangle, maximizing sum of radii."""
    best_value = 0.0
    best_solution = None
    
    def evaluate_solution(params):
        """
        params: array of length 21*3 + 1 = 64
        First 63 elements: (x_i, y_i, r_i) for each circle i
        Last element: width w of rectangle (height = 2 - w)
        """
        w = params[-1]
        h = 2.0 - w
        
        # Clamp w to valid range
        if w <= 0 or h <= 0:
            return -1e6, None
        
        circles = params[:63].reshape(21, 3)
        xs = circles[:, 0]
        ys = circles[:, 1]
        rs = circles[:, 2]
        
        # Radii must be positive
        if np.any(rs <= 0):
            return -1e6, None
        
        # Penalty for circles outside rectangle
        penalty = 0.0
        for i in range(21):
            x, y, r = xs[i], ys[i], rs[i]
            # Outside left
            if x < r:
                penalty += (r - x) ** 2
            # Outside right
            if x > w - r:
                penalty += (x - (w - r)) ** 2
            # Outside bottom
            if y < r:
                penalty += (r - y) ** 2
            # Outside top
            if y > h - r:
                penalty += (y - (h - r)) ** 2
        
        # Penalty for overlapping circles
        for i in range(21):
            for j in range(i + 1, 21):
                dx = xs[i] - xs[j]
                dy = ys[i] - ys[j]
                dist = np.sqrt(dx * dx + dy * dy)
                min_dist = rs[i] + rs[j]
                if dist < min_dist:
                    penalty += (min_dist - dist) ** 2
        
        value = np.sum(rs)
        # Penalize invalid solutions
        if penalty > 1e-8:
            return value - 100 * penalty, None
        return value, params.copy()
    
    def objective(params):
        val, _ = evaluate_solution(params)
        return -val  # minimize negative value
    
    # Multiple random restarts
    np.random.seed(42)
    
    for restart in range(50):
        # Initialize with reasonable random configuration
        # Rectangle width around 1.0 to 1.5
        w_init = np.random.uniform(0.8, 1.2)
        h_init = 2.0 - w_init
        
        # Place circles in a rough grid pattern with some randomness
        cols = 5
        rows = 5
        init_params = np.zeros(64)
        
        for i in range(21):
            col = i % cols
            row = i // cols
            x = (col + 0.5) / cols * w_init + np.random.uniform(-0.05, 0.05)
            y = (row + 0.5) / rows * h_init + np.random.uniform(-0.05, 0.05)
            r = min(w_init / cols, h_init / rows) * 0.45 * np.random.uniform(0.8, 1.2)
            init_params[3*i] = x
            init_params[3*i+1] = y
            init_params[3*i+2] = r
        
        init_params[-1] = w_init
        
        # Use Nelder-Mead optimization
        try:
            result = minimize(objective, init_params, method='Nelder-Mead',
                             options={'maxiter': 2000, 'xatol': 1e-6, 'fatol': 1e-6})
            val, sol = evaluate_solution(result.x)
            if sol is not None and val > best_value:
                best_value = val
                best_solution = sol
        except Exception:
            pass
    
    # If no valid solution found, return a simple grid
    if best_solution is None:
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3) for i in range(side) for j in range(side)]
        return np.array(cells[:N])
    
    # Extract circles from best solution
    circles = best_solution[:63].reshape(21, 3)
    return circles
# EVOLVE-BLOCK-END
