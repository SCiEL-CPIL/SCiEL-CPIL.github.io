# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time

def squares_in_square10() -> tuple:
    """
    Maximizes the sum of side lengths of 10 non-overlapping axis-aligned squares
    inside the unit square [0, 1]^2.
    
    Uses a multi-start Nelder-Mead optimization with a penalty method for
    non-overlap constraints.
    """
    n = 10
    
    def objective(params):
        # params is a flat array of length 3*n: [x1, y1, s1, ..., x10, y10, s10]
        # Reshape for easier access
        p = params.reshape(n, 3)
        xs = p[:, 0]
        ys = p[:, 1]
        ss = p[:, 2]
        
        # Check bounds and positivity
        # If invalid, return a large penalty
        if np.any(ss <= 1e-9):
            return 1e10
        if np.any(xs - ss/2 < -1e-7) or np.any(xs + ss/2 > 1 + 1e-7):
            return 1e10
        if np.any(ys - ss/2 < -1e-7) or np.any(ys + ss/2 > 1 + 1e-7):
            return 1e10
            
        # Calculate overlap penalty
        penalty = 0.0
        for i in range(n):
            for j in range(i + 1, n):
                dx = abs(xs[i] - xs[j])
                dy = abs(ys[i] - ys[j])
                req_dist = (ss[i] + ss[j]) / 2.0
                
                # If they overlap in both x and y, there is a conflict
                if dx < req_dist and dy < req_dist:
                    # Penalty proportional to the square of the overlap intrusion
                    # This encourages separation
                    ox = req_dist - dx
                    oy = req_dist - dy
                    penalty += (ox * ox) + (oy * oy)
        
        # We want to maximize sum(ss), so minimize -sum(ss) + penalty
        # The penalty weight is tuned to be large enough to prevent overlap
        # but small enough to allow growth
        return -np.sum(ss) + 1000.0 * penalty

    def is_valid(centers, sides):
        if len(centers) != n or len(sides) != n:
            return False
        if not np.all(np.isfinite(centers)) or not np.all(np.isfinite(sides)):
            return False
        if np.any(sides <= 0):
            return False
        
        xs = centers[:, 0]
        ys = centers[:, 1]
        ss = sides
        
        # Boundary check
        if np.any(xs - ss/2 < -1e-7) or np.any(xs + ss/2 > 1 + 1e-7):
            return False
        if np.any(ys - ss/2 < -1e-7) or np.any(ys + ss/2 > 1 + 1e-7):
            return False
            
        # Overlap check
        for i in range(n):
            for j in range(i + 1, n):
                dx = abs(xs[i] - xs[j])
                dy = abs(ys[i] - ys[j])
                req = (ss[i] + ss[j]) / 2.0
                if dx < req - 1e-7 and dy < req - 1e-7:
                    return False
        return True

    best_centers = None
    best_sides = None
    best_score = -np.inf
    
    start_time = time.time()
    deadline = 550.0 # Safety margin
    
    # Strategy: Multi-start Nelder-Mead
    rng = np.random.default_rng(42)
    
    # Define a few specific initial guesses to help convergence
    initial_guesses = []
    
    # 1. Random small squares
    for _ in range(20):
        ss_init = rng.uniform(0.1, 0.2, n)
        xs_init = rng.uniform(ss_init/2 + 0.01, 1 - ss_init/2 - 0.01, n)
        ys_init = rng.uniform(ss_init/2 + 0.01, 1 - ss_init/2 - 0.01, n)
        params_init = np.column_stack((xs_init, ys_init, ss_init)).flatten()
        initial_guesses.append(params_init)
        
    # 2. Grid-based: 5x2 grid (side 0.2)
    ss_init = np.full(n, 0.2)
    xs_init = []
    ys_init = []
    for i in range(5):
        for j in range(2):
            xs_init.append(0.1 + i * 0.2)
            ys_init.append(0.1 + j * 0.8)
    params_init = np.column_stack((np.array(xs_init), np.array(ys_init), ss_init)).flatten()
    initial_guesses.append(params_init)
    
    # 3. Grid-based: 3x3 grid + 1 (side 1/3 approx, but scaled down to fit)
    # Actually, just use random for the rest.
    
    for params_init in initial_guesses:
        if time.time() > deadline:
            break
            
        try:
            res = minimize(objective, params_init, method='Nelder-Mead',
                           options={'maxiter': 2000, 'xatol': 1e-9, 'fatol': 1e-9, 'adaptive': True})
            
            if res.success:
                params_opt = res.x.reshape(n, 3)
                centers = params_opt[:, :2]
                sides = params_opt[:, 2]
                
                if is_valid(centers, sides):
                    score = np.sum(sides)
                    if score > best_score:
                        best_score = score
                        best_centers = centers.copy()
                        best_sides = sides.copy()
        except Exception:
            continue

    if best_centers is None:
        # Fallback: 5x2 grid of 0.2 squares
        centers = np.array([
            [0.1, 0.1], [0.3, 0.1], [0.5, 0.1], [0.7, 0.1], [0.9, 0.1],
            [0.1, 0.9], [0.3, 0.9], [0.5, 0.9], [0.7, 0.9], [0.9, 0.9]
        ])
        sides = np.full(10, 0.2)
        return centers, sides

    return best_centers, best_sides

# EVOLVE-BLOCK-END
