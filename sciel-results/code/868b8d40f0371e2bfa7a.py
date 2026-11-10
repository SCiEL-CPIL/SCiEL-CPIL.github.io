import numpy as np
from scipy.optimize import minimize

def squares_in_square10() -> tuple:
    n = 10
    tol = 1e-7

    # Analytic seed 1: 4 corners (1/3), 4 edges (1/3), 2 center (1/6)
    # Sum = 3.0
    a = 1.0 / 3.0
    b = 1.0 / 3.0
    c = 1.0 / 6.0

    centers0 = np.zeros((n, 2))
    sides0 = np.zeros(n)

    # Corners (0-3)
    centers0[0] = [a/2, a/2]
    centers0[1] = [1-a/2, a/2]
    centers0[2] = [a/2, 1-a/2]
    centers0[3] = [1-a/2, 1-a/2]
    sides0[0:4] = a

    # Edges (4-7)
    centers0[4] = [0.5, b/2]      # Bottom
    centers0[5] = [0.5, 1-b/2]    # Top
    centers0[6] = [b/2, 0.5]      # Left
    centers0[7] = [1-b/2, 0.5]    # Right
    sides0[4:8] = b

    # Center (8-9)
    centers0[8] = [0.5, 5.0/12.0]
    centers0[9] = [0.5, 7.0/12.0]
    sides0[8:10] = c

    x0_1 = np.concatenate([centers0.flatten(), sides0])

    # Analytic seed 2: 3x3 grid of 1/3 squares, 10th is tiny in center?
    # Actually, 3x3 fills it. Let's try 4 corners 0.4, 4 edges 0.2, 2 center 0.2?
    # Let's just use random seeds as well.
    
    seeds = [x0_1]
    rng = np.random.default_rng(42)
    for _ in range(5):
        c_rand = rng.uniform(0.1, 0.9, size=(n, 2))
        s_rand = rng.uniform(0.1, 0.3, size=n)
        x_rand = np.concatenate([c_rand.flatten(), s_rand])
        seeds.append(x_rand)

    def objective(x):
        c = x[:2*n].reshape(n, 2)
        s = x[2*n:]
        
        # Penalty
        pen = 0.0
        for i in range(n):
            # Bounds
            # We want s > 0, c - s/2 >= 0, c + s/2 <= 1
            # Penalty for s <= 0
            if s[i] <= 0:
                pen += 10.0 * (s[i])**2
            # Penalty for out of bounds
            dx_min = c[i,0] - s[i]/2
            dx_max = c[i,0] + s[i]/2
            dy_min = c[i,1] - s[i]/2
            dy_max = c[i,1] + s[i]/2
            
            if dx_min < 0: pen += (dx_min)**2
            if dx_max > 1: pen += (dx_max - 1)**2
            if dy_min < 0: pen += (dy_min)**2
            if dy_max > 1: pen += (dy_max - 1)**2

        for i in range(n):
            for j in range(i+1, n):
                dx = abs(c[i,0] - c[j,0])
                dy = abs(c[i,1] - c[j,1])
                t = (s[i] + s[j])/2
                # Overlap if dx < t and dy < t
                ox = max(0.0, t - dx)
                oy = max(0.0, t - dy)
                pen += ox * oy

        # Maximize sum(s), so minimize -sum(s)
        return -np.sum(s) + 100.0 * pen

    def is_valid(c, s):
        for i in range(n):
            if not (np.isfinite(c[i,0]) and np.isfinite(c[i,1]) and np.isfinite(s[i])):
                return False
            if s[i] <= 0:
                return False
            if c[i,0] - s[i]/2 < -tol or c[i,0] + s[i]/2 > 1 + tol:
                return False
            if c[i,1] - s[i]/2 < -tol or c[i,1] + s[i]/2 > 1 + tol:
                return False
        for i in range(n):
            for j in range(i+1, n):
                t = (s[i] + s[j])/2 - tol
                if abs(c[i,0] - c[j,0]) < t and abs(c[i,1] - c[j,1]) < t:
                    return False
        return True

    best_x = x0_1
    best_val = np.sum(sides0)
    best_obj = objective(x0_1)

    # Nelder-Mead options
    options = {
        'maxiter': 500,
        'maxfev': 5000,
        'xatol': 1e-8,
        'fatol': 1e-8,
        'adaptive': True
    }

    for x_start in seeds:
        try:
            res = minimize(objective, x_start, method='Nelder-Mead', options=options)
            if res.fun < best_obj:
                best_obj = res.fun
                best_x = res.x
        except Exception:
            pass

    c_opt = best_x[:2*n].reshape(n, 2)
    s_opt = best_x[2*n:]

    # Check validity of the best found
    if is_valid(c_opt, s_opt):
        return c_opt, s_opt
    
    # If the best found is invalid (penalty might have been non-zero but low), 
    # try to repair or fall back to seed
    # The seed is valid, so we can return it if optimization failed to improve validly
    if is_valid(centers0, sides0):
        return centers0, sides0
    
    # Last resort: shrink everything
    scale = 0.9
    c_shrunk = c_opt * scale
    s_shrunk = s_opt * scale
    # This might not be valid, but let's just return the seed which is known valid
    return centers0, sides0
