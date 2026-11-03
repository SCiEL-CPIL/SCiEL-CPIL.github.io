import numpy as np
from scipy.optimize import minimize
import time

N = 21

def construct():
    start_time = time.time()
    deadline = start_time + 110  # 10s safety margin
    
    best_score = -np.inf
    best_circles = None
    
    # Initial guesses
    # 1. Grid 5x5
    # 2. Grid 4x6 (24, take 21)
    # 3. Random
    
    def get_grid_guess(cols, rows):
        # Returns list of (x, y, r)
        # Normalize to 1x1
        guess = []
        for i in range(cols):
            for j in range(rows):
                if len(guess) < N:
                    x = (i + 0.5) / cols
                    y = (j + 0.5) / rows
                    r = 0.5 / max(cols, rows) * 0.9 # slightly smaller to avoid initial overlap
                    guess.append((x, y, r))
        return guess

    guesses = [
        get_grid_guess(5, 5),
        get_grid_guess(4, 6),
        get_grid_guess(6, 4),
        get_grid_guess(3, 7),
        get_grid_guess(7, 3)
    ]
    
    # Add some random guesses
    np.random.seed(42)
    for _ in range(5):
        x = np.random.rand(N)
        y = np.random.rand(N)
        r = np.full(N, 0.1)
        guesses.append(list(zip(x, y, r)))

    def objective(vars):
        # vars: [x0, y0, r0, ..., x20, y20, r20, w]
        w = vars[63]
        if w <= 0.1 or w >= 1.9: # h = 2-w, so w in [0.1, 1.9]
            return 1e10
        h = 2.0 - w
        
        # Extract arrays
        # Reshape for vectorization
        # vars[0:63] -> (21, 3)
        pts_r = vars[:63].reshape(N, 3)
        x = pts_r[:, 0]
        y = pts_r[:, 1]
        r = np.maximum(pts_r[:, 2], 0) # radii must be positive
        
        sum_r = np.sum(r)
        
        # Boundary penalties
        # x in [r, w-r]
        pen_x_low = np.maximum(r - x, 0)**2
        pen_x_high = np.maximum(x - (w - r), 0)**2
        pen_y_low = np.maximum(r - y, 0)**2
        pen_y_high = np.maximum(y - (h - r), 0)**2
        
        boundary_pen = np.sum(pen_x_low) + np.sum(pen_x_high) + np.sum(pen_y_low) + np.sum(pen_y_high)
        
        # Overlap penalties
        # Distance matrix
        # D[i,j] = (xi-xj)^2 + (yi-yj)^2
        # We need sqrt(D)
        # For N=21, 21x21 matrix is small.
        diff_x = x[:, np.newaxis] - x[np.newaxis, :]
        diff_y = y[:, np.newaxis] - y[np.newaxis, :]
        dist_sq = diff_x**2 + diff_y**2
        dist = np.sqrt(dist_sq)
        
        sum_r_ij = r[:, np.newaxis] + r[np.newaxis, :]
        
        overlap = np.maximum(sum_r_ij - dist, 0)
        # Exclude diagonal
        np.fill_diagonal(overlap, 0)
        
        overlap_pen = np.sum(overlap**2)
        
        penalty = 10.0 * (boundary_pen + overlap_pen)
        
        return -sum_r + penalty

    for guess in guesses:
        if time.time() > deadline:
            break
            
        init_vars = []
        for px, py, pr in guess:
            init_vars.extend([px, py, pr])
        init_vars.append(1.0) # initial w
        
        try:
            res = minimize(objective, init_vars, method='Nelder-Mead', 
                           options={'maxiter': 1000, 'xatol': 1e-5, 'fatol': 1e-5})
            
            # Check validity and score
            vars = res.x
            w = vars[63]
            h = 2.0 - w
            
            if w <= 0 or h <= 0:
                continue
                
            pts_r = vars[:63].reshape(N, 3)
            x = pts_r[:, 0]
            y = pts_r[:, 1]
            r = np.maximum(pts_r[:, 2], 0)
            
            # Check if constraints are satisfied (penalty should be ~0)
            # Re-calculate penalty to be sure
            pen = objective(vars)
            sum_r = np.sum(r)
            
            # If penalty is low enough, it's a valid packing
            # The objective returns -sum_r + penalty.
            # So score = -objective if penalty is small.
            # But we want to maximize sum_r.
            # If penalty > 0, the packing is invalid.
            # We should only accept if penalty is very small.
            
            # Let's re-evaluate penalty separately for clarity
            # (Using the same logic as objective)
            
            # If we trust the minimizer, res.fun is -sum_r + penalty.
            # We want to maximize sum_r, so minimize -sum_r.
            # But we must ensure penalty is ~0.
            
            # Let's check the actual penalty value
            # We can modify objective to return penalty separately? No, minimize takes scalar.
            # We can just check if res.fun is close to -sum_r.
            
            # Actually, if penalty is non-zero, the "score" (sum_r) is effectively reduced by the penalty term in the objective, but the returned circles might still overlap.
            # We need to verify the final circles are disjoint.
            
            # Verification:
            valid = True
            for i in range(N):
                if x[i] < r[i] or x[i] > w - r[i] or y[i] < r[i] or y[i] > h - r[i]:
                    valid = False
                    break
            if valid:
                for i in range(N):
                    for j in range(i+1, N):
                        if np.sqrt((x[i]-x[j])**2 + (y[i]-y[j])**2) < r[i] + r[j] - 1e-5:
                            valid = False
                            break
                    if not valid: break
            
            if valid:
                score = np.sum(r)
                if score > best_score:
                    best_score = score
                    best_circles = np.column_stack((x, y, r))
                    
        except Exception as e:
            pass

    if best_circles is None:
        # Fallback to grid
        side = 5
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3) for i in range(side) for j in range(side)]
        return np.array(cells[:N])
        
    return best_circles
