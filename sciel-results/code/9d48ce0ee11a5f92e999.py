"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 21

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize


def construct():
    """
    Pack 21 circles in a rectangle of perimeter at most 4.
    Use force-directed layout with contact constraints.
    """
    # Rectangle with perimeter 4: let width = w, height = h, 2(w+h) = 4 => w+h = 2
    # We'll optimize over w and h as well, or fix aspect ratio and scale.
    # Actually, the problem says "variable rectangle of perimeter 4", so we can choose w,h with w+h=2.
    # The bounding rectangle of the circles must have perimeter at most 4.
    # So if we place circles in [0, w] x [0, h] with w+h=2, the bounding box is exactly w x h, perimeter 4.
    
    # Strategy: 
    # 1. Choose an aspect ratio (w, h) with w+h=2.
    # 2. Place circles inside [0,w] x [0,h].
    # 3. Maximize sum of radii.
    
    # Force-directed approach:
    # - Variables: x_i, y_i, r_i for each circle i, and w (h = 2-w)
    # - Objective: maximize sum(r_i)
    # - Constraints:
    #   - r_i >= 0
    #   - x_i - r_i >= 0, x_i + r_i <= w
    #   - y_i - r_i >= 0, y_i + r_i <= h
    #   - (x_i - x_j)^2 + (y_i - y_j)^2 >= (r_i + r_j)^2 for all i != j
    
    # We'll use a penalty-based approach with scipy.optimize.minimize,
    # but with a smart initialization and force-directed relaxation.
    
    # Initial guess: grid layout
    side = int(np.ceil(np.sqrt(N)))
    # Use a 5x5 grid (25 cells), take first 21
    # Let's try different aspect ratios
    
    best_val = -1
    best_result = None
    
    # Try several aspect ratios
    for w in np.linspace(0.5, 1.5, 11):
        h = 2.0 - w
        
        # Initialize with grid
        cols = side
        rows = int(np.ceil(N / cols))
        x0 = []
        y0 = []
        r0 = []
        count = 0
        for i in range(cols):
            for j in range(rows):
                if count >= N:
                    break
                x = (i + 0.5) * w / cols
                y = (j + 0.5) * h / rows
                r = min(w / cols, h / rows) / 2 * 0.9  # slightly smaller to avoid overlap
                x0.append(x)
                y0.append(y)
                r0.append(r)
                count += 1
        
        x0 = np.array(x0)
        y0 = np.array(y0)
        r0 = np.array(r0)
        
        # Variables: [x0, y0, ..., xN-1, yN-1, r0, ..., rN-1]
        def objective(v):
            x = v[:N]
            y = v[N:2*N]
            r = v[2*N:3*N]
            
            # Penalty for violations
            pen = 0.0
            
            # Boundary constraints
            for i in range(N):
                if r[i] < 0:
                    pen += 100 * (-r[i]) ** 2
                if x[i] - r[i] < 0:
                    pen += 100 * (-(x[i] - r[i])) ** 2
                if x[i] + r[i] > w:
                    pen += 100 * (x[i] + r[i] - w) ** 2
                if y[i] - r[i] < 0:
                    pen += 100 * (-(y[i] - r[i])) ** 2
                if y[i] + r[i] > h:
                    pen += 100 * (y[i] + r[i] - h) ** 2
            
            # Non-overlap constraints
            for i in range(N):
                for j in range(i+1, N):
                    dist_sq = (x[i] - x[j]) ** 2 + (y[i] - y[j]) ** 2
                    min_dist = r[i] + r[j]
                    if dist_sq < min_dist ** 2:
                        pen += 100 * (min_dist - np.sqrt(dist_sq)) ** 2
            
            # Maximize sum of radii, so minimize -sum(r)
            return -np.sum(r) + pen
        
        v0 = np.concatenate([x0, y0, r0])
        
        # Use L-BFGS-B with bounds
        bounds = []
        for i in range(N):
            bounds.append((0, w))  # x
        for i in range(N):
            bounds.append((0, h))  # y
        for i in range(N):
            bounds.append((0, min(w, h)/2))  # r
        
        res = minimize(objective, v0, method='L-BFGS-B', bounds=bounds, 
                       options={'maxiter': 500, 'ftol': 1e-10, 'gtol': 1e-8})
        
        x = res.x[:N]
        y = res.x[N:2*N]
        r = res.x[2*N:3*N]
        
        # Verify constraints
        valid = True
        for i in range(N):
            if r[i] < -1e-6:
                valid = False
                break
            if x[i] - r[i] < -1e-6 or x[i] + r[i] > w + 1e-6:
                valid = False
                break
            if y[i] - r[i] < -1e-6 or y[i] + r[i] > h + 1e-6:
                valid = False
                break
        
        if valid:
            for i in range(N):
                for j in range(i+1, N):
                    dist_sq = (x[i] - x[j]) ** 2 + (y[i] - y[j]) ** 2
                    if dist_sq < (r[i] + r[j]) ** 2 - 1e-8:
                        valid = False
                        break
                if not valid:
                    break
        
        if valid:
            val = np.sum(r)
            if val > best_val:
                best_val = val
                best_result = np.column_stack([x, y, r])
    
    if best_result is not None:
        return best_result
    
    # Fallback: simple grid
    side = int(np.ceil(np.sqrt(N)))
    cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3) for i in range(side) for j in range(side)]
    return np.array(cells[:N])

# EVOLVE-BLOCK-END
