import numpy as np
from scipy.optimize import minimize

def squares_in_square10() -> tuple:
    n = 10
    
    # Initial guess: 3x3 grid + 1 small
    # We'll use a random initialization and local search
    # Variables: 10 centers (20) + 10 sides (10) = 30 variables
    
    def objective(x):
        # x: [cx1, cy1, s1, cx2, cy2, s2, ...]
        centers = x[:2*n].reshape(n, 2)
        sides = x[2*n:3*n]
        
        # Penalty for being out of bounds
        penalty = 0
        for i in range(n):
            cx, cy = centers[i]
            s = sides[i]
            if s <= 0:
                penalty += 1e6 * (1 - s)
            if cx - s/2 < -1e-7:
                penalty += 1e6 * abs(cx - s/2)
            if cx + s/2 > 1 + 1e-7:
                penalty += 1e6 * abs(cx + s/2 - 1)
            if cy - s/2 < -1e-7:
                penalty += 1e6 * abs(cy - s/2)
            if cy + s/2 > 1 + 1e-7:
                penalty += 1e6 * abs(cy + s/2 - 1)
        
        # Penalty for overlap
        for i in range(n):
            for j in range(i+1, n):
                dx = abs(centers[i,0] - centers[j,0])
                dy = abs(centers[i,1] - centers[j,1])
                t = (sides[i] + sides[j])/2 - 1e-7
                if dx < t and dy < t:
                    # Overlap penalty: proportional to overlap depth
                    overlap_x = max(0, t - dx)
                    overlap_y = max(0, t - dy)
                    penalty += 1e6 * (overlap_x * overlap_y)
        
        # Maximize sum of sides -> Minimize negative sum
        return penalty - np.sum(sides)

    # Initial guess: 3x4 grid
    x0 = np.zeros(3*n)
    for i in range(n):
        col = i % 4
        row = i // 4
        cx = (col + 0.5) * 0.25
        cy = (row + 0.5) * 0.25
        x0[3*i] = cx
        x0[3*i+1] = cy
        x0[3*i+2] = 0.25
        
    # Add some randomness to escape local minima
    x0 += np.random.normal(0, 0.01, 3*n)
    
    # Bounds: centers in [0,1], sides in [0,1]
    bounds = []
    for i in range(n):
        bounds.append((0, 1)) # cx
        bounds.append((0, 1)) # cy
        bounds.append((0, 1)) # s
        
    # Run optimization
    res = minimize(objective, x0, method='Nelder-Mead', bounds=bounds, 
                   options={'maxiter': 20000, 'xatol': 1e-5, 'fatol': 1e-5})
    
    x_opt = res.x
    centers = x_opt[:2*n].reshape(n, 2)
    sides = x_opt[2*n:3*n]
    
    # Ensure sides are positive
    sides = np.maximum(sides, 1e-6)
    
    return centers, sides
