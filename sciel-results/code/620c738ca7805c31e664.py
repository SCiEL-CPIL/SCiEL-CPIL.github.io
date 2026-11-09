import numpy as np
from scipy.optimize import minimize
import time

def squares_in_square10() -> tuple:
    n = 10
    start_time = time.time()
    deadline = start_time + 590  # 10s margin
    
    best_score = -np.inf
    best_centers = None
    best_sides = None
    
    # Penalty function
    def calculate_penalty_and_objective(params, lambda_overlap=100.0, lambda_bound=100.0):
        # params: [x0, y0, s0, x1, y1, s1, ...]
        centers = params[0:2*n].reshape(n, 2)
        sides = params[2*n:3*n]
        
        # Objective: -sum(sides)
        obj = -np.sum(sides)
        
        # Boundary Penalty
        bound_pen = 0.0
        for i in range(n):
            x, y, s = centers[i, 0], centers[i, 1], sides[i]
            # x bounds: s/2 <= x <= 1 - s/2
            pen_x = 0.0
            if x < s/2:
                pen_x += (s/2 - x)**2
            if x > 1 - s/2:
                pen_x += (x - (1 - s/2))**2
            
            pen_y = 0.0
            if y < s/2:
                pen_y += (s/2 - y)**2
            if y > 1 - s/2:
                pen_y += (y - (1 - s/2))**2
            
            bound_pen += pen_x + pen_y
        
        # Overlap Penalty
        overlap_pen = 0.0
        for i in range(n):
            for j in range(i+1, n):
                dx = abs(centers[i, 0] - centers[j, 0])
                dy = abs(centers[i, 1] - centers[j, 1])
                sx = sides[i] + sides[j]
                sy = sides[i] + sides[j]
                
                pen_x = 0.0
                if dx < sx/2:
                    pen_x = (sx/2 - dx)**2
                pen_y = 0.0
                if dy < sy/2:
                    pen_y = (sy/2 - dy)**2
                
                # Product ensures it's 0 if separated in either dim
                overlap_pen += pen_x * pen_y
        
        return obj + lambda_overlap * overlap_pen + lambda_bound * bound_pen

    def get_centers_sides(params):
        centers = params[0:2*n].reshape(n, 2)
        sides = params[2*n:3*n]
        return centers, sides

    def is_valid(centers, sides):
        tol = 1e-7
        # Check bounds
        for i in range(n):
            x, y, s = centers[i, 0], centers[i, 1], sides[i]
            if not (0 - tol <= x - s/2 and x + s/2 <= 1 + tol):
                return False
            if not (0 - tol <= y - s/2 and y + s/2 <= 1 + tol):
                return False
            if s <= 0:
                return False
        
        # Check overlaps
        for i in range(n):
            for j in range(i+1, n):
                dx = abs(centers[i, 0] - centers[j, 0])
                dy = abs(centers[i, 1] - centers[j, 1])
                t = (sides[i] + sides[j])/2 - tol
                if dx < t and dy < t:
                    return False
        return True

    def update_best(params):
        nonlocal best_score, best_centers, best_sides
        centers, sides = get_centers_sides(params)
        if is_valid(centers, sides):
            score = np.sum(sides)
            if score > best_score:
                best_score = score
                best_centers = centers.copy()
                best_sides = sides.copy()

    # Seeds
    seeds = []
    
    # Seed 1: Corner configuration (1 large, 9 small)
    c1 = np.array([
        [0.25, 0.25], 
        [0.125, 0.625], [0.375, 0.625],
        [0.125, 0.875], [0.375, 0.875],
        [0.625, 0.125], [0.875, 0.125],
        [0.625, 0.375], [0.875, 0.375],
        [0.625, 0.625]
    ])
    s1 = np.array([0.5] + [0.25]*9)
    seeds.append(np.concatenate([c1.flatten(), s1]))
    
    # Seed 2: 2x5 Grid (Side 0.2)
    c2 = np.zeros((10, 2))
    s2 = np.full(10, 0.2)
    for i in range(10):
        col = i % 5
        row = i // 5
        c2[i, 0] = (col + 0.5) * 0.2
        c2[i, 1] = (row + 0.5) * 0.5 # Height is 1.0, 2 rows of 0.5? No, 2 rows means height 0.5 each?
        # Wait, 2 rows of squares of side 0.2.
        # Total height 0.4. Centered? Or bottom aligned?
        # Let's just place them in a 2x5 grid of cells 0.2x0.2?
        # No, 2 rows means 2 squares high. 5 columns means 5 squares wide.
        # Total width 1.0 (5*0.2). Total height 0.4 (2*0.2).
        # This leaves space.
        c2[i, 0] = (col + 0.5) * 0.2
        c2[i, 1] = (row + 0.5) * 0.2
    seeds.append(np.concatenate([c2.flatten(), s2]))
    
    # Seed 3: Random
    np.random.seed(42)
    for _ in range(5):
        c_rand = np.random.rand(10, 2) * 0.8 + 0.1
        s_rand = np.random.rand(10) * 0.2 + 0.1
        # Simple rejection sampling for valid start?
        # Just use it, optimizer will fix.
        seeds.append(np.concatenate([c_rand.flatten(), s_rand]))

    # Optimization
    for seed in seeds:
        if time.time() > deadline:
            break
            
        try:
            res = minimize(
                calculate_penalty_and_objective,
                seed,
                method='SLSQP',
                bounds=[(1e-4, 1.0)] * (3 * n), # x, y, s bounds
                options={'maxiter': 200, 'ftol': 1e-8}
            )
            if res.success:
                update_best(res.x)
        except Exception:
            pass

    # If no valid solution found (should not happen), fallback to seed 1
    if best_centers is None:
        best_centers = c1
        best_sides = s1

    return best_centers, best_sides
