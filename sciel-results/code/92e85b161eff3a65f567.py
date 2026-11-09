import numpy as np
import time
from scipy.optimize import minimize, linprog
from itertools import combinations

def squares_in_square10() -> tuple:
    """
    Maximizes the sum of side lengths of 10 axis-aligned squares 
    packed inside the unit square [0, 1]^2.
    """
    n = 10
    start_time = time.time()
    deadline = 600  # Safety margin before 630s kill
    
    # Initial heuristic layout
    # This is a valid packing with sum = 3.0
    # 2 squares of 0.5, 8 squares of 0.25
    centers_init = np.array([
        [0.25, 0.25],
        [0.75, 0.25],
        [0.125, 0.625],
        [0.375, 0.625],
        [0.625, 0.625],
        [0.875, 0.625],
        [0.125, 0.875],
        [0.375, 0.875],
        [0.625, 0.875],
        [0.875, 0.875]
    ])
    sides_init = np.array([
        0.5, 0.5,
        0.25, 0.25, 0.25, 0.25,
        0.25, 0.25, 0.25, 0.25
    ])
    
    best_centers = centers_init.copy()
    best_sides = sides_init.copy()
    best_value = np.sum(best_sides)
    
    # Helper to validate and update best
    def update_best(centers, sides):
        nonlocal best_centers, best_sides, best_value
        tol = 1e-7
        # Check boundaries
        for i in range(n):
            if centers[i, 0] - sides[i]/2 < -tol or centers[i, 0] + sides[i]/2 > 1 + tol:
                return
            if centers[i, 1] - sides[i]/2 < -tol or centers[i, 1] + sides[i]/2 > 1 + tol:
                return
        # Check overlaps
        for i in range(n):
            for j in range(i+1, n):
                dx = abs(centers[i, 0] - centers[j, 0])
                dy = abs(centers[i, 1] - centers[j, 1])
                t = (sides[i] + sides[j])/2 - tol
                if dx < t and dy < t:
                    return
        
        val = np.sum(sides)
        if val > best_value:
            best_value = val
            best_centers = centers.copy()
            best_sides = sides.copy()

    update_best(centers_init, sides_init)
    
    # Objective function for continuous optimization
    # We use a smooth penalty for overlaps to allow gradient-based methods
    def objective(x):
        cx = x[0::3]
        cy = x[1::3]
        s = x[2::3]
        
        # Ensure positive sides
        s = np.maximum(s, 1e-6)
        
        # Boundary penalties
        penalty = 0.0
        for i in range(n):
            # x-boundary
            if cx[i] - s[i]/2 < 0:
                penalty += 100 * (cx[i] - s[i]/2)**2
            if cx[i] + s[i]/2 > 1:
                penalty += 100 * (cx[i] + s[i]/2 - 1)**2
            # y-boundary
            if cy[i] - s[i]/2 < 0:
                penalty += 100 * (cy[i] - s[i]/2)**2
            if cy[i] + s[i]/2 > 1:
                penalty += 100 * (cy[i] + s[i]/2 - 1)**2
        
        # Overlap penalties (smooth)
        # For each pair, if they overlap, add penalty
        # Overlap in x: max(0, (s_i+s_j)/2 - |cx_i - cx_j|)
        # Overlap in y: max(0, (s_i+s_j)/2 - |cy_i - cy_j|)
        # Penalty = product of overlaps (approx area of overlap)
        for i in range(n):
            for j in range(i+1, n):
                dx = abs(cx[i] - cx[j])
                dy = abs(cy[i] - cy[j])
                dist_needed = (s[i] + s[j]) / 2
                overlap_x = max(0, dist_needed - dx)
                overlap_y = max(0, dist_needed - dy)
                penalty += 50 * overlap_x * overlap_y
        
        return -(np.sum(s)) + penalty

    # Initial state for optimization
    x0 = np.zeros(3 * n)
    for i in range(n):
        x0[3*i] = centers_init[i, 0]
        x0[3*i+1] = centers_init[i, 1]
        x0[3*i+2] = sides_init[i]
        
    # Run Nelder-Mead optimization
    try:
        res = minimize(objective, x0, method='Nelder-Mead', 
                       options={'maxiter': 5000, 'xatol': 1e-8, 'fatol': 1e-8, 'adaptive': True})
        
        cx = res.x[0::3]
        cy = res.x[1::3]
        s = res.x[2::3]
        
        # Clamp to valid range to ensure boundary constraints are met
        for i in range(n):
            s[i] = max(s[i], 1e-6)
            cx[i] = np.clip(cx[i], s[i]/2, 1 - s[i]/2)
            cy[i] = np.clip(cy[i], s[i]/2, 1 - s[i]/2)
            
        centers = np.stack([cx, cy], axis=1)
        sides = s
        update_best(centers, sides)
        
    except Exception as e:
        pass

    # Try a few random restarts if time permits
    np.random.seed(42)
    while time.time() - start_time < deadline - 30:
        # Generate random initial guess
        # Try to create a non-overlapping random layout
        centers_rand = np.zeros((n, 2))
        sides_rand = np.zeros(n)
        
        # Place squares one by one, ensuring no overlap with existing ones
        # This is a greedy approach for initialization
        for i in range(n):
            placed = False
            attempts = 0
            while not placed and attempts < 100:
                s = np.random.uniform(0.1, 0.4)
                cx = np.random.uniform(s/2, 1 - s/2)
                cy = np.random.uniform(s/2, 1 - s/2)
                
                # Check overlap with previously placed squares
                overlap = False
                for j in range(i):
                    dx = abs(cx - centers_rand[j, 0])
                    dy = abs(cy - centers_rand[j, 1])
                    dist_needed = (s + sides_rand[j]) / 2
                    if dx < dist_needed and dy < dist_needed:
                        overlap = True
                        break
                
                if not overlap:
                    centers_rand[i] = [cx, cy]
                    sides_rand[i] = s
                    placed = True
                attempts += 1
            
            if not placed:
                # Fallback to a small square in a corner
                sides_rand[i] = 0.05
                centers_rand[i] = [0.025, 0.025]
        
        x_rand = np.zeros(3 * n)
        for i in range(n):
            x_rand[3*i] = centers_rand[i, 0]
            x_rand[3*i+1] = centers_rand[i, 1]
            x_rand[3*i+2] = sides_rand[i]
            
        try:
            res = minimize(objective, x_rand, method='Nelder-Mead', 
                           options={'maxiter': 2000, 'xatol': 1e-8, 'fatol': 1e-8, 'adaptive': True})
            
            cx = res.x[0::3]
            cy = res.x[1::3]
            s = res.x[2::3]
            
            for i in range(n):
                s[i] = max(s[i], 1e-6)
                cx[i] = np.clip(cx[i], s[i]/2, 1 - s[i]/2)
                cy[i] = np.clip(cy[i], s[i]/2, 1 - s[i]/2)
                
            centers = np.stack([cx, cy], axis=1)
            sides = s
            update_best(centers, sides)
            
        except Exception as e:
            pass

    # Final polish: Try to increase sides slightly if there is slack
    # This is a simple local search
    if time.time() - start_time < deadline - 10:
        for _ in range(100):
            improved = False
            for i in range(n):
                # Try increasing side i
                old_s = best_sides[i]
                new_s = old_s * 1.01
                if new_s > 1: continue
                
                # Check if we can fit the larger square
                # We need to ensure it doesn't overlap with others
                # This requires solving for position, which is complex.
                # So we skip this simple polish and rely on the optimizer.
                pass
                
            # Break if no improvement or time is running out
            if time.time() - start_time > deadline - 5:
                break

    return best_centers, best_sides
