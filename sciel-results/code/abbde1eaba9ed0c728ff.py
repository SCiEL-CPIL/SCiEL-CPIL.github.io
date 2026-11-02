import numpy as np
import time
from scipy.optimize import minimize

N = 21

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin for 120s limit
    
    best_val = -np.inf
    best_solution = None
    
    # Aspect ratios to try (w, h=2-w)
    # w in [1, 2]
    aspects = np.linspace(1.0, 2.0, 11)
    
    # Grid configurations (nx, ny)
    # We need at least N circles. We will place N circles in a nx x ny grid.
    # If nx*ny > N, we leave some spots empty (or just pick the first N).
    # It's better to distribute them.
    grids = [
        (5, 5),
        (4, 6),
        (6, 4),
        (3, 7),
        (7, 3),
        (2, 11),
        (11, 2),
        (1, 21),
        (21, 1)
    ]
    
    # Penalty weight
    lam = 100.0
    
    def objective(params, w, h):
        # params: [x0, y0, r0, x1, y1, r1, ...]
        x = params[0::3]
        y = params[1::3]
        r = params[2::3]
        
        # Sum of radii (to maximize, so minimize negative)
        sum_r = np.sum(r)
        
        # Boundary penalties
        pen_bound = 0.0
        for i in range(N):
            # Left
            v = r[i] - x[i]
            if v > 0:
                pen_bound += v * v
            # Right
            v = r[i] - (w - x[i])
            if v > 0:
                pen_bound += v * v
            # Bottom
            v = r[i] - y[i]
            if v > 0:
                pen_bound += v * v
            # Top
            v = r[i] - (h - y[i])
            if v > 0:
                pen_bound += v * v
        
        # Overlap penalties
        pen_overlap = 0.0
        for i in range(N):
            for j in range(i + 1, N):
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                dist_sq = dx * dx + dy * dy
                min_dist = r[i] + r[j]
                # If dist_sq < min_dist^2, they overlap
                if dist_sq < min_dist * min_dist:
                    dist = np.sqrt(dist_sq)
                    v = min_dist - dist
                    pen_overlap += v * v
        
        return -sum_r + lam * (pen_bound + pen_overlap)
    
    # We will iterate through aspects and grids
    for w in aspects:
        if time.time() > deadline:
            break
        h = 2.0 - w
        
        for nx, ny in grids:
            if time.time() > deadline:
                break
            
            # Create initial positions
            # Place circles in a grid
            # If nx*ny >= N, we can map indices to grid cells
            # Simple mapping: row-major order
            # But we want to fill the rectangle nicely.
            # Let's just place N circles in the first N cells of an nx x ny grid
            # This might be biased. Better to distribute evenly.
            
            # Calculate grid spacing
            # Assume circles are roughly equal size
            # For now, just place them at grid centers
            xs = np.zeros(N)
            ys = np.zeros(N)
            rs = np.zeros(N)
            
            # Fill grid
            count = 0
            for i in range(nx):
                for j in range(ny):
                    if count >= N:
                        break
                    # Center of cell
                    cx = (i + 0.5) * (w / nx)
                    cy = (j + 0.5) * (h / ny)
                    xs[count] = cx
                    ys[count] = cy
                    # Initial radius: half the min spacing
                    # Spacing is min(w/nx, h/ny)
                    spacing = min(w / nx, h / ny)
                    rs[count] = 0.4 * spacing # Start a bit smaller
                    count += 1
                if count >= N:
                    break
            
            # Ensure at least N circles
            if count < N:
                # This shouldn't happen with chosen grids
                continue
                
            params_init = np.concatenate([xs, ys, rs])
            
            # Bounds for L-BFGS-B
            # x in [0, w], y in [0, h], r in [0, 1]
            bounds = []
            for i in range(N):
                bounds.append((0, w))
                bounds.append((0, h))
                bounds.append((0, 1.0))
            
            # Run optimization
            try:
                res = minimize(objective, params_init, args=(w, h), method='L-BFGS-B', bounds=bounds, options={'maxiter': 200, 'ftol': 1e-8})
                
                # Extract solution
                x_sol = res.x[0::3]
                y_sol = res.x[1::3]
                r_sol = res.x[2::3]
                
                # Check validity and calculate score
                # We need to ensure constraints are satisfied for the score to be valid
                # The penalty method might leave small violations.
                # We should check if violations are negligible.
                valid = True
                min_dist_viol = 0
                for i in range(N):
                    if x_sol[i] < r_sol[i] - 1e-6 or x_sol[i] > w - r_sol[i] + 1e-6:
                        valid = False
                    if y_sol[i] < r_sol[i] - 1e-6 or y_sol[i] > h - r_sol[i] + 1e-6:
                        valid = False
                    for j in range(i + 1, N):
                        dist = np.sqrt((x_sol[i] - x_sol[j])**2 + (y_sol[i] - y_sol[j])**2)
                        if dist < r_sol[i] + r_sol[j] - 1e-6:
                            valid = False
                            min_dist_viol = min(min_dist_viol, dist - (r_sol[i] + r_sol[j]))
                
                if valid:
                    score = np.sum(r_sol)
                    if score > best_val:
                        best_val = score
                        best_solution = np.column_stack((x_sol, y_sol, r_sol))
                        
            except Exception as e:
                pass
                
    # If no valid solution found, return a simple valid one
    if best_solution is None:
        # Fallback: small circles in a grid
        w = 1.0
        h = 1.0
        side = 5
        cells = []
        for i in range(side):
            for j in range(side):
                if len(cells) >= N:
                    break
                x = (i + 0.5) / side
                y = (j + 0.5) / side
                r = 0.4 / side
                cells.append([x, y, r])
        best_solution = np.array(cells[:N])
        
    return best_solution

# EVOLVE-BLOCK-END
