"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 26

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin before 150s kill
    
    best_sum = 0.0
    best_result = None
    
    def objective(x):
        # x = [x1, y1, r1, x2, y2, r2, ...]
        # Maximize sum of radii -> minimize negative sum
        radii = x[2::3]
        return -np.sum(radii)
    
    def make_constraints():
        cons = []
        # Boundary constraints: circle must be inside [0,1]^2
        # x_i >= r_i, x_i <= 1 - r_i, y_i >= r_i, y_i <= 1 - r_i
        for i in range(N):
            xi = 3*i
            yi = 3*i+1
            ri = 3*i+2
            cons.append({'type': 'ineq', 'fun': lambda x, i=i: x[xi] - x[ri]})
            cons.append({'type': 'ineq', 'fun': lambda x, i=i: (1 - x[xi]) - x[ri]})
            cons.append({'type': 'ineq', 'fun': lambda x, i=i: x[yi] - x[ri]})
            cons.append({'type': 'ineq', 'fun': lambda x, i=i: (1 - x[yi]) - x[ri]})
            # Radius positive
            cons.append({'type': 'ineq', 'fun': lambda x, i=i: x[ri]})
        
        # Pairwise non-overlap: (x_i - x_j)^2 + (y_i - y_j)^2 >= (r_i + r_j)^2
        for i in range(N):
            for j in range(i+1, N):
                xi, yi, ri = 3*i, 3*i+1, 3*i+2
                xj, yj, rj = 3*j, 3*j+1, 3*j+2
                def non_overlap(x, i=i, j=j):
                    dx = x[xi] - x[xj]
                    dy = x[yi] - x[yj]
                    return dx*dx + dy*dy - (x[ri] + x[rj])**2
                cons.append({'type': 'ineq', 'fun': non_overlap})
        
        return cons
    
    def make_bounds():
        bounds = []
        for i in range(N):
            bounds.append((1e-6, 0.5))  # x in [0,1], but effectively constrained by radius
            bounds.append((1e-6, 0.5))  # y
            bounds.append((1e-6, 0.5))  # radius
        return bounds
    
    constraints = make_constraints()
    bounds = make_bounds()
    
    # Generate multiple initial guesses
    def grid_init(perturb=0.0):
        side = int(np.ceil(np.sqrt(N)))
        cells = []
        for i in range(side):
            for j in range(side):
                if len(cells) >= N:
                    break
                cx = (i + 0.5) / side
                cy = (j + 0.5) / side
                r = 0.5 / side * 0.95
                if perturb > 0:
                    cx += np.random.uniform(-perturb, perturb)
                    cy += np.random.uniform(-perturb, perturb)
                cells.append([cx, cy, r])
        return np.array(cells[:N])
    
    def random_init():
        x = np.random.uniform(0.1, 0.9, N)
        y = np.random.uniform(0.1, 0.9, N)
        r = np.random.uniform(0.03, 0.15, N)
        return np.column_stack([x, y, r])
    
    # Try different initializations
    inits = []
    # Grid with no perturbation
    inits.append(grid_init(0.0))
    # Grid with small perturbations
    for _ in range(3):
        inits.append(grid_init(0.02))
    # Random inits
    for _ in range(5):
        inits.append(random_init())
    
    for init_idx, init in enumerate(inits):
        if time.time() > deadline:
            break
        
        x0 = init.flatten()
        
        try:
            result = minimize(
                objective, x0,
                method='SLSQP',
                bounds=bounds,
                constraints=constraints,
                options={'maxiter': 200, 'ftol': 1e-10, 'disp': False}
            )
            
            if result.success or result.fun < 0:
                x_opt = result.x
                radii = x_opt[2::3]
                sum_r = np.sum(radii)
                
                # Verify feasibility
                feasible = True
                for i in range(N):
                    xi, yi, ri = x_opt[3*i], x_opt[3*i+1], x_opt[3*i+2]
                    if xi < ri - 1e-6 or xi > 1 - ri + 1e-6:
                        feasible = False
                        break
                    if yi < ri - 1e-6 or yi > 1 - ri + 1e-6:
                        feasible = False
                        break
                if feasible:
                    for i in range(N):
                        for j in range(i+1, N):
                            dx = x_opt[3*i] - x_opt[3*j]
                            dy = x_opt[3*i+1] - x_opt[3*j+1]
                            dist = np.sqrt(dx*dx + dy*dy)
                            if dist < x_opt[3*i+2] + x_opt[3*j+2] - 1e-6:
                                feasible = False
                                break
                        if not feasible:
                            break
                
                if feasible and sum_r > best_sum:
                    best_sum = sum_r
                    best_result = x_opt.copy()
                    
        except Exception:
            continue
    
    if best_result is not None:
        # Format as (x, y, r) array
        result_array = best_result.reshape(N, 3)
        return result_array
    else:
        # Fallback: return grid
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3) for i in range(side) for j in range(side)]
        return np.array(cells[:N])

# EVOLVE-BLOCK-END
