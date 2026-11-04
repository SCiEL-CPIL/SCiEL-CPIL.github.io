"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 21

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time

def construct():
    t_start = time.time()
    deadline = 130.0  # safety margin
    
    def hex_init(W, H, n_circles):
        """Initialize circles in a hexagonal-like pattern."""
        positions = []
        radii = []
        # Estimate radius
        # For hexagonal packing in W x H rectangle
        # Try to fit n_circles
        r_est = 0.15  # initial guess
        rows = int(H / (r_est * 2)) + 1
        cols = int(W / (r_est * 2)) + 1
        
        # Generate hexagonal grid positions
        pos = []
        for row in range(rows + 5):
            for col in range(cols + 5):
                x = col * r_est * 2 * 0.866
                y = row * r_est * 2 * 0.866
                if row % 2 == 1:
                    x += r_est * 2 * 0.866 / 2
                pos.append((x, y))
        
        # Select n_circles positions closest to center
        cx, cy = W / 2, H / 2
        pos_sorted = sorted(pos, key=lambda p: (p[0] - cx)**2 + (p[1] - cy)**2)
        selected = pos_sorted[:n_circles]
        
        return np.array(selected), np.full(n_circles, r_est)
    
    def objective_and_constraints(x, W, H):
        """x = [x_1, y_1, r_1, x_2, y_2, r_2, ...]"""
        n = len(x) // 3
        positions = x.reshape(n, 3)
        
        # Objective: maximize sum of radii -> minimize negative sum
        obj = -np.sum(positions[:, 2])
        
        # Constraints (all >= 0)
        cons = []
        
        # Boundary constraints: r_i <= x_i <= W - r_i, r_i <= y_i <= H - r_i
        for i in range(n):
            xi, yi, ri = positions[i]
            cons.append({'type': 'ineq', 'fun': lambda xi=xi, ri=ri: xi - ri})
            cons.append({'type': 'ineq', 'fun': lambda xi=xi, W=W, ri=ri: W - xi - ri})
            cons.append({'type': 'ineq', 'fun': lambda yi=yi, ri=ri: yi - ri})
            cons.append({'type': 'ineq', 'fun': lambda yi=yi, H=H, ri=ri: H - yi - ri})
            cons.append({'type': 'ineq', 'fun': lambda ri=ri: ri - 1e-6})
        
        # Non-overlap: (xi-xj)^2 + (yi-yj)^2 >= (ri+rj)^2
        for i in range(n):
            for j in range(i+1, n):
                xi, yi, ri = positions[i]
                xj, yj, rj = positions[j]
                cons.append({
                    'type': 'ineq',
                    'fun': lambda xi=xi, yi=yi, ri=ri, xj=xj, yj=yj, rj=rj: 
                        (xi - xj)**2 + (yi - yj)**2 - (ri + rj)**2
                })
        
        return obj, cons
    
    def solve_for_rect(W, H):
        """Solve for given rectangle dimensions."""
        # Initialize
        x0_pos, x0_r = hex_init(W, H, N)
        x0 = np.column_stack([x0_pos, x0_r]).flatten()
        
        # Objective function
        def obj(x):
            n = len(x) // 3
            positions = x.reshape(n, 3)
            return -np.sum(positions[:, 2])
        
        # Constraints
        def make_constraints(W, H):
            cons = []
            # Boundary
            for i in range(N):
                cons.append({'type': 'ineq', 'fun': lambda x, i=i, W=W: x[3*i] - x[3*i+2]})
                cons.append({'type': 'ineq', 'fun': lambda x, i=i, W=W: W - x[3*i] - x[3*i+2]})
                cons.append({'type': 'ineq', 'fun': lambda x, i=i, H=H: x[3*i+1] - x[3*i+2]})
                cons.append({'type': 'ineq', 'fun': lambda x, i=i, H=H: H - x[3*i+1] - x[3*i+2]})
                cons.append({'type': 'ineq', 'fun': lambda x, i=i: x[3*i+2] - 1e-6})
            
            # Non-overlap
            for i in range(N):
                for j in range(i+1, N):
                    cons.append({
                        'type': 'ineq',
                        'fun': lambda x, i=i, j=j: 
                            (x[3*i] - x[3*j])**2 + (x[3*i+1] - x[3*j+1])**2 - (x[3*i+2] + x[3*j+2])**2
                    })
            
            return cons
        
        constraints = make_constraints(W, H)
        
        # Bounds: positions in [0, W] x [0, H], radii in [0, min(W,H)/2]
        r_max = min(W, H) / 2
        bounds = []
        for i in range(N):
            bounds.append((0, W))
            bounds.append((0, H))
            bounds.append((1e-6, r_max))
        
        result = minimize(
            obj, x0, method='SLSQP',
            bounds=bounds,
            constraints=constraints,
            options={'maxiter': 200, 'ftol': 1e-8}
        )
        
        if result.success or result.fun < 0:
            positions = result.x.reshape(N, 3)
            sum_r = np.sum(positions[:, 2])
            return positions, sum_r
        return None, -np.inf
    
    # Try different aspect ratios
    best_positions = None
    best_sum = -np.inf
    
    aspect_ratios = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.7, 2.0]
    
    for ar in aspect_ratios:
        if time.time() > deadline:
            break
        
        # W + H = 2, W/H = ar -> W = 2*ar/(1+ar), H = 2/(1+ar)
        W = 2 * ar / (1 + ar)
        H = 2 / (1 + ar)
        
        positions, sum_r = solve_for_rect(W, H)
        
        if positions is not None and sum_r > best_sum:
            best_sum = sum_r
            best_positions = positions.copy()
    
    # If optimization failed, fall back to grid
    if best_positions is None or best_sum <= 0:
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3) for i in range(side) for j in range(side)]
        return np.array(cells[:N])
    
    # Ensure validity: clip positions to be within bounds
    # Find the rectangle that was used
    # Re-determine W, H from best aspect ratio
    # Actually, let's just return the positions with the rectangle implied
    # The evaluator checks perimeter of bounding rectangle
    
    # Verify perimeter
    min_x = np.min(best_positions[:, 0])
    max_x = np.max(best_positions[:, 0])
    min_y = np.min(best_positions[:, 1])
    max_y = np.max(best_positions[:, 1])
    
    # Shift to origin
    best_positions[:, 0] -= min_x
    best_positions[:, 1] -= min_y
    
    W_final = max_x - min_x
    H_final = max_y - min_y
    perimeter = 2 * (W_final + H_final)
    
    # If perimeter > 4, scale down
    if perimeter > 4:
        scale = 4.0 / perimeter
        best_positions *= scale
    
    return best_positions
# EVOLVE-BLOCK-END
