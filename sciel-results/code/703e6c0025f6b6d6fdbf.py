"""
Circle packing in unit square.
Pack 26 circles with disjoint interiors inside [0,1]^2 and maximize the sum of their radii.
"""

import numpy as np
from scipy.optimize import minimize
import time

N = 26
START_TIME = time.time()
TIME_LIMIT = 120.0 # Seconds
SAFETY_MARGIN = 5.0 # Seconds

def construct():
    """
    Returns an array of shape (N, 3) with rows (x, y, radius).
    """
    best_solution = None
    best_score = -np.inf
    
    # Generate diverse initial guesses
    inits = []
    
    # 1. Random Uniform
    for _ in range(3):
        x = np.random.rand(N)
        y = np.random.rand(N)
        r = np.full(N, 0.05) # Start small to avoid immediate huge overlaps
        inits.append(np.column_stack((x, y, r)))
        
    # 2. Jittered Grid
    side = int(np.ceil(np.sqrt(N)))
    grid_x, grid_y = np.meshgrid(np.linspace(0.5/side, 1-0.5/side, side), 
                                  np.linspace(0.5/side, 1-0.5/side, side))
    grid_x = grid_x.flatten()[:N]
    grid_y = grid_y.flatten()[:N]
    r_init = np.full(N, 0.5/side * 0.9)
    jitter = 0.01 * np.random.randn(N)
    grid_x += jitter
    grid_y += jitter
    inits.append(np.column_stack((grid_x, grid_y, r_init)))
    
    # 3. Hexagonal-ish initial guess
    # Create a hex grid pattern
    hex_inits = []
    for offset in [0, 0.1, -0.1]:
        pts = []
        row = 0
        while len(pts) < N:
            for col in range(side + 1):
                x = (col + (0.5 if row % 2 else 0)) * (1.0 / (side * 0.866)) + offset
                y = row * (1.0 / side) + offset
                if 0 <= x <= 1 and 0 <= y <= 1:
                    pts.append((x, y))
            row += 1
        if len(pts) >= N:
            pts = pts[:N]
            x = np.array([p[0] for p in pts])
            y = np.array([p[1] for p in pts])
            r = np.full(N, 0.5 / side * 0.8)
            hex_inits.append(np.column_stack((x, y, r)))
    inits.extend(hex_inits)
    
    # Optimization function
    def objective(params, penalty_factor):
        # params: (N, 3) -> (x, y, r)
        x = params[:, 0]
        y = params[:, 1]
        r = params[:, 2]
        
        # Sum of radii (to be maximized, so we minimize negative)
        sum_r = np.sum(r)
        
        # Boundary violations
        # x < r, x > 1-r, y < r, y > 1-r, r < 0
        bound_pen = 0.0
        bound_pen += np.sum(np.maximum(0, r - x)**2)
        bound_pen += np.sum(np.maximum(0, r - (1 - x))**2)
        bound_pen += np.sum(np.maximum(0, r - y)**2)
        bound_pen += np.sum(np.maximum(0, r - (1 - y))**2)
        bound_pen += np.sum(np.maximum(0, -r)**2)
        
        # Overlap violations
        # Vectorized distance calculation
        # d_ij = sqrt((x_i - x_j)^2 + (y_i - y_j)^2)
        # We need d_ij >= r_i + r_j
        # Violation = max(0, r_i + r_j - d_ij)
        
        # Create distance matrix
        dx = x[:, np.newaxis] - x[np.newaxis, :]
        dy = y[:, np.newaxis] - y[np.newaxis, :]
        d = np.sqrt(dx**2 + dy**2)
        
        # Sum of radii matrix
        r_sum = r[:, np.newaxis] + r[np.newaxis, :]
        
        # Violation matrix
        viol = np.maximum(0, r_sum - d)
        # Zero out diagonal
        np.fill_diagonal(viol, 0)
        
        overlap_pen = np.sum(viol**2)
        
        total_penalty = bound_pen + overlap_pen
        return -sum_r + 0.5 * penalty_factor * total_penalty

    def gradient(params, penalty_factor):
        # Numerical gradient is slow, let's provide analytical or use finite diff in minimize?
        # L-BFGS-B can use finite differences if jac=None, but it's expensive.
        # Let's try to implement analytical gradient for speed.
        
        x = params[:, 0]
        y = params[:, 1]
        r = params[:, 2]
        
        grad_x = np.zeros(N)
        grad_y = np.zeros(N)
        grad_r = -np.ones(N) # Derivative of -sum(r)
        
        # Boundary gradients
        # d/dx max(0, r-x)^2 = 2*max(0, r-x)*(-1)
        v1 = np.maximum(0, r - x)
        v2 = np.maximum(0, r - (1 - x))
        v3 = np.maximum(0, r - y)
        v4 = np.maximum(0, r - (1 - y))
        v5 = np.maximum(0, -r)
        
        grad_x -= 2 * penalty_factor * v1
        grad_x += 2 * penalty_factor * v2
        grad_y -= 2 * penalty_factor * v3
        grad_y += 2 * penalty_factor * v4
        grad_r += 2 * penalty_factor * (v1 + v2 + v3 + v4)
        grad_r -= 2 * penalty_factor * v5
        
        # Overlap gradients
        dx = x[:, np.newaxis] - x[np.newaxis, :]
        dy = y[:, np.newaxis] - y[np.newaxis, :]
        d = np.sqrt(dx**2 + dy**2)
        d[d == 0] = 1e-8 # Avoid division by zero
        
        r_sum = r[:, np.newaxis] + r[np.newaxis, :]
        viol = np.maximum(0, r_sum - d)
        np.fill_diagonal(viol, 0)
        
        # Gradient of 0.5 * mu * sum(viol^2)
        # d/dx_i sum_{j!=i} 0.5 * mu * max(0, r_i+r_j - d_ij)^2
        # If viol_ij > 0:
        # d/dx_i = mu * viol_ij * d/dx_i (r_i+r_j - d_ij)
        # d/dx_i d_ij = -(x_i - x_j) / d_ij = (x_j - x_i) / d_ij
        # So d/dx_i term = mu * viol_ij * (-(x_i - x_j) / d_ij)
        
        # We can compute this via matrix operations
        # G_x[i] = sum_j mu * viol_ij * (-(x_i - x_j) / d_ij)
        # Note: viol is symmetric, d is symmetric.
        
        # Create weight matrix W where W_ij = mu * viol_ij / d_ij if viol_ij > 0 else 0
        W = np.zeros((N, N))
        mask = viol > 0
        W[mask] = penalty_factor * viol[mask] / d[mask]
        
        # grad_x[i] = - sum_j W_ij * (x_i - x_j)
        # This is equivalent to - (W * x)_i + sum_j W_ij * x_j
        # Wait, let's look at the term: sum_j W_ij * (x_j - x_i)
        # = sum_j W_ij * x_j - x_i * sum_j W_ij
        
        grad_x -= np.sum(W * (x[:, np.newaxis] - x[np.newaxis, :]), axis=1)
        grad_y -= np.sum(W * (y[:, np.newaxis] - y[np.newaxis, :]), axis=1)
        
        # grad_r[i] from overlap:
        # d/dr_i sum_j 0.5 * mu * viol_ij^2
        # If viol_ij > 0, d/d r_i = mu * viol_ij * 1
        # So grad_r[i] += mu * sum_j viol_ij
        grad_r += penalty_factor * np.sum(viol, axis=1)
        
        return np.column_stack((grad_x, grad_y, grad_r)).flatten()

    # Run optimization for each init
    for init in inits:
        if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
            break
            
        current_params = init.copy()
        
        # Multi-stage penalty
        penalties = [10, 100, 1000, 5000, 10000]
        
        for mu in penalties:
            if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
                break
                
            try:
                res = minimize(
                    lambda p: objective(p, mu),
                    current_params,
                    jac=lambda p: gradient(p, mu),
                    method='L-BFGS-B',
                    bounds=[(0, 1), (0, 1), (0, 0.5)] * N, # Rough bounds
                    options={'maxiter': 500, 'ftol': 1e-8, 'gtol': 1e-8}
                )
                current_params = res.x
            except Exception:
                pass
        
        # Check validity of the final result
        x = current_params[:, 0]
        y = current_params[:, 1]
        r = current_params[:, 2]
        
        # Simple validity check
        valid = True
        if np.any(x < r) or np.any(x > 1 - r) or np.any(y < r) or np.any(y > 1 - r) or np.any(r < 0):
            valid = False
            
        if valid:
            # Check overlaps
            dx = x[:, np.newaxis] - x[np.newaxis, :]
            dy = y[:, np.newaxis] - y[np.newaxis, :]
            d = np.sqrt(dx**2 + dy**2)
            r_sum = r[:, np.newaxis] + r[np.newaxis, :]
            if np.any(r_sum > d + 1e-7):
                valid = False
        
        if valid:
            score = np.sum(r)
            if score > best_score:
                best_score = score
                best_solution = current_params.copy()
                
    # If no valid solution found, fallback to a safe trivial one
    if best_solution is None:
        # Return a very small grid of circles to ensure validity
        side = int(np.ceil(np.sqrt(N)))
        cells = []
        for i in range(side):
            for j in range(side):
                if len(cells) >= N: break
                x = (i + 0.5) / side
                y = (j + 0.5) / side
                r = 0.5 / side - 1e-4
                cells.append([x, y, r])
        best_solution = np.array(cells[:N])
        
    return best_solution

# Ensure the function is named construct
# The task requires construct() -> array

if __name__ == "__main__":
    # Simple test
    res = construct()
    print(res.shape)
    print("Sum of radii:", np.sum(res[:, 2]))
