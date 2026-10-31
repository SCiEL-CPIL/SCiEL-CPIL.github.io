"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 21
import numpy as np
import time
from scipy.optimize import minimize

# EVOLVE-BLOCK-START

def construct():
    start_time = time.time()
    deadline = 145.0  # Safety margin before 150s kill
    
    # Initial grid setup in a 1x1 square
    side = int(np.ceil(np.sqrt(N)))
    cells = []
    for i in range(side):
        for j in range(side):
            if len(cells) < N:
                x = (i + 0.5) / side
                y = (j + 0.5) / side
                r = 0.5 / side - 1e-3
                cells.append([x, y, r])
    
    # Pad if N is not a perfect square? No, we just take the first N.
    # Ensure we have exactly N circles.
    while len(cells) < N:
        # Add small circles in corners if needed, but grid covers it.
        cells.append([0.1, 0.1, 0.01])
    
    positions = np.array([c[0:2] for c in cells])
    radii = np.array([c[2] for c in cells])
    
    W = 1.0
    H = 1.0
    
    # Objective function to minimize: sum of squared violations
    def penalty(params):
        pos = params[:2*N].reshape(N, 2)
        rad = params[2*N:]
        
        # Wall violations
        pen = 0.0
        for i in range(N):
            x, y = pos[i]
            r = rad[i]
            # x < r
            if x < r:
                pen += (r - x)**2
            # x > W - r
            if x > W - r:
                pen += (x - (W - r))**2
            # y < r
            if y < r:
                pen += (r - y)**2
            # y > H - r
            if y > H - r:
                pen += (y - (H - r))**2
        
        # Circle-circle overlaps
        for i in range(N):
            for j in range(i+1, N):
                dist = np.linalg.norm(pos[i] - pos[j])
                min_dist = rad[i] + rad[j]
                if dist < min_dist:
                    pen += (min_dist - dist)**2
        
        return pen

    def get_params(pos, rad):
        return np.concatenate([pos.flatten(), rad])

    # Initial parameters
    current_params = get_params(positions, radii)
    
    # Check initial validity
    current_pen = penalty(current_params)
    
    best_params = current_params.copy()
    best_value = np.sum(radii)
    
    # Optimization loop: Try to increase radii
    # We will scale radii by a factor and try to fix positions
    scale = 1.0
    while time.time() < deadline:
        # Try to expand
        scale *= 1.01
        new_rad = radii * scale
        
        # If any radius becomes too large for the box, stop
        if np.max(new_rad) > 0.5: 
            break
            
        # Set up optimization problem
        # We want to find positions such that penalty is minimized
        # Initial guess: current positions
        x0 = get_params(positions, new_rad)
        
        # Use L-BFGS-B or Nelder-Mead? L-BFGS-B is faster for smooth functions.
        # However, the penalty function is non-smooth at the boundary of violations.
        # We can smooth it slightly or just use Nelder-Mead which handles non-smooth better.
        # Given N=21, 63 variables, Nelder-Mead might be slow.
        # Let's use a simple gradient descent on the squared penalty.
        
        # Actually, let's use scipy.optimize.minimize with method='L-BFGS-B' 
        # and bound constraints to keep circles roughly inside.
        # Bounds: x in [0, W], y in [0, H], r in [0, 0.5]
        bounds = []
        for _ in range(N):
            bounds.append((0.0, W))
            bounds.append((0.0, H))
        for _ in range(N):
            bounds.append((0.0, 0.5))
            
        # Run optimization
        try:
            res = minimize(penalty, x0, method='L-BFGS-B', bounds=bounds, 
                           options={'maxiter': 500, 'ftol': 1e-12})
            new_params = res.x
            new_pen = res.fun
        except Exception:
            break
            
        if new_pen < 1e-6:
            # Valid configuration found
            positions = new_params[:2*N].reshape(N, 2)
            radii = new_params[2*N:]
            current_params = new_params
            best_params = new_params
            best_value = np.sum(radii)
        else:
            # Failed to fix overlaps, revert scale
            break
            
    # Final polish: Run a few more iterations with tighter tolerance if time allows
    if time.time() < deadline - 10:
        try:
            res = minimize(penalty, best_params, method='L-BFGS-B', bounds=bounds, 
                           options={'maxiter': 1000, 'ftol': 1e-15})
            if res.fun < 1e-9:
                best_params = res.x
                best_value = np.sum(best_params[2*N:])
        except Exception:
            pass

    # Construct final array
    final_params = best_params
    pos = final_params[:2*N].reshape(N, 2)
    rad = final_params[2*N:]
    
    result = np.hstack([pos, rad[:, np.newaxis]])
    
    return result

# EVOLVE-BLOCK-END
