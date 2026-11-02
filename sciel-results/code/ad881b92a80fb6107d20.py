"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
from scipy.optimize import minimize

N = 21

# EVOLVE-BLOCK-START

def _constraints_pack(w, x, y, r):
    """Constraints for circle packing:
    1. Circles inside rectangle: x_i - r_i >= 0, y_i - r_i >= 0,
       x_i + r_i <= w, y_i + r_i <= h where h = 2 - w (perimeter 4 => 2(w+h)=4 => h=2-w)
    2. Non-overlapping: (x_i - x_j)^2 + (y_i - y_j)^2 >= (r_i + r_j)^2
    """
    h = 2.0 - w
    cons = []
    # Bound constraints
    for i in range(N):
        cons.append({'type': 'ineq', 'fun': lambda i=i, w=w, x=x, y=y, r=r: x[i] - r[i]})
        cons.append({'type': 'ineq', 'fun': lambda i=i, w=w, x=x, y=y, r=r: y[i] - r[i]})
        cons.append({'type': 'ineq', 'fun': lambda i=i, w=w, x=x, y=y, r=r: w - x[i] - r[i]})
        cons.append({'type': 'ineq', 'fun': lambda i=i, w=w, x=x, y=y, r=r: h - y[i] - r[i]})
    # Overlap constraints
    for i in range(N):
        for j in range(i + 1, N):
            cons.append({
                'type': 'ineq',
                'fun': lambda i=i, j=j, w=w, x=x, y=y, r=r:
                    (x[i] - x[j])**2 + (y[i] - y[j])**2 - (r[i] + r[j])**2
            })
    return cons


def _objective(params):
    """Negative sum of radii (to minimize)."""
    w = params[0]
    x = params[1:1 + N]
    y = params[1 + N:1 + 2 * N]
    r = params[1 + 2 * N:1 + 3 * N]
    return -np.sum(r)


def _initial_grid(w):
    """Create initial grid placement for N circles in rectangle w x (2-w)."""
    h = 2.0 - w
    # Use a grid layout
    cols = int(np.ceil(np.sqrt(N * w / h))) if h > 0 else 1
    cols = max(1, cols)
    rows = int(np.ceil(N / cols))
    
    x = np.zeros(N)
    y = np.zeros(N)
    r = np.zeros(N)
    
    for i in range(N):
        col = i % cols
        row = i // cols
        x[i] = (col + 0.5) * w / cols
        y[i] = (row + 0.5) * h / rows
        r[i] = min(w / cols, h / rows) * 0.49
    
    return x, y, r


def construct():
    start_time = time.time()
    deadline = start_time + 130  # 20s safety margin
    
    best_value = 0.0
    best_params = None
    
    # Try several initial aspect ratios
    aspect_ratios = [0.5, 0.6, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0]
    
    for w0 in aspect_ratios:
        if time.time() > deadline:
            break
        
        x0, y0, r0 = _initial_grid(w0)
        
        # Initial parameters
        params0 = np.concatenate([[w0], x0, y0, r0])
        
        # Bounds
        bounds = [(w0 * 0.5, w0 * 1.5)]  # w
        for i in range(N):
            bounds.append((0.01, w0))       # x
        for i in range(N):
            bounds.append((0.01, 2.0 - w0)) # y
        for i in range(N):
            bounds.append((0.01, min(w0, 2.0 - w0) * 0.49))  # r
        
        # Constraints
        cons = _constraints_pack(w0, x0, y0, r0)
        
        try:
            result = minimize(
                _objective,
                params0,
                method='SLSQP',
                bounds=bounds,
                constraints=cons,
                options={'maxiter': 200, 'ftol': 1e-8}
            )
            
            if result.success or result.fun < 0:
                val = -result.fun
                if val > best_value:
                    best_value = val
                    best_params = result.x.copy()
                    
        except Exception:
            continue
        
        if time.time() > deadline:
            break
    
    # If optimization failed, fall back to grid
    if best_params is None:
        w = 1.0
        x, y, r = _initial_grid(w)
        # Scale down to ensure validity
        h = 2.0 - w
        scale = 0.49
        for i in range(N):
            x[i] = max(r[i], min(w - r[i], x[i]))
            y[i] = max(r[i], min(h - r[i], y[i]))
        best_value = np.sum(r)
        return np.column_stack([x, y, r])
    
    # Extract best result
    w = best_params[0]
    x = best_params[1:1 + N]
    y = best_params[1 + N:1 + 2 * N]
    r = best_params[1 + 2 * N:1 + 3 * N]
    
    # Clip to ensure validity
    h = 2.0 - w
    x = np.clip(x, r, w - r)
    y = np.clip(y, r, h - r)
    
    # Verify and adjust if needed
    for i in range(N):
        for j in range(i + 1, N):
            dist = np.sqrt((x[i] - x[j])**2 + (y[i] - y[j])**2)
            min_dist = r[i] + r[j]
            if dist < min_dist:
                # Shrink the smaller circle
                if r[i] < r[j]:
                    r[i] = max(0.001, (dist - r[j]) / 2)
                else:
                    r[j] = max(0.001, (dist - r[i]) / 2)
    
    # Re-clip after adjustment
    x = np.clip(x, r, w - r)
    y = np.clip(y, r, h - r)
    
    return np.column_stack([x, y, r])

# EVOLVE-BLOCK-END
