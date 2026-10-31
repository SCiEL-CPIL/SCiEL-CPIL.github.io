import numpy as np
from scipy.optimize import minimize
import time

N = 26

def construct():
    start_time = time.time()
    deadline = start_time + 140.0 # 10s margin
    
    best_val = -np.inf
    best_x = None
    
    # Initial guess generation
    # Using a 6x5 grid (30 cells) and picking 26
    # Or 5x6
    rows, cols = 5, 6
    total = rows * cols
    indices = np.arange(total)
    
    # We will do several restarts
    # Each restart:
    # 1. Shuffle indices, pick first N
    # 2. Initialize positions and radii
    # 3. Optimize
    
    for restart in range(20): # Max 20 restarts, time-limited
        if time.time() > deadline:
            break
            
        # Get 26 unique indices from 0..29
        np.random.shuffle(indices)
        selected = indices[:N]
        
        # Initialize positions
        # x = col / cols + 0.5/cols
        # y = row / rows + 0.5/rows
        # Map selected index to row, col
        # index = row * cols + col
        xs = np.zeros(N)
        ys = np.zeros(N)
        rs = np.zeros(N)
        
        for k, idx in enumerate(selected):
            r_idx = idx // cols
            c_idx = idx % cols
            xs[k] = (c_idx + 0.5) / cols
            ys[k] = (r_idx + 0.5) / rows
            rs[k] = 0.04 # Small initial radius
        
        x0 = np.concatenate([xs, ys, rs])
        
        # Define objective and gradient
        def obj_and_grad(x):
            x_coords = x[:N]
            y_coords = x[N:2*N]
            r_coords = x[2*N:3*N]
            
            # Objective: -sum(r)
            val = -np.sum(r_coords)
            grad = np.zeros_like(x)
            grad[2*N:3*N] = -1.0
            
            # Penalty for boundaries
            # x_i >= r_i, x_i <= 1-r_i
            # y_i >= r_i, y_i <= 1-r_i
            
            # Left: x_i - r_i >= 0 -> viol = max(0, r_i - x_i)
            viol_l = np.maximum(0, r_coords - x_coords)
            val += np.sum(viol_l**2)
            grad[:N] += -2 * viol_l # d/dx_i (r_i - x_i)^2 = -2(r_i-x_i)
            grad[2*N:3*N] += 2 * viol_l # d/dr_i (r_i - x_i)^2 = 2(r_i-x_i)
            
            # Right: 1 - x_i - r_i >= 0 -> viol = max(0, x_i + r_i - 1)
            viol_r = np.maximum(0, x_coords + r_coords - 1)
            val += np.sum(viol_r**2)
            grad[:N] += 2 * viol_r
            grad[2*N:3*N] += 2 * viol_r
            
            # Bottom: y_i - r_i >= 0
            viol_b = np.maximum(0, r_coords - y_coords)
            val += np.sum(viol_b**2)
            grad[N:2*N] += -2 * viol_b
            grad[2*N:3*N] += 2 * viol_b
            
            # Top: 1 - y_i - r_i >= 0
            viol_t = np.maximum(0, y_coords + r_coords - 1)
            val += np.sum(viol_t**2)
            grad[N:2*N] += 2 * viol_t
            grad[2*N:3*N] += 2 * viol_t
            
            # Penalty for overlaps
            # d_ij >= r_i + r_j
            # Use vectorized computation for pairs
            # This is O(N^2) which is 26^2 = 676, very fast.
            
            # We can compute all pair distances
            # But for gradient, we need to accumulate.
            
            # Let's iterate pairs for clarity and correctness of gradient accumulation
            # N=26 is small enough
            for i in range(N):
                for j in range(i+1, N):
                    dx = x_coords[i] - x_coords[j]
                    dy = y_coords[i] - y_coords[j]
                    dist = np.sqrt(dx*dx + dy*dy + 1e-12)
                    sum_r = r_coords[i] + r_coords[j]
                    
                    viol = max(0, sum_r - dist)
                    if viol > 0:
                        val += viol**2
                        
                        # Gradient w.r.t x_i, y_i, r_i, x_j, y_j, r_j
                        # d(viol^2)/dx_i = 2*viol * d(sum_r - dist)/dx_i = 2*viol * (-dx/dist)
                        factor = 2 * viol / dist
                        
                        grad[i] += -factor * dx
                        grad[N+i] += -factor * dy
                        grad[2*N+i] += 2 * viol
                        
                        grad[j] += factor * dx
                        grad[N+j] += factor * dy
                        grad[2*N+j] += 2 * viol
                        
            return val, grad
        
        # Bounds
        bounds = []
        for i in range(N):
            bounds.append((0, 1)) # x
        for i in range(N):
            bounds.append((0, 1)) # y
        for i in range(N):
            bounds.append((0, 0.5)) # r
            
        # Optimize
        try:
            res = minimize(
                obj_and_grad, 
                x0, 
                method='L-BFGS-B', 
                jac=True, 
                bounds=bounds, 
                options={'maxiter': 1000, 'ftol': 1e-12, 'gtol': 1e-12}
            )
            
            # Check result
            x_res = res.x
            # Calculate penalty to ensure validity
            # If penalty is small, it's a valid packing
            # The objective function returns -sum(r) + penalty
            # We want to maximize sum(r) -> minimize -sum(r)
            # If penalty is 0, obj = -sum(r). So sum(r) = -obj.
            
            # Let's re-evaluate penalty separately to be sure
            x_c = x_res[:N]
            y_c = x_res[N:2*N]
            r_c = x_res[2*N:3*N]
            
            # Check bounds
            if np.any(x_c < r_c - 1e-6) or np.any(x_c > 1 - r_c + 1e-6) or \
               np.any(y_c < r_c - 1e-6) or np.any(y_c > 1 - r_c + 1e-6):
                 # Invalid
                 continue
                 
            # Check overlaps
            valid = True
            for i in range(N):
                for j in range(i+1, N):
                    d = np.sqrt((x_c[i]-x_c[j])**2 + (y_c[i]-y_c[j])**2)
                    if d < r_c[i] + r_c[j] - 1e-6:
                        valid = False
                        break
                if not valid: break
            
            if valid:
                val = np.sum(r_c)
                if val > best_val:
                    best_val = val
                    best_x = x_res
                    
        except Exception as e:
            pass # Skip failed optimization
            
    if best_x is None:
        # Fallback to seed-like solution
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3) for i in range(side) for j in range(side)]
        return np.array(cells[:N])
        
    # Reshape to (N, 3)
    x_out = best_x[:N]
    y_out = best_x[N:2*N]
    r_out = best_x[2*N:3*N]
    
    return np.column_stack((x_out, y_out, r_out))
